import json
from pathlib import Path
from unittest.mock import patch
import unittest

from test_distillation_loop import ControllerTests, FakeRuntime, SCHEMAS, report


def scored(value, *, validity=1, losses=None, incomplete=(), claims=(), missed=()):
    rows = {}
    losses = losses or {}
    for task in SCHEMAS:
        for i in range(20):
            key = f'{task}/{i}'
            rows[key] = dict(task=task, score=value - (min(value, 1 if value == 1 else .5)
                                                       if i < losses.get(task, 0) else 0),
                             valid=i < 19 if validity == .95 else True,
                             incomplete=key in incomplete, unsupported_claims=key in claims,
                             missed_required_review=key in missed)
    result = report(value, validity)
    result['case_scores'] = rows
    result['examples_per_task'] = dict.fromkeys(SCHEMAS, 20)
    result['per_task'] = {task: sum(row['score'] for row in rows.values() if row['task'] == task) / 20
                          for task in SCHEMAS}
    result['macro_task_score'] = sum(result['per_task'].values()) / len(SCHEMAS)
    result['minimum_task_score'] = min(result['per_task'].values())
    return result


class CorrectiveGateTests(unittest.TestCase):
    def test_high_uncertainty_and_9472_validity_can_allow_an_experiment(self):
        from distillation_loop.quality import diagnostic_decision
        before, after = scored(.6, validity=.95), scored(.62, validity=.95)
        # The saved Pass A figures are 345/360 and 341/360.
        for summary, count in ((before, 345), (after, 341)):
            summary['schema_evidence_valid_rate'] = count / 360
            for i, row in enumerate(summary['case_scores'].values()):
                row['valid'] = i < count
        comparison = {'delta': .02, 'bootstrap_95pct_ci': [-.03, .06],
                      'conservative_95pct_ci': [-.12, .16]}
        decision = diagnostic_decision(before, after, comparison, {})
        self.assertEqual(decision['outcome'], 'inconclusive')
        self.assertTrue(decision['experimental_continuation_eligible'])
        self.assertFalse(decision['quality_target_met'])

    def test_task_and_safety_events_reject_despite_inconclusive_sign_test(self):
        from distillation_loop.quality import operational_gate, regression_gate
        task = next(iter(SCHEMAS))
        before, after = scored(1), scored(1, losses={task: 3},
                                        claims=[f'{task}/4'], missed=[f'{task}/5'])
        stats = regression_gate(before, after, {'paired_regression_alpha': .05})
        self.assertTrue(stats['accepted'])
        gate = operational_gate(before, after, {})
        self.assertFalse(gate['accepted'])
        reasons = {row['code']: row['cases'] for row in gate['reasons']}
        self.assertEqual(set(reasons['single_task_regression']), {f'{task}/{i}' for i in range(3)})
        self.assertEqual(reasons['new_unsupported_claim'], [f'{task}/4'])
        self.assertEqual(reasons['new_missed_required_review'], [f'{task}/5'])

    def test_multiple_task_deterioration_and_duplicate_stop_labels(self):
        from distillation_loop.evaluation import incomplete_prediction
        from distillation_loop.quality import operational_gate
        tasks = list(SCHEMAS)[:4]
        baseline = scored(1)
        stops = [f'{tasks[0]}/{i}' for i in range(11)]
        candidate = scored(1, losses=dict.fromkeys(tasks, 2), incomplete=stops)
        decision = operational_gate(baseline, candidate, {})
        reasons = {row['code']: row for row in decision['reasons']}
        self.assertEqual(set(reasons), {'multiple_task_regression', 'incomplete_generation'})
        self.assertAlmostEqual(reasons['incomplete_generation']['rate'], 11 / 360)
        self.assertTrue(incomplete_prediction({'truncated': True, 'stop_reason': 'repetition'}))
        self.assertTrue(incomplete_prediction({'truncated': False, 'stop_reason': 'token_limit'}))

    def test_bootstrap_is_repeatable_and_paired_within_tasks(self):
        from distillation_loop.paired import paired_summary
        changes = [1, 0, -1, 0] * 10
        report = paired_summary(changes, strata=['one'] * 20 + ['two'] * 20)
        self.assertEqual((report['gains'], report['losses'], report['ties']), (10, 10, 20))
        self.assertEqual(report, paired_summary(changes, strata=['one'] * 20 + ['two'] * 20))
        self.assertLess(report['bootstrap_95pct_ci'][0], 0)
        self.assertGreater(report['bootstrap_95pct_ci'][1], 0)


def test_rollback(self):
    self.config['max_rounds'] = 5
    self.config['evaluation']['semantic_judge'] = True
    self.config['quality']['paired_regression_alpha'] = .05
    task = next(iter(SCHEMAS))
    reports = dict(baseline=scored(.5), round_01=scored(.7), round_02=scored(.65),
                   round_03=scored(.72, validity=.95), round_04=scored(.77, losses={task: 11}),
                   round_05=scored(.75))
    starting_adapters, evaluations = [], []

    class Runtime(FakeRuntime):
        def train(self, pairs, schemas, output):
            starting_adapters.append(self.restores[-1])
            return super().train(pairs, schemas, output)

    class Evaluator:
        def evaluate(self, _, output):
            key = output.parent.name if output.name == 'lighteval' else output.name
            evaluations.append(key)
            return reports[key]

    from distillation_loop import controller
    save = controller.atomic_json
    crashed = False

    def interrupt(path, data):
        nonlocal crashed
        if path.name == 'loop_state.json' and data['completed_rounds'] == 2 and not crashed:
            crashed = True
            raise RuntimeError('interrupted promotion commit')
        save(path, data)

    runtime, evaluator = Runtime(), Evaluator()
    with patch('self_instruct_pipeline.parent_backend.create_parent'),             patch('distillation_loop.controller.atomic_json', side_effect=interrupt):
        with self.assertRaisesRegex(RuntimeError, 'interrupted promotion commit'):
            self.run_loop(runtime, evaluator)
        saved = json.loads((Path(self.tmp.name) / 'loop_state.json').read_text())
        self.assertEqual(saved['completed_rounds'], 1)
        result = self.run_loop(runtime, evaluator)
    best = result['history'][0]['checkpoint']
    self.assertEqual(starting_adapters, [result['initial_checkpoint']] + [best] * 4)
    self.assertEqual(evaluations.count('round_02'), 2)
    self.assertEqual([row['selection']['promoted'] for row in result['history']], [True, False, False, False, True])
    self.assertEqual(result['history'][3]['selection']['comparator_checkpoint'], best)
    self.assertEqual(result['next_round_start_checkpoint'], result['latest_checkpoint'])
    self.assertNotEqual(result['initial_checkpoint'], result['next_round_start_checkpoint'])


# Reuse the existing controller harness without collecting its whole suite twice.
class CorrectiveControllerTests(unittest.TestCase):
    setUp = ControllerTests.setUp
    run_loop = ControllerTests.run_loop
    test_resume_and_rejections_use_current_best = test_rollback

del ControllerTests


class EvidenceTests(unittest.TestCase):
    def test_old_student_predictions_bind_the_exposed_record(self):
        from tempfile import TemporaryDirectory
        from distillation_loop.leakage import ExclusionRegistry, historical_exposures
        from self_instruct_pipeline.persistence import atomic_jsonl, digest
        from test_distillation_loop import record
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'round_02').mkdir()
            (root / 'self_instruct').mkdir()
            original = record(i=505)
            roster = root / 'self_instruct/accepted_scenarios.jsonl'
            atomic_jsonl(roster, [original])
            atomic_jsonl(root / 'round_02/student_responses.jsonl',
                         [{'scenario_id': original['id'], 'scenario_hash': digest(original)}])
            exposed = historical_exposures([root], [])
            self.assertEqual(ExclusionRegistry(exposed).match({**original, 'id': 'resampled'}), 'canonical_content')
            roster.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing historical scenario'):
                historical_exposures([root], [])

    def test_repeat_judge_results_cannot_be_reused_between_arms(self):
        from distillation_loop.judge_reproducibility import analyze
        arms = ['medium-1', 'medium-2']
        plan = {'arms': arms, 'experiment': {'checkpoint': 'sft'},
                'targets': {'one': {'task': 'example', 'keys': dict(zip(arms, ['a', 'b'])),
                                    'review_flag_correct': True}}}
        judgment = {'decision_correct': True, 'facts_correct': True,
                    'abstention_correct': True, 'unsupported_claims': False}
        row = {'scenario_id': 'one', 'cost_usd': '.01', 'latency_seconds': 2, 'judgment': judgment}
        first = {**row, 'arm': 'medium-1', 'cache_key': 'a'}
        repeat = {**row, 'arm': 'medium-2', 'cache_key': 'b',
                  'judgment': {**judgment, 'decision_correct': False}}
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            analyze(plan, [first, {**repeat, 'cache_key': 'a'}])
        report = analyze(plan, [first, repeat])
        comparison = report['comparisons']['medium-2 vs medium-1']['example']
        self.assertEqual(comparison['decision_correct_flips'], 1)
        self.assertEqual(comparison['task_score_change'], -1)
