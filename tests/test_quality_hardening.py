import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from distillation_loop.core import clean_response, score_response, load_loop_config, ResponseError
from distillation_loop.evaluation import LightevalEvaluator
from distillation_loop.quality import regression_gate, promotion_gate
from distillation_loop.review import grade_reviewed, review_key, ReviewRequired, DIMENSIONS
from distillation_loop.smoke import smoke_configs
from distillation_loop.trusted_templates import template_pair, template_pairs
from self_instruct_pipeline.rule_checks import check_rules
from self_instruct_pipeline.config import load_config
from self_instruct_pipeline.parent_backend import ParentError
from self_instruct_pipeline.persistence import atomic_jsonl, read_jsonl
from self_instruct_pipeline.pipeline import Pipeline
import test_distillation_loop as fixtures
from test_self_instruct import MockParent

SCHEMAS = fixtures.SCHEMAS


def reasoning_item():
    r = fixtures.record('policy_coverage_qa')
    reference = {'answer': 'Prior authorization is required.', 'evidence': [{'document_id': 'D1', 'quote': 'Synthetic hospital invoice'}],
                 'requires_review': False, 'missing_information': []}
    return {'record': r, 'reference': reference, 'review': {'status': 'approved', 'reviewer': 'fixture-reviewer', 'source_group': 'fixture'},
            'rubric': {'decision': 'authorization_required', 'required_facts': ['Authorization precedes submission.']}}


class QualityHardeningTests(unittest.TestCase):
    def test_classification_flag_and_coding_prose_are_not_ignored(self):
        r = fixtures.record()
        ref = {'document_type': 'hospital_bill', 'evidence': ['Synthetic hospital invoice'], 'requires_review': False}
        self.assertEqual(score_response(r, ref, json.dumps(dict(ref, requires_review=True)), SCHEMAS)['task_score'], 0)
        item = reasoning_item()
        item['record']['task'] = 'clinical_coding'
        item['reference']['suggested_codes'] = []
        wrong = dict(item['reference'], answer='Authorization is not required.')
        self.assertEqual(score_response(item['record'], item['reference'], json.dumps(wrong), SCHEMAS)['task_score'], 0)

    def test_reviewed_paraphrase_and_contradiction_adjudication(self):
        item = reasoning_item()
        paraphrase = dict(item['reference'], answer='Obtain authorization before submission.')
        prediction = {'raw': json.dumps(paraphrase), 'truncated': False}
        scores, request = grade_reviewed(item, prediction, SCHEMAS, {})
        self.assertIsNone(scores)
        self.assertEqual(request['rubric']['decision'], 'authorization_required')
        judgment = {'reviewer': 'independent-fixture', 'reason': 'Same supported decision.',
                    'decision_correct': True, 'facts_correct': True, 'abstention_correct': True, 'unsupported_claims': False}
        scores, request = grade_reviewed(item, prediction, SCHEMAS, {review_key(item, prediction): judgment})
        self.assertEqual(scores['task_score'], 1)
        judgment['unsupported_claims'] = True
        self.assertEqual(grade_reviewed(item, prediction, SCHEMAS, {review_key(item, prediction): judgment})[0]['task_score'], 0)
        item['accepted_references'] = [paraphrase]
        self.assertEqual(grade_reviewed(item, prediction, SCHEMAS, {})[0]['task_score'], 1)
        prediction['raw'] = json.dumps(dict(paraphrase, answer='Authorization is not required.'))
        self.assertIsNone(grade_reviewed(item, prediction, SCHEMAS, {})[0])

    def test_frozen_predictions_resume_after_review_without_regeneration(self):
        item = reasoning_item()
        answer = dict(item['reference'], answer='Obtain authorization before submission.')
        class Runtime(fixtures.FakeRuntime):
            def infer(self, payload):
                self.inferences.append(payload)
                return {'raw': json.dumps(answer), 'truncated': False}
        runtime = Runtime()
        evaluator = LightevalEvaluator(runtime, {'policy_coverage_qa': SCHEMAS['policy_coverage_qa']}, 1)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with self.assertRaises(ReviewRequired): evaluator.evaluate([item], output)
            queue = read_jsonl(output / 'review_queue.jsonl')
            self.assertEqual(len(queue), 1)
            judgment = {'review_key': queue[0]['review_key'], 'reviewer': 'fixture', 'reason': 'Equivalent supported meaning.',
                        'decision_correct': True, 'facts_correct': True, 'abstention_correct': True, 'unsupported_claims': False}
            atomic_jsonl(output / 'adjudications.jsonl', [judgment])
            report = evaluator.evaluate([item], output)
            self.assertEqual(report['macro_task_score'], 1)
            self.assertEqual(len(runtime.inferences), 1)
            self.assertEqual(read_jsonl(output / 'review_queue.jsonl'), [])
            changed = copy.deepcopy(item); changed['rubric']['decision'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'inputs changed'): evaluator.evaluate([changed], output)
            atomic_jsonl(output / 'adjudications.jsonl', [judgment, judgment])
            with self.assertRaisesRegex(ValueError, 'Duplicate'): evaluator.evaluate([item], output)

    def test_rule_malformed_inputs_fail_closed(self):
        for operation, operands, actual in [('equals', [True], 1), ('sum', [], '0'), ('sum', ['NaN'], 'NaN'),
                                            ('all', [True, 1], True), ('any', [False], 0), ('date_offset', ['2025-01-01', 1.5], '2025-01-02'),
                                            ('equals', [1, 2], 1)]:
            with self.subTest(operation=operation, operands=operands), self.assertRaises((ValueError, ArithmeticError)):
                check_rules({'answer_checks': [{'path': ['value'], 'operation': operation, 'operands': operands}]}, {'value': actual})
        for raw in ('{"value":NaN}', '{"value":Infinity}'):
            with self.assertRaises(ResponseError): clean_response(raw)
        check_rules({'answer_checks': [{'path': ['value'], 'operation': 'date_offset', 'operands': ['2024-02-28', 1]}]}, {'value': '2024-02-29'})

    def test_templates_are_valid_computed_and_mutation_rejected(self):
        pairs = template_pairs(90, SCHEMAS, [])
        self.assertEqual(len({r['id'] for r, _ in pairs}), 90)
        from self_instruct_pipeline.validation import validate_answer, ValidationError
        for r, answer in pairs:
            validate_answer(r, answer, SCHEMAS[r['task']])
            with self.assertRaises(ValidationError): validate_answer(r, dict(answer, answer='Wrong conclusion'), SCHEMAS[r['task']])
        first = pairs[0][0]
        self.assertNotIn(first['id'], {r['id'] for r, _ in template_pairs(1, SCHEMAS, [first])})

    def test_regression_blocks_selection_even_with_better_macro(self):
        baseline = fixtures.report(.5)
        candidate = fixtures.report(.8)
        candidate['per_task']['medical_necessity_support'] = .4
        self.assertFalse(regression_gate(baseline, candidate, {})['accepted'])
        settings = {'per_task_max_regression': {'medical_necessity_support': .1}}
        self.assertTrue(regression_gate(baseline, candidate, settings)['accepted'])
        self.assertFalse(promotion_gate(baseline, baseline, {}, load_loop_config()['thresholds'], SCHEMAS)['accepted'])

    def test_smoke_explicitly_uses_smoke_mode(self):
        _, cfg = smoke_configs(load_config(), load_loop_config(), '/tmp/smoke-fixture')
        self.assertEqual(cfg['evaluation']['mode'], 'fixed')
        self.assertEqual(cfg['quality']['minimum_deterministic_check_rate'], 0)

    def test_service_failure_does_not_exhaust_sample_quality_allowance(self):
        class Transient(MockParent):
            def generate_json(self, messages, schema, operation='generation', temperature=None):
                if operation == 'teacher_answer': raise ParentError('network failure', True)
                return super().generate_json(messages, schema, operation, temperature)
        with tempfile.TemporaryDirectory() as folder:
            cfg = load_config(); cfg['paths']['output_dir'] = folder
            cfg['generation'].update(target_accepted_scenarios=1, batch_size=1)
            pipeline = Pipeline(cfg, Transient()); pipeline.generate()
            with self.assertRaises(ParentError): pipeline.answer(allow_partial=True, max_sample_attempts=1)
            resumed = Pipeline(cfg, MockParent())
            self.assertEqual(resumed.answer(allow_partial=True, max_sample_attempts=1), 1)
            self.assertEqual(len(resumed.store.read('answer_transport_failures.jsonl')), 1)

class ControllerHardeningTests(unittest.TestCase):
    def fixture(self):
        fixture = fixtures.ControllerTests(); fixture.setUp()
        self.addCleanup(fixture.tmp.cleanup)
        fixture.config['max_rounds'] = 1
        return fixture

    def test_failed_review_resume_keeps_round_inputs_and_does_not_retrain(self):
        f = self.fixture(); runtime = fixtures.FakeRuntime()
        evaluator = fixtures.FakeEvaluator([.1, .95], fail_at=2)
        with self.assertRaises(RuntimeError): f.run_loop(runtime, evaluator)
        with patch.object(fixtures.FakeCurriculum, 'verified_pairs', side_effect=AssertionError('must not regenerate frozen round')):
            result = f.run_loop(runtime, evaluator)
        self.assertEqual(len(runtime.trains), 1)
        self.assertEqual(result['completed_rounds'], 1)

    def test_tampered_training_plan_fails_before_retraining(self):
        f = self.fixture(); runtime = fixtures.FakeRuntime()
        evaluator = fixtures.FakeEvaluator([.1, .95], fail_at=2)
        with self.assertRaises(RuntimeError): f.run_loop(runtime, evaluator)
        path = Path(f.tmp.name) / 'round_01/training_plan.json'
        plan = json.loads(path.read_text()); plan['pairs'][0]['record']['instruction'] = 'modified'
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, 'training pairs have changed'): f.run_loop(runtime, evaluator)
        self.assertEqual(len(runtime.trains), 1)

    def test_shortfall_blocks_full_target_default(self):
        f = self.fixture(); f.config['quality']['minimum_new_pair_fraction'] = 1.0
        original = fixtures.FakeCurriculum.answer_pairs
        with patch.object(fixtures.FakeCurriculum, 'answer_pairs', lambda data: original(data)[:1]):
            runtime = fixtures.FakeRuntime()
            with self.assertRaisesRegex(ValueError, 'Too few'): f.run_loop(runtime, fixtures.FakeEvaluator([.1]))
        self.assertEqual(runtime.trains, [])
        coverage = json.loads((Path(f.tmp.name) / 'round_01/training_coverage.json').read_text())
        self.assertEqual(coverage['shortfall'], 1)
        self.assertFalse((Path(f.tmp.name) / 'round_01/training_plan.json').exists())

    def test_final_regression_keeps_baseline_as_release(self):
        f = self.fixture(); f.config['evaluation'].update(mode='reviewed', minimum_test_examples_per_task=3)
        final = [{'record': fixtures.record(i=1000), 'reference': {}}]
        runtime = fixtures.FakeRuntime()
        with patch('distillation_loop.controller.load_benchmarks', return_value=(f.development, final)):
            result = f.run_loop(runtime, fixtures.FakeEvaluator([.1, .95, .9, .8]))
        self.assertFalse(result['promotion']['accepted'])
        self.assertEqual(result['release_checkpoint'], result['initial_checkpoint'])
        self.assertEqual(runtime.restores[-1], result['initial_checkpoint'])

    def test_validation_regression_excludes_candidate(self):
        f = self.fixture()
        class Regressing(fixtures.FakeEvaluator):
            def evaluate(self, *args):
                self.calls += 1
                if self.calls == 1: return fixtures.report(.5)
                result = fixtures.report(.95)
                result['per_task']['patient_billing'] = .4
                values = list(result['per_task'].values())
                result.update(macro_task_score=sum(values)/len(values), minimum_task_score=min(values))
                return result
        result = f.run_loop(fixtures.FakeRuntime(), Regressing([]))
        self.assertEqual(result['best_checkpoint'], result['initial_checkpoint'])
        self.assertFalse(result['history'][0]['regression_gate']['accepted'])

class FullLoopIntegrationTests(unittest.TestCase):
    def test_real_pipeline_evaluator_selection_and_resume_with_model_boundaries_faked(self):
        from distillation_loop.controller import run_loop
        from self_instruct_pipeline.prompt_builder import teacher_messages
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent = MockParent()
            cfg = load_loop_config()
            cfg['student_conditioned_training'] = False
            cfg.update(output_dir=str(root / 'run'), max_rounds=1, scenarios_per_round=4, development_examples_per_task=1)
            cfg['evaluation'].update(mode='reviewed', validation_path=str(root / 'validation.jsonl'), test_path=str(root / 'test.jsonl'), minimum_test_examples_per_task=1)
            for split in ('validation', 'test'):
                rows = []
                for task in SCHEMAS:
                    record = {'id': f'{split}-{task}', 'task': task,
                              'instruction': f'Integration fixture {split}: answer the {task} request from the record.',
                              'scenario': {'synthetic': True, 'documents': [{'document_id': 'D1', 'page_number': 1,
                                  'text': (f'Integration validation {task}: required claim form missing.' if split == 'validation' else
                                           f'Archive {task}: no claim form included; submission incomplete.')}], 'context': {}}}
                    answer = MockParent().generate_json(teacher_messages(record, SCHEMAS), SCHEMAS[task], 'teacher_answer')
                    item = {'record': record, 'reference': answer,
                            'review': {'status': 'approved', 'reviewer': 'TEST FIXTURE ONLY', 'source_group': f'{split}-{task}'}}
                    if 'answer' in answer:
                        item['rubric'] = {'decision': 'missing_form', 'required_facts': ['Claim form missing.']}
                    rows.append(item)
                atomic_jsonl(root / f'{split}.jsonl', rows)

            class Runtime:
                def __init__(self): self.version = 0; self.trains = 0; self.inferences = 0
                def identity(self): return {'model': 'offline-integration-fixture'}
                def prepare(self): pass
                def save_checkpoint(self, path):
                    path = Path(path); path.mkdir(parents=True, exist_ok=True)
                    (path / 'version.json').write_text(json.dumps(self.version))
                    return str(path)
                def restore(self, path): self.version = json.loads((Path(path) / 'version.json').read_text())
                def train(self, pairs, schemas, directory):
                    self.trains += 1; self.version = 1
                    self.pairs = pairs
                    return self.save_checkpoint(directory / 'adapter')
                def infer(self, payload):
                    self.inferences += 1
                    if set(payload) != {'instruction', 'scenario', 'response_schema'}: raise AssertionError('Hidden label leakage')
                    answer = MockParent().generate_json([{'role': 'system', 'content': ''}, {'role': 'user', 'content': json.dumps(payload)}], payload['response_schema'], 'teacher_answer')
                    if self.version == 0: answer['invalid_fixture_key'] = True
                    return {'raw': json.dumps(answer), 'truncated': False}

            base = load_config(); runtime = Runtime()
            with patch('self_instruct_pipeline.pipeline.create_parent', return_value=parent):
                result = run_loop(runtime, base, cfg)
                inference_count = runtime.inferences
                resumed = run_loop(runtime, base, cfg)
            self.assertEqual(runtime.trains, 1)
            self.assertEqual(len(runtime.pairs), 4)
            self.assertEqual(runtime.inferences, inference_count)
            self.assertTrue(result['promotion']['accepted'])
            self.assertEqual(result, resumed)
            self.assertEqual(runtime.version, 1)
            self.assertEqual(result['history'][0]['coverage']['deterministic_check_count'], 1)
            self.assertEqual(result['final_comparison']['macro_delta'], 1)
            heldout = {f'{split}-{task}' for split in ('validation', 'test') for task in SCHEMAS}
            self.assertFalse(heldout & {r['id'] for r, _ in runtime.pairs})

class ServiceFailureTests(unittest.TestCase):
    def test_auth_budget_and_audit_outages_preserve_quality_attempts(self):
        from self_instruct_pipeline.budget import BudgetExceeded
        for operation, failure in [('teacher_answer', ParentError('HTTP 401', False)),
                                   ('teacher_answer', BudgetExceeded('budget reached')),
                                   ('answer_audit', ParentError('network failure', True))]:
            class Failing(MockParent):
                def generate_json(self, messages, schema, op='generation', temperature=None):
                    if op == operation: raise failure
                    return super().generate_json(messages, schema, op, temperature)
            with self.subTest(operation=operation, failure=failure), tempfile.TemporaryDirectory() as folder:
                cfg = load_config(); cfg['paths']['output_dir'] = folder
                cfg['generation'].update(target_accepted_scenarios=1, batch_size=1)
                pipeline = Pipeline(cfg, Failing()); pipeline.generate()
                with self.assertRaises(type(failure)): pipeline.answer(allow_partial=True, max_sample_attempts=1)
                resumed = Pipeline(cfg, MockParent())
                self.assertEqual(resumed.answer(allow_partial=True, max_sample_attempts=1), 1)
                self.assertEqual(resumed.store.read('validated_answers.jsonl')[0]['answer_attempt'], 1)

    def test_generation_outage_can_resume_without_consuming_generation_limit(self):
        class Failing(MockParent):
            def generate_json(self, messages, schema, operation='generation', temperature=None):
                if operation == 'scenario_generation': raise ParentError('network failure', True)
                return super().generate_json(messages, schema, operation, temperature)
        with tempfile.TemporaryDirectory() as folder:
            cfg = load_config(); cfg['paths']['output_dir'] = folder
            cfg['generation'].update(target_accepted_scenarios=1, batch_size=1)
            for _ in range(7):
                with self.assertRaises(ParentError): Pipeline(cfg, Failing()).generate()
            resumed = Pipeline(cfg, MockParent())
            self.assertEqual(resumed.generate(), 1)
            self.assertEqual(len(resumed.store.read('generation_transport_failures.jsonl')), 7)

class FinalAdversarialTests(unittest.TestCase):
    def test_empty_evidence_cannot_pass_positive_classification(self):
        r = fixtures.record()
        ref = {'document_type': 'hospital_bill', 'evidence': ['Synthetic hospital invoice'], 'requires_review': False}
        self.assertEqual(score_response(r, ref, json.dumps(dict(ref, evidence=[])), SCHEMAS)['schema_evidence_valid'], 0)

    def test_case_distinct_codes_do_not_get_automatic_reference_agreement(self):
        item = reasoning_item(); item['record']['task'] = 'clinical_coding'
        source = 'Fictional catalog: A and a are distinct nonbillable training codes.'
        item['record']['scenario']['documents'][0]['text'] = source
        item['reference'].update(answer='Use the supported training code.', suggested_codes=['A'], evidence=[{'document_id': 'D1', 'quote': source}])
        wrong = dict(item['reference'], suggested_codes=['a'])
        prediction = {'raw': json.dumps(wrong), 'truncated': False}
        self.assertIsNone(grade_reviewed(item, prediction, SCHEMAS, {})[0])
        self.assertEqual(score_response(item['record'], item['reference'], prediction['raw'], SCHEMAS)['task_score'], 0)

    def test_preflight_reports_both_missing_splits_without_loading_runtime(self):
        from distillation_loop.preflight import readiness
        cfg = load_loop_config()
        with tempfile.TemporaryDirectory() as folder:
            cfg['evaluation'].update(mode='reviewed', validation_path=f'{folder}/validation.jsonl', test_path=f'{folder}/test.jsonl')
            result = readiness(load_config(), cfg, check_runtime=False)
        self.assertFalse(result['ready'])
        self.assertEqual(len(result['issues']), 2)

    def test_unknown_quality_option_is_rejected(self):
        from distillation_loop.quality import validate_quality_settings
        with self.assertRaises(ValueError): validate_quality_settings({'quality': {'max_regresion': .1}})

    def test_template_boundary_answers_and_and_or_counterfactual(self):
        self.assertIn('USD 0.00', template_pair(3)[1]['answer'])
        self.assertIn('USD -25.00', template_pair(6)[1]['answer'])
        self.assertIn('2024-02-29', template_pair(19)[1]['answer'])
        self.assertIn('not met', template_pair(5)[1]['answer'])  # true AND false
        self.assertIn('criteria are met', template_pair(17)[1]['answer'])  # true OR false


class PairedRegressionTests(unittest.TestCase):
    def test_small_churn_passes_but_systematic_losses_fail(self):
        from distillation_loop.quality import regression_gate

        tasks = [f'task_{n}' for n in range(18)]

        def report(losses):
            return {
                'per_task': {task: 1 - (losses / 20 if task == tasks[0] else 0) for task in tasks},
                'case_scores': {f'{task}/{index}': {'task': task, 'score': int(task != tasks[0] or index >= losses)}
                                for task in tasks for index in range(20)},
            }

        settings = {'max_task_regression': .05, 'paired_regression_alpha': .05}
        small = regression_gate(report(0), report(2), settings)
        self.assertTrue(small['accepted'])
        self.assertEqual(small['paired'][tasks[0]]['losses'], 2)
        self.assertFalse(regression_gate(report(0), report(12), settings)['accepted'])
