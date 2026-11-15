import tempfile
import unittest
from pathlib import Path

from distillation_loop.calibration import calibrate
from distillation_loop.colab import distillation_1000_configs
from distillation_loop.controller import run_loop
from distillation_loop.core import load_loop_config
from distillation_loop.leakage import ExclusionRegistry, freeze_loss_split
from distillation_loop.paired import paired_summary
from self_instruct_pipeline.config import load_config
from self_instruct_pipeline.persistence import atomic_json, atomic_jsonl, digest

from test_distillation_loop import SCHEMAS, record



class RevisedStageTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_profile_blocks_loading_runtime_without_semantic_result_and_authorization(self):
        parent, config = distillation_1000_configs(load_config(), load_loop_config(), self.root)
        self.assertFalse(config['preference_training'])
        self.assertEqual(config['development_examples_per_task'], 20)
        self.assertEqual(config['evaluation']['total_examples'], 360)
        self.assertIn('split_path', config['evaluation'])
        self.assertNotIn('holdout_families_per_task', config['evaluation'])
        with self.assertRaisesRegex(ValueError, 'Stage 2 stop'):
            run_loop(object(), parent, config)
        result = self.root / 'comparison.json'
        atomic_json(result, {'diagnostic_decision': {'outcome': 'inconclusive'}})
        with self.assertRaisesRegex(ValueError, 'requires investigation'):
            run_loop(object(), parent, config, semantic_result=result, continuation_authorized=True)

    def test_historical_exclusions_and_frozen_teacher_split(self):
        # Start from verified document classifications in the bundled corpus.
        from distillation_loop.semantic_eval import CORPUS
        from self_instruct_pipeline.persistence import read_jsonl
        item = next(row for row in read_jsonl(CORPUS) if row['record']['task'] == 'document_classification')
        historic = [{'record': {**item['record'], 'id': f'historical-{i}', 'source_group': 'history'},
                     'answer': item['reference']} for i in range(200)]
        plan = self.root / 'training_plan.json'
        atomic_json(plan, {'pairs': historic, 'hash': digest(historic)})
        audit = {'grounded': True, 'calculations_checked': True, 'uncertainty_handled': True,
                 'unsupported_actions': False, 'safe_for_synthetic_sft': True, 'reasons': []}
        duplicate = {**historic[0]['record'], 'id': 'different-id', 'source_group': 'different-group'}
        candidates = self.root / 'candidates.jsonl'
        atomic_jsonl(candidates, [{'record': duplicate, 'answer': item['reference'],
                                   'scenario_hash': digest(duplicate), 'audit': audit,
                                   'verification_status': 'automatically_checked_silver'}])
        output = self.root / 'split.json'
        frozen = freeze_loss_split(output, plan, [], SCHEMAS, candidates_path=candidates)
        self.assertEqual(frozen['excluded'], {'historical_canonical_content': 1})
        self.assertEqual(frozen['rows'], [])
        self.assertIn('Only 0', frozen['limitations'][0])
        self.assertEqual(freeze_loss_split(output, plan, [], SCHEMAS, candidates_path=candidates), frozen)
        atomic_jsonl(candidates, [])
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            freeze_loss_split(output, plan, [], SCHEMAS, candidates_path=candidates)
        registry = ExclusionRegistry([historic[0]])
        unrelated = record(i=1000)
        unrelated['source_family_id'] = 'history'
        with self.assertRaisesRegex(ValueError, 'source_family'):
            registry.require_clean([unrelated], 'replay')

    def test_source_family_intervals_are_wider_than_case_independent_intervals(self):
        scores = [0.1] * 108
        separate = paired_summary(scores)
        grouped = paired_summary(scores, [i // 3 for i in range(len(scores))])
        self.assertEqual(grouped['independent_units'], 36)
        self.assertLess(grouped['conservative_95pct_ci'][0], separate['conservative_95pct_ci'][0])

    def test_calibration_binds_prediction_and_human_label(self):
        comparison = self.root / 'semantic_comparison.json'
        identity = {'development_sha256': 'fixed'}
        prediction = '12345'
        atomic_json(comparison, {'scope': 'development synthetic silver', 'scoring': 'source-rubric-v2',
                                 'identity': identity, 'runs': {'sft': {}}})
        atomic_jsonl(self.root / 'sft_semantic_cases.jsonl', [
            {'scenario_id': 'example', 'task': 'appeal_drafting', 'prediction_sha256': prediction,
             'task_score': 0.0},
        ])
        labels = self.root / 'labels.jsonl'
        expected = {'checkpoint': 'sft', 'scenario_id': 'example', 'reviewer': 'Independent Reviewer',
                    'comparison_identity_sha256': digest(identity), 'prediction_sha256': prediction,
                    'review_flag_correct': True, 'schema_evidence_valid': True,
                    'judgment': {'decision_correct': True, 'facts_correct': True,
                                 'abstention_correct': True, 'unsupported_claims': False}}
        atomic_jsonl(labels, [expected])
        result = calibrate(comparison, labels)
        self.assertEqual(result['human_vs_semantic_confusion']['pass'], {'1/0': 1})
        atomic_jsonl(labels, [{**expected, 'prediction_sha256': 'different'}])
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            calibrate(comparison, labels)

    def test_saved_release_split_is_explicit_and_cannot_be_reused_for_tuning(self):
        from unittest.mock import patch
        from distillation_loop.benchmarks import fixed_family_split
        from test_distillation_loop import FakeRuntime, report, with_safety_evidence

        parent, config = distillation_1000_configs(load_config(), load_loop_config(), self.root / 'run')
        config.update(require_semantic_decision=False, stop_when_thresholds_met=True)
        config['evaluation']['semantic_judge'] = False
        selection_config = {**config, 'evaluation': {**config['evaluation'], 'holdout_families_per_task': 2}}
        development, holdout = fixed_family_split(selection_config, SCHEMAS)
        split = self.root / 'evaluation_split.json'
        atomic_json(split, {
            name: {'case_ids': [row['record']['id'] for row in rows], 'sha256': digest(rows),
                   'source_groups': sorted({row['source_group'] for row in rows})}
            for name, rows in [('development', development), ('holdout', holdout)]
        })
        config['evaluation']['split_path'] = str(split)
        trained = [{'record': row['record'], 'answer': row['reference']} for row in development[:200]]
        history = self.root / 'training_plan.json'
        atomic_json(history, {'pairs': trained, 'hash': digest(trained)})
        config['historical_training_plan'] = str(history)

        class Runtime(FakeRuntime):
            def save_checkpoint(self, path):
                result = super().save_checkpoint(path)
                (path / 'adapter_model.safetensors').write_bytes(b'test weights')
                (path / 'adapter_config.json').write_text('{}')
                return result

        evaluations = []
        class Evaluator:
            def __init__(self, runtime, schemas, minimum):
                self.minimum = minimum

            def evaluate(self, rows, directory):
                evaluations.append((len(rows), directory.name))
                directory.mkdir(parents=True, exist_ok=True)
                atomic_jsonl(directory / 'predictions.jsonl', [
                    {'scenario_id': row['record']['id'], 'task_score': .99} for row in rows])
                return with_safety_evidence({**report(.99), 'examples_per_task': dict.fromkeys(SCHEMAS, self.minimum)})

        with patch('distillation_loop.controller.LightevalEvaluator', Evaluator):
            result = run_loop(Runtime(), parent, config)
            self.assertEqual(evaluations, [(360, 'baseline')])
            self.assertEqual(result['provisional_best_checkpoint'], result['initial_checkpoint'])
            run_loop(Runtime(), parent, config, evaluate_release=True)
            self.assertEqual(evaluations[-2:], [(108, 'initial'), (108, 'selected')])
            run_loop(Runtime(), parent, config, evaluate_release=True)
            self.assertEqual(len(evaluations), 3)
            with self.assertRaisesRegex(ValueError, 'already been used'):
                run_loop(Runtime(), parent, config, round_limit=config['max_rounds'] + 1)
        self.assertIsNone(result['release_checkpoint'])
        self.assertNotIn('final_comparison', result)
