import json
import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from distillation_loop.benchmarks import fixed_family_split
from distillation_loop.colab import distillation_1000_configs
from distillation_loop.controller import run_loop
from distillation_loop.core import load_loop_config
from distillation_loop.hybrid import refill_window
from distillation_loop.leakage import ExclusionRegistry, scenario_content
from self_instruct_pipeline.budget import APIBudget, BudgetExceeded, ACTIVE_BUDGET
from self_instruct_pipeline.config import load_config
from self_instruct_pipeline.persistence import atomic_json, digest, now
from self_instruct_pipeline.pipeline import Pipeline
from self_instruct_pipeline.usage import export_usage
from test_distillation_loop import FakeRuntime, SCHEMAS, report, with_safety_evidence
from test_self_instruct import GOOD_AUDIT, MockParent, scenario


class SyntheticProvider:
    def __init__(self, logger=lambda event: None, interruption=None, delay=0):
        self.logger = logger
        self.interruption = interruption
        self.delay = delay
        self.calls, self.running, self.peak = Counter(), Counter(), Counter()
        self.serial = 0
        self.lock = threading.Lock()

    def check_model(self):
        return {'model': 'offline fixture'}

    def generate_json(self, messages, schema, operation, temperature=None):
        with self.lock:
            self.calls[operation] += 1
            self.running[operation] += 1
            self.peak[operation] = max(self.peak[operation], self.running[operation])
        started = now()
        budget = ACTIVE_BUDGET.get()
        try:
            if operation == self.interruption:
                with self.lock:
                    self.interruption = None
                raise BudgetExceeded('mock interruptions preserve completed siblings')
            reservation = budget.reserve({'provider': 'openai', 'model': 'gpt-6-luna'}, None) if budget else None
            if operation in ('scenario_generation', 'scenario_audit'):
                time.sleep(self.delay)
            payload = json.loads(messages[1]['content'])
            if operation == 'scenario_generation':
                request = payload['request']
                difficulties = [d for d, n in request['difficulty'].items() for _ in range(n)]
                output = []
                for task, count in request['task_quotas'].items():
                    for _ in range(count):
                        with self.lock:
                            self.serial += 1
                            number = self.serial
                        row = scenario(task, number, difficulties.pop())
                        row['scenario']['documents'][0]['text'] = ' '.join(
                            [f'case{number}word{i}' for i in range(6)] + ['The required form is missing.'])
                        output.append(row)
            elif operation == 'scenario_audit':
                output = [{'generation_id': row['generation_id'], 'usable': True,
                           'coverage_dimensions': ['insufficient_ambiguous_unreadable'],
                           'duplicate_of': '', 'reasons': []} for row in payload]
            elif operation == 'student_assessment':
                output = {'verdict': 'correct', 'error_categories': [], 'findings': 'offline'}
            elif operation.endswith('_batch'):
                output = {key: self.answer(item, schema['properties'][key], operation[:-6])
                          for key, item in payload.items()}
            else:
                output = self.answer(payload, schema, operation)
            event = {'operation': operation, 'timestamp': started, 'finished_at': now(),
                     'duration_seconds': self.delay if operation.startswith('scenario_') else 0,
                     'status': 'ok', 'model': 'gpt-6-luna', 'provider': 'openai',
                     'usage': {'input_tokens': 0, 'output_tokens': 0}}
            self.logger(event)
            if budget:
                budget.settle(reservation, event)
            return output
        finally:
            with self.lock:
                self.running[operation] -= 1

    @staticmethod
    def answer(payload, schema, operation):
        if operation == 'answer_audit':
            return GOOD_AUDIT.copy()
        if 'document_type' in schema['required']:
            return {'document_type': 'other', 'evidence': ['The required form is missing.'],
                    'requires_review': True}
        return MockParent().generate_json(
            [{}, {'content': json.dumps(payload)}], schema, 'teacher_answer')


class HybridReviewTests(unittest.TestCase):
    def test_parallel_wave_resume_preserves_requests_and_exact_quotas(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config()
            config['paths']['output_dir'] = tmp
            config['generation'].update(batch_size=2, target_accepted_scenarios=8)
            provider = SyntheticProvider(interruption='scenario_audit', delay=.03)
            pipeline = Pipeline(config, provider)
            pipeline.request_options = {'workers': 4}
            pipeline.initialize_manifest()
            quotas = {'claim_scrubbing': 5, 'compliance_qa': 3}
            with self.assertRaises(BudgetExceeded):
                pipeline.generate(task_targets=quotas)
            self.assertEqual(provider.calls['scenario_generation'], 4)
            self.assertEqual(provider.peak['scenario_generation'], 4)
            self.assertGreater(provider.peak['scenario_audit'], 1)

            resumed = Pipeline(config, provider)
            resumed.request_options = {'workers': 4}
            resumed.initialize_manifest()
            resumed.generate(task_targets=quotas)
            self.assertEqual(provider.calls['scenario_generation'], 4)
            self.assertEqual(provider.calls['scenario_audit'], 5)
            self.assertEqual(Counter(r['task'] for r in resumed.accepted), quotas)
            self.assertEqual(len({r['id'] for r in resumed.accepted}), 8)
            self.assertFalse(resumed.store.path('pending_generation_wave.json').exists())

    def test_frozen_rows_never_reach_audit_and_cross_batch_duplicates_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config()
            config['paths']['output_dir'] = tmp
            config['generation'].update(batch_size=1, target_accepted_scenarios=4,
                                        max_consecutive_failed_batches=1)
            frozen = scenario(i=917)
            repeated = scenario(i=218)
            repeated['instruction'] = 'Reconcile the parcel custody and receipt timestamps.'
            repeated['scenario']['documents'][0]['text'] = 'Depot 71 accepted the envelope at 08:33. No receipt accompanied it.'
            received = []

            class DuplicateProvider(SyntheticProvider):
                def generate_json(self, messages, schema, operation, temperature=None):
                    if operation == 'scenario_generation':
                        payload = json.loads(messages[1]['content'])['request']
                        with self.lock:
                            row = dict(repeated if operation in received else frozen)
                            received.append(operation)
                        row['difficulty'] = next(key for key, count in payload['difficulty'].items() if count)
                        return [row]
                    for row in json.loads(messages[1]['content']):
                        if row['scenario'] == frozen['scenario']:
                            raise AssertionError('Frozen case entered an API audit')
                    return super().generate_json(messages, schema, operation, temperature)

            pipeline = Pipeline(config, DuplicateProvider())
            pipeline.request_options = {'workers': 4}
            pipeline.exclusion_guard = ExclusionRegistry([{**frozen, 'id': 'release-case'}]).match
            pipeline.initialize_manifest()
            from self_instruct_pipeline.pipeline import GenerationExhausted
            with self.assertRaises(GenerationExhausted):
                pipeline.generate(task_targets={'claim_scrubbing': 4})
            self.assertEqual(len(pipeline.accepted), 1)
            self.assertEqual(pipeline.accepted[0]['scenario'], repeated['scenario'])
            rejections = pipeline.store.read('rejected_records.jsonl')
            self.assertIn('held_out_overlap', {row['reason'] for row in rejections})
            self.assertIn('exact_duplicate', {row['reason'] for row in rejections})

    def test_refill_scales_down_to_missing_slots_and_coverage(self):
        self.assertEqual(refill_window(80, {}, 8), 88)
        self.assertEqual(refill_window(1, {}, 8), 2)
        self.assertEqual(refill_window(0, {'one': 1, 'two': 2}, 8), 3)
        self.assertEqual(refill_window(0, {}, 8), 0)

    def test_four_round_end_to_end_with_mock_api_and_research_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            parent, config = distillation_1000_configs(load_config(), load_loop_config(), root)
            config.update(max_rounds=1, require_semantic_decision=False, research_macro_continuation=True,
                          hybrid_curriculum={'deterministic': 60, 'general': 60, 'adaptive': 80,
                                             'buffer': 8, 'workers': 4, 'batch_size': 3,
                                             'requests_per_second': 4})
            config['quality']['minimum_new_pair_fraction'] = 1.0
            config['training']['replay_fraction'] = 0.0
            split_config = {**config, 'evaluation': {**config['evaluation'], 'holdout_families_per_task': 2}}
            development, holdout = fixed_family_split(split_config, SCHEMAS)
            split = Path(tmp) / 'evaluation_split.json'
            atomic_json(split, {
                name: {'case_ids': [row['record']['id'] for row in cases], 'sha256': digest(cases),
                       'source_groups': sorted({row['source_group'] for row in cases})}
                for name, cases in [('development', development), ('holdout', holdout)]})
            config['evaluation']['split_path'] = str(split)

            class Runtime(FakeRuntime):
                def __init__(self):
                    super().__init__()
                    self.starts = []

                def train(self, pairs, schemas, output_dir):
                    self.starts.append(self.restored_adapter[0])
                    return super().train(pairs, schemas, output_dir)

            evaluated = []
            class Evaluator:
                def __init__(self, runtime, *_):
                    self.runtime = runtime

                def evaluate(self, rows, directory):
                    directory.mkdir(parents=True, exist_ok=True)
                    self_rows = [row['record']['id'] for row in rows]
                    if self_rows != [row['record']['id'] for row in development]:
                        raise AssertionError('Evaluation received non-development cases')
                    evaluated.append(directory.name)
                    attempt = len(evaluated)
                    summary = with_safety_evidence({
                        **report((.4, .6, .55, .56, .57)[attempt - 1]), 'examples_per_task': dict.fromkeys(SCHEMAS, 20)})
                    summary.update(adapter_sha256=self.runtime.restored_adapter[1], generation_key='mock')
                    if attempt == 2:
                        next(iter(summary['case_scores'].values()))['unsupported_claims'] = True
                    return summary

            provider = SyntheticProvider()
            def create(settings, *, logger):
                provider.logger = logger
                return provider

            runtime = Runtime()
            with (patch('distillation_loop.leakage.historical_records', return_value=([], 'mock-plan')),
                  patch('distillation_loop.controller.freeze_loss_split', return_value={
                      'rows': [], 'split_sha256': 'mock-loss', 'per_task': {}, 'excluded': {}, 'limitations': []}),
                  patch('distillation_loop.controller.LightevalEvaluator', Evaluator),
                  patch('self_instruct_pipeline.parent_backend.create_parent', side_effect=create),
                  patch('self_instruct_pipeline.pipeline.create_parent', side_effect=create),
                  patch('self_instruct_pipeline.parallel.sleep'),
                  APIBudget(root / 'api_budget', 2000, '1').activate() as budget):
                first = run_loop(runtime, parent, config)
                self.assertEqual(first['completed_rounds'], 1)
                self.assertNotEqual(first['next_round_start_checkpoint'], first['best_checkpoint'])
                self.assertEqual(runtime.restores[-1], first['best_checkpoint'])
                result = run_loop(runtime, parent, config, round_limit=4)
                self.assertEqual(export_usage(root)['calls'], budget.state['calls'])
                self.assertEqual(sum(provider.calls.values()), budget.state['calls'])
                with self.assertRaisesRegex(ValueError, 'release scoring separate'):
                    run_loop(runtime, parent, config, evaluate_release=True)

            self.assertEqual(evaluated, ['baseline', *['lighteval'] * 4])
            self.assertEqual(len(runtime.trains), 4)
            self.assertEqual(result['completed_rounds'], 4)
            self.assertEqual(result['status'], 'max_rounds_reached')
            registry = ExclusionRegistry(development + holdout)
            prior_content = set()
            for index, pairs in enumerate(runtime.trains, 1):
                self.assertEqual(len(pairs), 200)
                registry.require_clean([record for record, _ in pairs], 'dry run')
                content = {scenario_content(record) for record, _ in pairs}
                self.assertEqual(len(content), 200)
                self.assertFalse(content & prior_content)
                prior_content.update(content)
                plan = json.loads((root / f'round_{index:02d}' / 'training_plan.json').read_text())
                self.assertEqual(plan['diagnostics']['training_mix'], {'deterministic': 60, 'general': 60, 'adaptive': 80})
                self.assertEqual(len({record['task'] for record, _ in pairs}), 18)
            top = result['experimental_macro_rank'][0]
            self.assertTrue(top['safety_failures'])
            self.assertFalse(top['release_eligible'])
            self.assertEqual(runtime.starts[1:], [str(Path(top['checkpoint']).resolve())] * 3)
            self.assertNotEqual(result['best_checkpoint'], top['checkpoint'])
            self.assertIsNone(result['release_checkpoint'])
            stages = json.loads((root / 'api_stage_profile.json').read_text())
            self.assertEqual({name: stage['calls'] for name, stage in stages.items()}, dict(provider.calls))
            for stage in stages.values():
                self.assertLessEqual(stage['active_request_wall_seconds'], stage['elapsed_span_seconds'])
            print('Hybrid dry run API mock counts:', dict(provider.calls))
