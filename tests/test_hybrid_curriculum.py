import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from distillation_loop.hybrid import hybrid_pairs
from distillation_loop.leakage import ExclusionRegistry, scenario_content
from distillation_loop.trusted_templates import fresh_template_pairs
from self_instruct_pipeline.budget import ACTIVE_BUDGET, APIBudget, BudgetExceeded
from self_instruct_pipeline.config import load_config, resolve_path
from self_instruct_pipeline.parallel import CachedParent, batch_requests, bounded_map
from self_instruct_pipeline.validation import ValidationError, validate_answer

SCHEMAS = json.loads(resolve_path(load_config()['paths']['schemas']).read_text())['tasks']
PROFILE = json.loads(resolve_path('config/hybrid_colab_v1.json').read_text())


class HybridCurriculumTests(unittest.TestCase):
    def test_two_fresh_template_rounds_have_unique_content_and_grounded_targets(self):
        registry = ExclusionRegistry([], near_duplicates=False)
        rows = []
        for seed in ('42-1', '42-2'):
            pairs = fresh_template_pairs(60, seed, SCHEMAS, registry.match)
            self.assertEqual(len({record['task'] for record, _ in pairs}), 6)
            self.assertIn('contradiction', {record['case_type'] for record, _ in pairs})
            self.assertIn('insufficient_evidence', {record['case_type'] for record, _ in pairs})
            for record, answer in pairs:
                self.assertIsNone(registry.match(record))
                registry.add(record)
                rows.append(scenario_content(record))
                broken = {**answer}
                key = record['answer_checks'][0]['path'][0]
                broken[key] = None
                with self.assertRaises(ValidationError):
                    validate_answer(record, broken, SCHEMAS[record['task']])
        self.assertEqual(len(set(rows)), 120)

    def test_hybrid_mix_small_window_and_resume(self):
        def candidate(i):
            task = list(SCHEMAS)[i % len(SCHEMAS)]
            return ({'id': f'teacher-{i}', 'task': task, 'difficulty': 'hard', 'instruction': 'Assess documents.',
                     'scenario': {'documents': [{'text': f'Independent case {i}'}]}}, {})
        pool = []
        calls = []
        def fetch(pipe, target, weights, old, *, eligible):
            calls.append(target)
            for i in range(len(pool), target):
                pool.append(candidate(i))
            return [p for p in pool if p[0]['id'] not in old and eligible(p)]

        def assess(pipe, record, reference, *args):
            return {'scenario': record, 'reference': reference, 'status': 'accepted',
                    'assessment': {'verdict': 'correct', 'error_categories': []}}

        pipe = SimpleNamespace(accepted=[], schemas=SCHEMAS, exclusion_dedup=SimpleNamespace(check=lambda record: None),
                               client=lambda: None)
        data = SimpleNamespace(pipeline=pipe, exclusion_registry=ExclusionRegistry([]))
        config = {**PROFILE, 'training': {'replay_previous_rounds': False}, 'random_seed': 42,
                  'student_conditioned_training': True, 'rollback_on_regression': True}
        state = {'completed_rounds': 0, 'last_checkpoint': 'test'}
        weights = dict.fromkeys(SCHEMAS, 1)
        windows = []
        def predict(runtime, records, *args):
            windows.append(len(records))
            return {record['id']: {} for record in records}
        with (tempfile.TemporaryDirectory() as tmp,
              patch('distillation_loop.hybrid.fresh_teacher_pool', side_effect=fetch),
              patch('distillation_loop.hybrid.assess_candidate', side_effect=assess)):
            directory = Path(tmp) / 'round_01'
            directory.mkdir()
            first, diagnostics = hybrid_pairs(data, None, directory, state, config, weights, predict, [])
            second, _ = hybrid_pairs(data, None, directory, state, config, weights, predict, [])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 200)
        self.assertEqual(len({record['task'] for record, _ in first}), 18)
        self.assertEqual(diagnostics['training_mix'], {'deterministic': 60, 'general': 60, 'adaptive': 80})
        self.assertEqual(windows, [88, 88])
        self.assertEqual(calls, [60, 0, 148, 0])

    def test_concurrent_budget_context_and_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = APIBudget(Path(tmp) / 'budget', max_calls=2, max_cost_usd='1')
            def reserve(_):
                self.assertIs(ACTIVE_BUDGET.get(), budget)
                return budget.reserve({'provider': 'openai', 'model': 'gpt-6-luna'}, None)
            with budget.activate():
                with self.assertRaises(BudgetExceeded):
                    bounded_map(reserve, range(8), 4)
                self.assertEqual(budget.state['calls'], 2)

    def test_batch_cache_persists_and_limits_concurrency(self):
        class Provider:
            def __init__(self):
                self.calls = self.running = self.peak = 0
                self.lock = threading.Lock()

            def generate_json(self, messages, schema, operation, temperature):
                with self.lock:
                    self.calls += 1
                    self.running += 1
                    self.peak = max(self.running, self.peak)
                time.sleep(.015)
                result = {key: {'value': item['value']}
                          for key, item in json.loads(messages[1]['content']).items()}
                with self.lock:
                    self.running -= 1
                return result

        schema = {'type': 'object', 'required': ['value'], 'additionalProperties': False,
                  'properties': {'value': {'type': 'integer'}}}
        requests = [([{'role': 'system', 'content': 'Independent test'},
                      {'role': 'user', 'content': json.dumps({'value': i})}], schema) for i in range(12)]
        with tempfile.TemporaryDirectory() as tmp:
            provider = Provider()
            client = CachedParent(provider, tmp, {}, requests_per_second=1000)
            batch_requests(client, requests, 'teacher_answer', 4, 3)
            self.assertEqual((provider.calls, provider.peak), (4, 4))
            resumed = CachedParent(provider, tmp, {}, requests_per_second=1000)
            batch_requests(resumed, requests, 'teacher_answer', 4, 3)
            self.assertEqual(provider.calls, 4)
            self.assertEqual([resumed.cached(messages, schema, 'teacher_answer')['value']
                              for messages, schema in requests], list(range(12)))
