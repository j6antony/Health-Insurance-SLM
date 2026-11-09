import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from distillation_loop.core import load_loop_config
from distillation_loop.smoke import smoke_configs
from self_instruct_pipeline.budget import APIBudget, BudgetExceeded
from self_instruct_pipeline.config import load_config
from self_instruct_pipeline.parent_backend import OpenAIParent


class BudgetTests(unittest.TestCase):
    def test_smoke_settings_do_not_mutate_full_config(self):
        parent, loop = load_config(), load_loop_config()
        p, l = smoke_configs(parent, loop, '/tmp/smoke')
        self.assertEqual((l['max_rounds'], l['scenarios_per_round'], l['development_examples_per_task']), (1, 8, 1))
        self.assertEqual(l['training']['max_steps_per_round'], 2)
        self.assertEqual(p['parent']['max_retries'], 1)
        self.assertEqual(loop['max_rounds'], 5)
        self.assertEqual(parent['parent']['max_retries'], 3)

    def test_shared_call_limit_includes_retries_and_persists(self):
        calls = []
        def opener(request, timeout):
            calls.append(1)
            raise urllib.error.HTTPError(request.full_url, 429, 'limited', {}, None)
        settings = load_config()['parent']
        with tempfile.TemporaryDirectory() as folder:
            budget = APIBudget(Path(folder)/'api_budget', max_calls=2)
            with budget.activate():
                client = OpenAIParent(settings, api_key='TEST', opener=opener, sleep=lambda _: None)
                with self.assertRaises(BudgetExceeded):
                    client.check_model()
                self.assertEqual(len(calls), 2)
            with APIBudget(Path(folder)/'api_budget', max_calls=2).activate():
                with self.assertRaises(BudgetExceeded):
                    OpenAIParent(settings, api_key='TEST', opener=opener).check_model()
            self.assertEqual(len(calls), 2)

    def test_pre_request_cost_reservation_and_settlement(self):
        settings = load_config()['parent']
        body = {'service_tier': 'default', 'max_output_tokens': 12000, 'input': 'test'}
        with tempfile.TemporaryDirectory() as folder:
            with APIBudget(folder, max_cost_usd='0.001').activate() as budget:
                with self.assertRaises(BudgetExceeded):
                    budget.reserve(settings, body)
                self.assertEqual(budget.state['calls'], 0)
        with tempfile.TemporaryDirectory() as folder:
            with APIBudget(folder).activate() as budget:
                reservation = budget.reserve(settings, body)
                self.assertGreater(float(budget.state['accounted_usd']), .006)
                budget.settle(reservation, {'provider': 'openai', 'model': settings['model'],
                    'operation': 'teacher', 'status': 'ok', 'usage': {'input_tokens': 100, 'output_tokens': 20}})
                self.assertAlmostEqual(float(budget.state['accounted_usd']), .00002)
                self.assertEqual(budget.state['unknown_usage_calls'], 0)
                reserved = budget.reserve(settings, body)
                budget.settle(reserved, {'provider': 'openai', 'model': settings['model'],
                    'operation': 'teacher', 'status': 'network_error', 'usage': {}})
                self.assertEqual(budget.state['unknown_usage_calls'], 1)
                self.assertGreater(float(budget.state['accounted_usd']), .006)

    def test_actual_transport_does_not_send_when_cost_too_high(self):
        calls = []
        def opener(*a, **k):
            calls.append(1)
            return io.BytesIO(b'{}')
        with tempfile.TemporaryDirectory() as folder:
            with APIBudget(folder, max_cost_usd='.001').activate():
                client = OpenAIParent(load_config()['parent'], api_key='TEST', opener=opener)
                with self.assertRaises(BudgetExceeded):
                    client.generate_json([{'role':'system','content':'JSON'}, {'role':'user','content':'test'}], {})
            self.assertEqual(calls, [])

    def test_refuse_old_usage_without_budget_and_changed_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'api_usage.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'fresh run directory'):
                with APIBudget(root/'api_budget').activate(): pass
            (root/'api_usage.jsonl').unlink()
            with APIBudget(root/'api_budget').activate() as budget:
                budget.reserve(load_config()['parent'], None)
            with self.assertRaisesRegex(ValueError, 'limits changed'):
                with APIBudget(root/'api_budget', max_calls=99).activate(): pass


if __name__ == '__main__': unittest.main()
