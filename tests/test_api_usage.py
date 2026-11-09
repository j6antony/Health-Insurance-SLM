import csv
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from self_instruct_pipeline.config import load_config
from self_instruct_pipeline.parent_backend import OpenAIParent, ParentError
from self_instruct_pipeline.usage import UsageLogger, export_usage, usage_row


def event(**changes):
    return {'provider': 'openai', 'model': 'gpt-6-luna', 'operation': 'teacher',
            'status': 'ok', 'usage': {'input_tokens': 1000, 'output_tokens': 200,
              'total_tokens': 1200, 'input_tokens_details': {'cached_tokens': 400},
              'output_tokens_details': {'reasoning_tokens': 150}}, **changes}


class UsageTests(unittest.TestCase):
    def test_cached_and_reasoning_cost(self):
        row = usage_row(event())
        self.assertEqual(row['estimated_cost_usd'], '0.000164000000')
        self.assertEqual(row['reasoning_tokens'], 150)

    def test_unknown_and_no_generation(self):
        self.assertNotIn('estimated_cost_usd', usage_row(event(usage={})))
        self.assertEqual(usage_row(event(operation='model_check', usage={}))['estimated_cost_usd'], '0.000000000000')
        self.assertEqual(usage_row(event(model='gpt-other'))['cost_status'], 'pricing_unavailable')

    def test_long_context_and_invalid_usage(self):
        row = usage_row(event(usage={'input_tokens': 300000, 'output_tokens': 100}))
        self.assertEqual(row['estimated_cost_usd'], '0.060075000000')
        row = usage_row(event(usage={'input_tokens': 1, 'output_tokens': 2, 'input_tokens_details': {'cached_tokens': 9}}))
        self.assertEqual(row['cost_status'], 'invalid_usage')

    def test_resume_and_aggregate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            logger = UsageLogger(root/'self_instruct')
            logger(event())
            # Simulate interrupted CSV write, then rebuild from authoritative JSONL.
            (root/'self_instruct/api_usage.csv').write_text('broken')
            logger = UsageLogger(root/'self_instruct')
            logger(event(status='network_error', usage={}))
            UsageLogger(root/'development_generation')(event())
            summary = export_usage(root)
            self.assertEqual(summary['calls'], 3)
            self.assertEqual(summary['unknown_cost_calls'], 1)
            self.assertEqual(float(summary['estimated_cost_usd']), .000328)
            self.assertEqual(export_usage(root), summary)
            with (root/'api_usage.csv').open(encoding='utf-8-sig') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)

    def test_retry_transport_and_incomplete_logged(self):
        logs = []
        calls = []
        def opener(request, timeout):
            calls.append(1)
            if len(calls) == 1:
                raise urllib.error.HTTPError(request.full_url, 429, 'SECRET', {}, None)
            return io.BytesIO(json.dumps({'status': 'incomplete', 'id': 'resp_test',
                                          'usage': event()['usage']}).encode())
        client = OpenAIParent(load_config()['parent'], api_key='SECRET', opener=opener,
                              logger=logs.append, sleep=lambda _: None)
        with self.assertRaises(ParentError):
            client.generate_json([{'role': 'system', 'content': 'SECRET'},
                                  {'role': 'user', 'content': 'SECRET'}], {})
        self.assertEqual([x['status'] for x in logs], ['http_error', 'ok'])
        self.assertEqual(logs[1]['response_status'], 'incomplete')
        self.assertEqual(logs[1]['attempt'], 2)
        self.assertIn('timestamp', logs[0])
        self.assertIn('duration_seconds', logs[0])
        self.assertNotIn('SECRET', json.dumps(logs))
        self.assertEqual(usage_row(logs[1])['estimated_cost_usd'], '0.000164000000')

    def test_log_failure_never_retries_billed_request(self):
        calls = []
        def opener(*args, **kwargs):
            calls.append(1)
            return io.BytesIO(b'{"id":"test"}')
        def logger(event):
            raise OSError('Disk full')
        client = OpenAIParent(load_config()['parent'], api_key='SECRET',
                              opener=opener, logger=logger)
        with self.assertRaisesRegex(OSError, 'Disk full'):
            client.check_model()
        self.assertEqual(len(calls), 1)

    def test_malformed_transport_logged(self):
        logs = []
        client = OpenAIParent(load_config()['parent'], api_key='SECRET',
                              opener=lambda *a, **k: io.BytesIO(b'bad'), logger=logs.append)
        with self.assertRaises(ParentError):
            client.check_model()
        self.assertEqual(logs[0]['status'], 'malformed_transport')


if __name__ == '__main__':
    unittest.main()
