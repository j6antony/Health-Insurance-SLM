import csv
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from self_instruct_pipeline.usage import RATES, UsageLogger, export_usage, usage_row
from self_instruct_pipeline.usage_cloud import SupabaseLedger, UsageSyncError
from test_api_usage import event


class MemoryLedger:
    ledger_id = 'TEST_PROJECT'
    def __init__(self): self.rows = {}; self.fail = False
    def check(self): pass
    def upload(self, events):
        for row in events: self.rows.setdefault(row['event_id'], row.copy())
        if self.fail: raise UsageSyncError('Lost upload acknowledgment')


class CloudUsageTests(unittest.TestCase):
    def test_idempotent_sync_and_resume_after_lost_ack(self):
        ledger = MemoryLedger()
        with tempfile.TemporaryDirectory() as folder, patch.object(SupabaseLedger, 'from_environment', return_value=ledger):
            logger = UsageLogger(Path(folder)/'self_instruct')
            ledger.fail = True
            with self.assertRaises(UsageSyncError): logger(event())
            local = logger.store.read('api_usage.jsonl')
            self.assertEqual(len(local), 1)
            self.assertEqual(len(ledger.rows), 1)
            self.assertEqual(logger.store.read('api_usage_synced.jsonl'), [])
            ledger.fail = False
            resumed = UsageLogger(Path(folder)/'self_instruct')
            self.assertEqual(len(ledger.rows), 1)
            self.assertEqual(resumed.store.read('api_usage_synced.jsonl')[0]['event_id'], local[0]['event_id'])
            self.assertEqual(resumed.sync_pending(), 0)
            resumed(event())
            self.assertEqual(len(ledger.rows), 2)

    def test_switching_project_resyncs_the_same_stable_events(self):
        first, second = MemoryLedger(), MemoryLedger()
        second.ledger_id = 'OTHER_PROJECT'
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(SupabaseLedger, 'from_environment', return_value=first):
                UsageLogger(folder)(event())
            with patch.object(SupabaseLedger, 'from_environment', return_value=second):
                UsageLogger(folder)
            self.assertEqual(set(first.rows), set(second.rows))
            self.assertEqual(len(second.rows), 1)

    def test_cost_snapshots_survive_rate_change_and_share_run_id(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            UsageLogger(root/'self_instruct')(event())
            UsageLogger(root/'development_generation')(event())
            with patch.dict(RATES, {('openai', 'gpt-6-luna'): ('100', '10', '500', '125')}):
                result = export_usage(root)
            self.assertEqual(float(result['estimated_cost_usd']), .000328)
            with (root/'api_usage.csv').open(encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['run_id'], rows[1]['run_id'])
            self.assertNotEqual(rows[0]['event_id'], rows[1]['event_id'])
            self.assertEqual(rows[0]['cost_origin'], 'recorded_at_call')

    def test_legacy_import_assigns_stable_ids_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'api_usage.jsonl').write_text(json.dumps(event())+'\n')
            one = UsageLogger(root).store.read('api_usage.jsonl')[0]
            two = UsageLogger(root).store.read('api_usage.jsonl')[0]
            self.assertEqual(one, two)
            self.assertEqual(one['cost_origin'], 'legacy_reconstructed')
            self.assertEqual(usage_row(one)['estimated_cost_usd'], '0.000164000000')

    def test_new_secret_header_upload_and_export_pagination(self):
        requests = []
        pages = [[{'source': 'self_instruct', 'details': usage_row(event())}],
                 [{'source': 'development_generation', 'details': usage_row(event(usage={}))}], []]
        def opener(request, timeout):
            requests.append(request)
            self.assertEqual(request.get_header('Apikey'), 'sb_secret_TEST')
            self.assertIsNone(request.get_header('Authorization'))
            self.assertNotIn('sb_secret_TEST', request.full_url)
            if request.method == 'POST':
                self.assertIn('ignore-duplicates', request.get_header('Prefer'))
                body = json.loads(request.data)
                self.assertNotIn('PROMPT SECRET', json.dumps(body))
                self.assertNotIn('sb_secret_TEST', json.dumps(body))
                return io.BytesIO(b'')
            return io.BytesIO(json.dumps(pages.pop(0)).encode())
        ledger = SupabaseLedger('https://project.supabase.co', 'sb_secret_TEST', opener=opener)
        record = event(event_id='11111111-1111-4111-8111-111111111111',
                       run_id='22222222-2222-4222-8222-222222222222', source='self_instruct', prompt='PROMPT SECRET')
        ledger.upload([record])
        with tempfile.TemporaryDirectory() as folder:
            summary = ledger.export_csv(Path(folder)/'history.csv')
            self.assertEqual(summary['calls'], 2)
            self.assertEqual(summary['unknown_cost_calls'], 1)
            self.assertEqual(float(summary['estimated_cost_usd']), .000164)
        self.assertIn('offset=1', requests[2].full_url)
        self.assertIn('offset=2', requests[3].full_url)

    def test_cloud_error_redacts_body_and_credentials(self):
        def opener(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 401, 'SECRET', {}, io.BytesIO(b'SECRET'))
        ledger = SupabaseLedger('https://project.supabase.co', 'sb_secret_SECRET', opener=opener)
        with self.assertRaisesRegex(UsageSyncError, 'HTTP 401') as caught: ledger.check()
        self.assertNotIn('SECRET', str(caught.exception))

    def test_incomplete_configuration_is_not_silently_local(self):
        with patch.dict('os.environ', {'SUPABASE_URL': 'https://project.supabase.co'}, clear=True):
            with self.assertRaises(UsageSyncError): SupabaseLedger.from_environment()
