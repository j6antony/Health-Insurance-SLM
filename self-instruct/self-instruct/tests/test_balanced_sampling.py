"""Compatibility-entry tests; full pipeline tests live in workspace/tests."""
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

WORKSPACE = pathlib.Path(__file__).resolve().parents[3]
ENTRY = WORKSPACE / 'self-instruct/self-instruct/self_instruct/bootstrap_instructions.py'
spec = importlib.util.spec_from_file_location('bootstrap_entry', ENTRY)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class BalancedEntryTests(unittest.TestCase):
    def test_legacy_argument_translation(self):
        self.assertEqual(entry.translate_arguments(['--dry_run', '--num_instructions_to_generate', '2', '--batch_dir', '/tmp/preview']),
                         ['dry-run', '--target', '2', '--output-dir', '/tmp/preview'])

    def test_offline_compatibility_entry(self):
        with tempfile.TemporaryDirectory() as output:
            result = subprocess.run([sys.executable, str(ENTRY), '--dry_run', '--batch_dir', output],
                                    capture_output=True, text=True, cwd='/tmp', check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(len(plan['seed_ids']), 8)
            self.assertEqual(sum(plan['task_quotas'].values()), 8)
            self.assertEqual([m['role'] for m in plan['messages']], ['system', 'user'])
            self.assertFalse(list(pathlib.Path(output).glob('*.jsonl')))


if __name__ == '__main__':
    unittest.main()
