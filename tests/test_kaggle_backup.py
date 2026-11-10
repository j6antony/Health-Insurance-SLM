import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from distillation_loop.kaggle_backup import save_adapter_dataset, AdapterBackupError


class KaggleBackupTests(unittest.TestCase):
    def test_private_dataset_upload_allowlist_and_resume_receipt(self):
        calls = []; remote = {}
        def cli(*args, **kwargs):
            calls.append(args)
            if args[0] == 'status':
                return SimpleNamespace(returncode=1, stderr='404 not found')
            if args[0] == 'download':
                stage = Path(args[args.index('-p') + 1])
                if remote:
                    (stage/'manifest.json').write_text(json.dumps(remote))
                    return SimpleNamespace(returncode=0)
                return SimpleNamespace(returncode=1)
            self.assertEqual(args[0], 'create')
            self.assertNotIn('--public', args)
            stage = Path(args[args.index('-p') + 1])
            remote.update(json.loads((stage/'manifest.json').read_text()))
            with zipfile.ZipFile(stage/'adapter.zip') as archive:
                self.assertEqual(set(archive.namelist()), {'adapter_config.json','adapter_model.safetensors','tokenizer.json','manifest.json'})
            self.assertIn('--keep-tabular', args)
            self.assertNotIn('SECRET', ''.join(p.read_text() for p in stage.glob('*.json')))
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {'KAGGLE_ADAPTER_DATASET':'owner/slm-adapters'}), patch('distillation_loop.kaggle_backup.cli', side_effect=cli):
            root = Path(folder); adapter = root/'adapter';adapter.mkdir()
            (adapter/'adapter_config.json').write_text('{}')
            (adapter/'adapter_model.safetensors').write_bytes(b'weights')
            (adapter/'tokenizer.json').write_text('{}')
            (adapter/'.env').write_text('SECRET')
            one = save_adapter_dataset(adapter, root, 1)
            count = len(calls)
            self.assertEqual(save_adapter_dataset(adapter, root, 1), one)
            self.assertEqual(len(calls), count)
            # Lost local receipt recovers from the remote manifest, without creating again.
            (root/'adapter_backups/round_01.json').unlink()
            save_adapter_dataset(adapter, root, 1)
            self.assertEqual(sum(call[0] == 'create' for call in calls), 1)

    def test_existing_dataset_gets_corrected_version_not_duplicate_create(self):
        remote = {}; calls = []
        def cli(*args, **kwargs):
            calls.append(args)
            if args[0] == 'status': return SimpleNamespace(returncode=0, stdout='ready')
            stage = Path(args[args.index('-p') + 1])
            if args[0] == 'download':
                if remote:
                    (stage/'manifest.json').write_text(json.dumps(remote))
                    return SimpleNamespace(returncode=0)
                return SimpleNamespace(returncode=1)
            self.assertEqual(args[0], 'version')
            self.assertIn('--keep-tabular', args)
            remote.update(json.loads((stage/'manifest.json').read_text()))
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {'KAGGLE_ADAPTER_DATASET':'owner/slm-adapters'}), patch('distillation_loop.kaggle_backup.cli', side_effect=cli):
            root = Path(folder); adapter = root/'adapter';adapter.mkdir()
            (adapter/'adapter_config.json').write_text('{}')
            (adapter/'adapter_model.safetensors').write_bytes(b'weights')
            receipt = save_adapter_dataset(adapter, root, 1)
            self.assertEqual(receipt['status'], 'remote_manifest_verified')
            self.assertFalse(any(args[0] == 'create' for args in calls))

    def test_upload_failure_retains_local_adapter(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {'KAGGLE_ADAPTER_DATASET':'owner/slm-adapters'}), patch('distillation_loop.kaggle_backup.cli', return_value=SimpleNamespace(returncode=1)):
            root = Path(folder);adapter=root/'adapter';adapter.mkdir()
            (adapter/'adapter_config.json').write_text('{}')
            (adapter/'adapter_model.safetensors').write_bytes(b'weights')
            with self.assertRaises(AdapterBackupError):save_adapter_dataset(adapter,root,1)
            self.assertTrue((adapter/'adapter_model.safetensors').exists())
            self.assertFalse((root/'adapter_backups/round_01.json').exists())

    def test_status_403_authenticated_private_create_and_remote_verification(self):
        calls = []; remote = {}
        def fake(*args, **kwargs):
            calls.append(args)
            if args[0] == 'status': return SimpleNamespace(returncode=1, stderr='403 Forbidden')
            if args[0] == 'list': return SimpleNamespace(returncode=0)
            stage = Path(args[args.index('-p') + 1])
            if args[0] == 'download':
                if not remote: return SimpleNamespace(returncode=1, stderr='403 Forbidden')
                (stage/'manifest.json').write_text(json.dumps(remote))
                return SimpleNamespace(returncode=0)
            self.assertEqual(args[0], 'create')
            self.assertNotIn('--public', args)
            remote.update(json.loads((stage/'manifest.json').read_text()))
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {'KAGGLE_ADAPTER_DATASET':'owner/slm-adapters'}), patch('distillation_loop.kaggle_backup.cli', side_effect=fake):
            root=Path(folder);adapter=root/'adapter';adapter.mkdir()
            (adapter/'adapter_config.json').write_text('{}')
            (adapter/'adapter_model.safetensors').write_bytes(b'weights')
            result=save_adapter_dataset(adapter,root,1)
            self.assertEqual(result['status'],'remote_manifest_verified')
            self.assertIn(('list','--mine'),calls)
            self.assertEqual(sum(c[0]=='create' for c in calls),1)

    def test_status_403_failed_authentication_never_attempts_upload(self):
        calls=[]
        def fake(*args, **kwargs):
            calls.append(args)
            return SimpleNamespace(returncode=1,stderr='403 Forbidden')
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {'KAGGLE_ADAPTER_DATASET':'owner/slm-adapters'}), patch('distillation_loop.kaggle_backup.cli', side_effect=fake):
            root=Path(folder);adapter=root/'adapter';adapter.mkdir()
            (adapter/'adapter_config.json').write_text('{}')
            (adapter/'adapter_model.safetensors').write_bytes(b'weights')
            with self.assertRaisesRegex(AdapterBackupError,'authentication check failed'): save_adapter_dataset(adapter,root,1)
            self.assertFalse(any(c[0] in ('create','version') for c in calls))
            self.assertFalse((root/'adapter_backups/round_01.json').exists())

    def test_private_creation_permission_denied_keeps_pending_without_receipt(self):
        def fake(*args, **kwargs):
            if args[0]=='list': return SimpleNamespace(returncode=0)
            return SimpleNamespace(returncode=1,stderr='403 Forbidden')
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {'KAGGLE_ADAPTER_DATASET':'owner/slm-adapters'}), patch('distillation_loop.kaggle_backup.cli', side_effect=fake):
            root=Path(folder);adapter=root/'adapter';adapter.mkdir()
            (adapter/'adapter_config.json').write_text('{}')
            (adapter/'adapter_model.safetensors').write_bytes(b'weights')
            with self.assertRaisesRegex(AdapterBackupError,'creation'): save_adapter_dataset(adapter,root,1)
            self.assertTrue((root/'adapter_backups/round_01.pending.json').exists())
            self.assertFalse((root/'adapter_backups/round_01.json').exists())
            self.assertTrue((adapter/'adapter_model.safetensors').exists())
