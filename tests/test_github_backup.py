import hashlib
import json
import shutil
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

from distillation_loop.github_backup import upload_latest_adapter, request_json, restore_latest_release


class GitHubBackupTests(unittest.TestCase):
    def setup_adapter(self, folder):
        root=Path(folder);adapter=root/'round_01/sft/adapter';adapter.mkdir(parents=True)
        (adapter/'adapter_config.json').write_text('{}')
        (adapter/'adapter_model.safetensors').write_bytes(b'weights')
        (adapter/'tokenizer.json').write_text('{}')
        (adapter/'.env').write_text('must never upload')
        (root/'round_01/training_complete.json').write_text(json.dumps({'checkpoint':str(adapter)}))
        return root

    def test_upload_verified_and_resume_does_not_upload_again(self):
        calls=[];asset={}
        release={'id':1,'html_url':'https://github.com/owner/repo/releases/tag/test',
                 'upload_url':'https://uploads.github.com/repos/owner/repo/releases/1/assets{?name,label}'}
        def api(method,url,token,**kwargs):
            calls.append((method,url))
            if '/releases/tags/' in url: return release if asset else None
            if method=='POST' and url.endswith('/releases'):
                self.assertEqual(kwargs['payload']['prerelease'],True)
                return release
            if method=='GET' and '/assets?' in url: return [asset] if asset else []
            self.assertTrue(url.startswith('https://uploads.github.com/'))
            archive=kwargs['file']
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(set(z.namelist()),{'adapter_config.json','adapter_model.safetensors','tokenizer.json'})
            asset.update(state='uploaded',size=archive.stat().st_size,digest='sha256:'+hashlib.sha256(archive.read_bytes()).hexdigest(),name='adapter.zip',browser_download_url='https://github.com/owner/repo/releases/download/test/adapter.zip')
            return asset
        with tempfile.TemporaryDirectory() as folder,patch('distillation_loop.github_backup.request_json',side_effect=api):
            root=self.setup_adapter(folder)
            one=upload_latest_adapter(root,'secret','owner/repo')
            two=upload_latest_adapter(root,'secret','owner/repo')
            self.assertEqual(one,two)
            self.assertEqual(one['status'],'github_digest_verified')
            self.assertEqual(sum(method=='POST' and 'uploads.github.com' in url for method,url in calls),1)

    def test_missing_token_or_checkpoint_fails_before_network(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict('os.environ',{'GITHUB_TOKEN':''}),patch('distillation_loop.github_backup.request_json') as api:
            with self.assertRaises(ValueError):upload_latest_adapter(folder)
            with self.assertRaises(ValueError):upload_latest_adapter(folder,'secret')
            api.assert_not_called()

    def test_existing_mismatched_asset_is_not_overwritten_or_receipted(self):
        def api(method,url,token,**kwargs):
            self.assertEqual(method,'GET')
            if '/releases/tags/' in url:return {'id':1}
            return [{'name':'adapter.zip','state':'uploaded','size':1,'digest':'sha256:wrong'}]
        with tempfile.TemporaryDirectory() as folder,patch('distillation_loop.github_backup.request_json',side_effect=api):
            root=self.setup_adapter(folder)
            with self.assertRaisesRegex(RuntimeError,'SHA256'):upload_latest_adapter(root,'secret')
            self.assertFalse(list((root/'github_backups').glob('*.json')))

    def test_http_error_does_not_expose_credentials_or_response_body(self):
        error=urllib.error.HTTPError('https://api.github.com/',403,'SECRET',{},None)
        with patch('urllib.request.urlopen',side_effect=error):
            with self.assertRaisesRegex(RuntimeError,'HTTP 403') as raised:request_json('GET','https://api.github.com/','SECRET')
            self.assertNotIn('SECRET',str(raised.exception))

    def test_restore_latest_compatible_release_and_pin_seed(self):
        model = 'Qwen/Qwen3-8B'
        config = {
            'base_model_name_or_path': model, 'peft_type': 'LORA', 'task_type': 'CAUSAL_LM',
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            'bias': 'none', 'r': 16, 'lora_alpha': 32, 'lora_dropout': .05,
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            releases, archives = [], {}
            for asset_id, date, base in ((1, '2026-11-09', model), (2, '2026-11-10', 'other')):
                archive = root / f'{asset_id}.zip'
                with zipfile.ZipFile(archive, 'w') as bundle:
                    bundle.writestr('adapter_config.json', json.dumps(dict(config, base_model_name_or_path=base)))
                    bundle.writestr('adapter_model.safetensors', b'weights')
                digest = hashlib.sha256(archive.read_bytes()).hexdigest()
                archives[asset_id] = archive
                releases.append({
                    'tag_name': f'adapter-round_01-{digest[:16]}', 'published_at': date,
                    'prerelease': True, 'assets': [{'id': asset_id, 'name': 'adapter.zip',
                        'state': 'uploaded', 'digest': 'sha256:' + digest, 'size': archive.stat().st_size}],
                })

            def download(url, token, path, size):
                shutil.copyfile(archives[int(url.split('/')[-1])], path)

            run = root / 'run'
            with patch('distillation_loop.github_backup.request_json', return_value=releases) as api, \
                    patch('distillation_loop.github_backup.download_release_asset', side_effect=download) as fetch:
                seed = restore_latest_release(run, model, 'secret', 'owner/repo')
                self.assertEqual(seed['tag'], releases[0]['tag_name'])
                self.assertEqual(seed['training']['lora_rank'], 16)
                self.assertEqual(fetch.call_count, 2)
                self.assertEqual(restore_latest_release(run, model, 'secret', 'owner/repo'), seed)
                api.assert_called_once()
                (Path(seed['checkpoint']) / 'adapter_model.safetensors').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'pinned GitHub seed'):
                    restore_latest_release(run, model, 'secret', 'owner/repo')
                (run / 'loop_state.json').write_text('{}')
                self.assertEqual(restore_latest_release(run, model, 'secret', 'owner/repo'), seed)

            with patch('distillation_loop.github_backup.request_json', return_value=releases), \
                    patch('distillation_loop.github_backup.download_release_asset',
                          side_effect=lambda url, token, path, size: path.write_bytes(b'corrupt')):
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    restore_latest_release(root / 'bad', model, 'secret', 'owner/repo')
                self.assertFalse((root / 'bad' / 'github_seed.json').exists())
