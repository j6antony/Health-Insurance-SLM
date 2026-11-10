import hashlib
import json
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

from distillation_loop.github_backup import upload_latest_adapter, request_json


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
