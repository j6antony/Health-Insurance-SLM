"""Upload a completed local adapter to an immutable GitHub prerelease asset."""
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

REPOSITORY = 'j6antony/Health-Insurance-SLM'


def request_json(method, url, token, *, payload=None, file=None, allow_missing=False):
    headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json',
               'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'health-insurance-slm-adapter-backup'}
    data = json.dumps(payload).encode() if payload is not None else None
    if payload is not None:
        headers['Content-Type'] = 'application/json'
    if file is not None:
        headers.update({'Content-Type': 'application/zip', 'Content-Length': str(Path(file).stat().st_size)})
    try:
        if file is not None:
            with Path(file).open('rb') as stream:
                request = urllib.request.Request(url, data=stream, headers=headers, method=method)
                with urllib.request.urlopen(request, timeout=900) as response:
                    return json.load(response)
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        if allow_missing and status == 404:
            return None
        raise RuntimeError(f'GitHub {method} failed: HTTP {status}. Check repository access and Contents read/write permission. Local adapter is unchanged; rerun to resume.') from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RuntimeError('GitHub connection failed. Local adapter is unchanged; rerun to resume.') from None


def upload_latest_adapter(run_dir, token=None, repository=REPOSITORY):
    token = token or os.environ.get('GITHUB_TOKEN', '')
    if not isinstance(token, str) or not token.strip():
        raise ValueError('Set GITHUB_TOKEN through Kaggle Secrets; never put it in source code.')
    token = token.strip()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
        raise ValueError('Expected owner/repository')
    run_dir = Path(run_dir)
    completed = sorted(run_dir.glob('round_*/training_complete.json'),
                       key=lambda p: int(p.parent.name.split('_')[-1]))
    if not completed:
        raise ValueError('No completed training checkpoint in RUN_DIR')
    checkpoint = Path(json.loads(completed[-1].read_text())['checkpoint'])
    if not (checkpoint/'adapter_config.json').is_file() or not list(checkpoint.glob('adapter_model*.safetensors')):
        raise ValueError('Completed adapter checkpoint is missing its configuration or weights')
    names = {'adapter_config.json', 'adapter_model.safetensors.index.json', 'tokenizer.json',
             'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt',
             'added_tokens.json', 'chat_template.jinja', 'README.md'}
    files = sorted(p for p in checkpoint.iterdir() if p.is_file() and not p.is_symlink() and
                   (p.name in names or re.fullmatch(r'adapter_model(?:-\d+-of-\d+)?\.safetensors', p.name)))
    directory = run_dir/'github_backups'; directory.mkdir(exist_ok=True)
    archive = directory/'adapter.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as output:
        for path in files:
            output.write(path, path.name)
    sha = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            sha.update(block)
    digest = sha.hexdigest()
    tag = f'adapter-{completed[-1].parent.name}-{digest[:16]}'
    api = f'https://api.github.com/repos/{repository}'
    release = request_json('GET', api+'/releases/tags/'+tag, token, allow_missing=True)
    if release is None:
        release = request_json('POST', api+'/releases', token, payload={
            'tag_name': tag, 'target_commitish': 'Distillation', 'name': tag,
            'body': 'Qwen/Qwen3-8B PEFT training checkpoint. Not a merged base model or a quality-approved release.\nZIP SHA256: '+digest,
            'draft': False, 'prerelease': True, 'make_latest': 'false'})
    assets = request_json('GET', api+f'/releases/{release["id"]}/assets?per_page=100', token)
    existing = next((asset for asset in assets if asset['name'] == 'adapter.zip'), None)
    if existing is None:
        upload_url = release['upload_url'].split('{')[0]
        parsed = urllib.parse.urlsplit(upload_url)
        if parsed.scheme != 'https' or parsed.netloc != 'uploads.github.com':
            raise ValueError('Unexpected GitHub asset upload host')
        asset = request_json('POST', upload_url+'?name=adapter.zip', token, file=archive)
    else:
        asset = existing
    if (asset.get('state') != 'uploaded' or asset.get('size') != archive.stat().st_size
            or asset.get('digest') != 'sha256:'+digest):
        raise RuntimeError('GitHub asset is not confirmed with the expected size and SHA256. No asset was overwritten. Local ZIP remains available.')
    receipt = {'release_url': release['html_url'], 'download_url': asset['browser_download_url'],
               'sha256': digest, 'checkpoint': str(checkpoint), 'status': 'github_digest_verified'}
    from self_instruct_pipeline.persistence import atomic_json
    atomic_json(directory/(tag+'.json'), receipt)
    print('Adapter uploaded and SHA256 verified:', receipt['download_url'], flush=True)
    return receipt
