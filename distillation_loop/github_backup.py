"""Upload and restore completed adapters using GitHub release assets."""
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

REPOSITORY = 'j6antony/Health-Insurance-SLM'



ADAPTER_FILES = {'adapter_config.json', 'adapter_model.safetensors.index.json', 'tokenizer.json',
                 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt',
                 'added_tokens.json', 'chat_template.jinja', 'README.md'}
MAX_ADAPTER_BYTES = 512 * 1024 * 1024


def adapter_filename(name):
    return name in ADAPTER_FILES or re.fullmatch(r'adapter_model(?:-\d+-of-\d+)?\.safetensors', name)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def upload_inference_predictions(path, release, token):
    """Attach an immutable results asset to the adapter release it evaluates."""
    if (release.get('repository') != REPOSITORY
            or not re.fullmatch(r'adapter-round_\d+-[0-9a-f]{16}', release.get('tag', ''))
            or not re.fullmatch(r'[0-9a-f]{64}', release.get('sha256', ''))
            or not release['sha256'].startswith(release['tag'].rsplit('-', 1)[1])):
        raise ValueError('Expected a verified adapter release from the configured repository')
    path = Path(path)
    digest = file_sha256(path)
    name = f'inference-review-{digest}.jsonl'
    api = f'https://api.github.com/repos/{REPOSITORY}'
    tag = urllib.parse.quote(release['tag'], safe='')
    published = request_json('GET', f'{api}/releases/tags/{tag}', token)
    assets_url = f'{api}/releases/{int(published["id"])}/assets'
    query = urllib.parse.urlencode({'name': name})
    page = 1
    while True:
        assets = request_json('GET', f'{assets_url}?per_page=100&page={page}', token)
        asset = next((item for item in assets if item['name'] == name), None)
        if asset is not None or len(assets) < 100:
            break
        page += 1
    if asset is None:
        upload_url = f'https://uploads.github.com/repos/{REPOSITORY}/releases/{int(published["id"])}/assets'
        asset = request_json('POST', f'{upload_url}?{query}', token, file=path,
                             content_type='application/x-ndjson')
    if (asset.get('state') != 'uploaded' or asset.get('size') != path.stat().st_size
            or asset.get('digest') != f'sha256:{digest}'):
        raise RuntimeError('GitHub did not confirm the expected prediction asset size and SHA-256.')
    return asset['browser_download_url']


def request_json(method, url, token, *, payload=None, file=None, allow_missing=False,
                 content_type='application/zip'):
    headers = {'Accept': 'application/vnd.github+json',
               'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'health-insurance-slm-adapter-backup'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    data = json.dumps(payload).encode() if payload is not None else None
    if payload is not None:
        headers['Content-Type'] = 'application/json'
    if file is not None:
        headers.update({'Content-Type': content_type, 'Content-Length': str(Path(file).stat().st_size)})
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
    files = sorted(p for p in checkpoint.iterdir() if p.is_file() and not p.is_symlink()
                   and adapter_filename(p.name))
    directory = run_dir/'github_backups'; directory.mkdir(exist_ok=True)
    archive = directory/'adapter.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as output:
        for path in files:
            output.write(path, path.name)
    digest = file_sha256(archive)
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


def download_release_asset(url, token, path, size):
    request = urllib.request.Request(url, headers={
        'Accept': 'application/octet-stream', 'X-GitHub-Api-Version': '2022-11-28',
        'User-Agent': 'health-insurance-slm-adapter-backup',
    })
    # GitHub may redirect to signed object storage. Keep the token on the API request.
    if token:
        request.add_unredirected_header('Authorization', f'Bearer {token}')
    with urllib.request.urlopen(request, timeout=120) as response, path.open('wb') as output:
        remaining = size
        while remaining:
            block = response.read(min(1024 * 1024, remaining))
            if not block:
                break
            output.write(block)
            remaining -= len(block)
        if remaining or response.read(1):
            raise ValueError('GitHub adapter asset size does not match the release metadata')


def adapter_training_settings(adapter, model_name):
    cfg = json.loads((Path(adapter) / 'adapter_config.json').read_text())
    expected_modules = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    if (cfg.get('base_model_name_or_path') != model_name or cfg.get('peft_type') != 'LORA'
            or cfg.get('task_type') != 'CAUSAL_LM'
            or set(cfg.get('target_modules') or []) != expected_modules or cfg.get('bias') != 'none'
            or any(cfg.get(key) for key in ('revision', 'rank_pattern', 'alpha_pattern', 'modules_to_save',
                                            'use_dora', 'use_rslora', 'lora_bias'))):
        return None
    rank, alpha, dropout = cfg.get('r'), cfg.get('lora_alpha'), cfg.get('lora_dropout')
    if (type(rank) is not int or rank < 1 or type(alpha) is not int or alpha < 1
            or type(dropout) not in (float, int) or not 0 <= dropout < 1):
        return None
    return {'lora_rank': rank, 'lora_alpha': alpha, 'lora_dropout': dropout}


def restore_latest_release(run_dir, model_name, token=None, repository=REPOSITORY):
    """Pin a verified warm-start adapter for a fresh local run."""
    from self_instruct_pipeline.persistence import Store, atomic_json

    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
        raise ValueError('Expected owner/repository')
    token = (token or os.environ.get('GITHUB_TOKEN', '')).strip()
    store = Store(run_dir)
    receipt_path = store.path('github_seed.json')
    adapter = store.path('github_seed_adapter')
    with store.lock():
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['repository'] != repository:
                raise ValueError('The run was initialized from a different GitHub repository')
            if store.path('loop_state.json').exists():
                return receipt
            if (adapter_training_settings(adapter, model_name) != receipt['training']
                    or any(file_sha256(adapter / name) != digest
                           for name, digest in receipt['files'].items())):
                raise ValueError('The pinned GitHub seed adapter has changed')
            return receipt
        if store.path('loop_state.json').exists():
            return None

        api = f'https://api.github.com/repos/{repository}'
        candidates = []
        for page in range(1, 1001):
            releases = request_json('GET', f'{api}/releases?per_page=100&page={page}', token)
            for release in releases:
                tag = release.get('tag_name', '')
                match = re.fullmatch(r'adapter-round_\d+-([0-9a-f]{16})', tag)
                if release.get('draft') or not release.get('published_at') or not match:
                    continue
                for asset in release.get('assets', []):
                    if asset.get('name') == 'adapter.zip' and asset.get('state') == 'uploaded':
                        candidates.append((release, asset, match[1]))
            if len(releases) < 100:
                break
        candidates.sort(key=lambda candidate: (candidate[0]['published_at'], candidate[1]['id']),
                        reverse=True)
        for release, asset, tag_digest in candidates:
            digest = (asset.get('digest') or '').removeprefix('sha256:')
            size = asset.get('size')
            if (not re.fullmatch(r'[0-9a-f]{64}', digest) or not digest.startswith(tag_digest)
                    or type(size) is not int or not 0 < size <= MAX_ADAPTER_BYTES):
                raise ValueError(f'Invalid asset metadata for {release["tag_name"]}')
            with tempfile.TemporaryDirectory(dir=store.directory) as temporary:
                stage = Path(temporary)
                archive = stage / 'adapter.zip'
                url = (f'{api}/releases/assets/{int(asset["id"])}' if token else
                       f'https://github.com/{repository}/releases/download/{release["tag_name"]}/adapter.zip')
                download_release_asset(url, token, archive, size)
                if file_sha256(archive) != digest:
                    raise ValueError(f'GitHub adapter SHA256 mismatch for {release["tag_name"]}')
                extracted = stage / 'adapter'
                extracted.mkdir()
                with zipfile.ZipFile(archive) as bundle:
                    entries = bundle.infolist()
                    names = {entry.filename for entry in entries}
                    if (len(names) != len(entries)
                            or sum(entry.file_size for entry in entries) > MAX_ADAPTER_BYTES
                            or any(not adapter_filename(name) for name in names)
                            or 'adapter_config.json' not in names
                            or 'adapter_model.safetensors' not in names):
                        raise ValueError(f'Unexpected adapter ZIP contents for {release["tag_name"]}')
                    for entry in entries:
                        with bundle.open(entry) as source, (extracted / entry.filename).open('wb') as output:
                            shutil.copyfileobj(source, output)
                training = adapter_training_settings(extracted, model_name)
                if training is None:
                    print(f'Skipping incompatible release {release["tag_name"]}.', flush=True)
                    continue
                receipt = {'repository': repository, 'tag': release['tag_name'], 'sha256': digest,
                           'checkpoint': str(adapter), 'training': training,
                           'files': {path.name: file_sha256(path) for path in extracted.iterdir()}}
                if adapter.exists():
                    # A previous attempt may have stopped before writing the receipt.
                    shutil.rmtree(adapter)
                extracted.replace(adapter)
                atomic_json(receipt_path, receipt)
                return receipt
        return None
