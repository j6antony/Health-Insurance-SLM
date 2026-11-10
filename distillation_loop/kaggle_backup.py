"""Publish one immutable private Kaggle dataset per trained adapter/round."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

from self_instruct_pipeline.persistence import atomic_json


class AdapterBackupError(RuntimeError):
    pass


def cli(*args, timeout=900):
    executable = shutil.which('kaggle')
    if executable is None:
        raise AdapterBackupError('Install kaggle, then configure Kaggle API credentials before enabling adapter backups')
    try:
        return subprocess.run([executable, 'datasets', *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        raise AdapterBackupError('Kaggle backup request failed/timed out. Adapter remains local; retry the run to retry backup.') from None


def backup_prefix():
    prefix = os.environ.get('KAGGLE_ADAPTER_DATASET', '').strip()
    if prefix and not re.fullmatch(r'[A-Za-z0-9_-]+/[a-z0-9][a-z0-9-]{2,49}', prefix):
        raise AdapterBackupError('KAGGLE_ADAPTER_DATASET must be your-username/adapter-backup-prefix')
    return prefix


def check_backup_setup():
    if not backup_prefix():
        return
    if cli('list', '--mine', timeout=30).returncode:
        raise AdapterBackupError('Kaggle authentication failed. Add KAGGLE_API_TOKEN (or KAGGLE_USERNAME and KAGGLE_KEY) in Kaggle Secrets.')


def cli_diagnostic(result):
    text = getattr(result, 'stdout', '') + ' ' + getattr(result, 'stderr', '')
    code = re.search(r'\b(400|401|403|404|409|429|5\d\d)\b', text)
    return f'exit={result.returncode}' + (f', HTTP {code.group(1)}' if code else '')


def remote_manifest(folder):
    for candidate in (folder/'manifest.json', folder/'manifest.json.zip'):
        if not candidate.is_file():
            continue
        try:
            if zipfile.is_zipfile(candidate):
                with zipfile.ZipFile(candidate) as archive:
                    return json.loads(archive.read('manifest.json'))
            return json.loads(candidate.read_text())
        except (ValueError, KeyError, OSError, zipfile.BadZipFile):
            pass
    return None


def save_adapter_dataset(checkpoint, run_dir, round_id):
    prefix = backup_prefix()
    if not prefix:
        return None
    checkpoint, run_dir = Path(checkpoint), Path(run_dir)
    if not (checkpoint/'adapter_config.json').is_file() or not list(checkpoint.glob('adapter_model*.safetensors')):
        raise AdapterBackupError('Adapter backup requires adapter_config.json and safetensors weights')
    # Only model/tokenizer artifacts are included, never a run folder or credentials.
    files = sorted(p for p in checkpoint.iterdir() if p.is_file() and
        (p.name == 'adapter_config.json' or re.fullmatch(r'adapter_model(?:-\d+-of-\d+)?\.safetensors', p.name) or p.name == 'adapter_model.safetensors.index.json' or p.name in {'tokenizer.json', 'tokenizer_config.json',
         'special_tokens_map.json', 'vocab.json', 'merges.txt', 'added_tokens.json', 'chat_template.jinja'}))
    hashes = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        hashes[path.name] = digest.hexdigest()
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    backup_dir = run_dir/'adapter_backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    identity = backup_dir/'run.json'
    if not identity.exists():
        atomic_json(identity, {'id': uuid.uuid4().hex[:8]})
    run_id = json.loads(identity.read_text())['id']
    owner, slug = prefix.split('/')
    dataset = f'{owner}/{slug[:20]}-{run_id}-r{round_id:02d}-{signature[:8]}'
    receipt = backup_dir/f'round_{round_id:02d}.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['dataset'] == dataset and saved['sha256'] == signature:
            return saved
    manifest = {'dataset': dataset, 'round': round_id, 'sha256': signature,
                'files': hashes, 'format': 'PEFT LoRA adapter; base model not included'}
    print(f'Backing up round {round_id} adapter to private Kaggle dataset {dataset}', flush=True)
    pending_path = backup_dir/f'round_{round_id:02d}.pending.json'
    atomic_json(pending_path, {**manifest, 'url': 'https://www.kaggle.com/datasets/' + dataset})
    with tempfile.TemporaryDirectory(dir=backup_dir) as folder:
        stage = Path(folder)
        # Recover a previous successful upload whose local acknowledgment was lost.
        probe = cli('download', '-d', dataset, '-f', 'manifest.json', '-p', str(stage), '--force', '--quiet', timeout=30)
        remote = stage/'manifest.json'
        matched = probe.returncode == 0 and remote_manifest(stage) == manifest
        if not matched:
            for old in stage.iterdir():
                if old.is_file(): old.unlink()
            atomic_json(stage/'manifest.json', manifest)
            atomic_json(stage/'dataset-metadata.json', {
                'id': dataset, 'title': f'SLM adapter {run_id} round {round_id}',
                'licenses': [{'name': 'copyright-authors'}],
                'description': 'Private training checkpoint. PEFT adapter only; load with the original base model. Original authors retain rights. No base-model weights included.'})
            with zipfile.ZipFile(stage/'adapter.zip', 'w', compression=zipfile.ZIP_STORED) as archive:
                for path in files:
                    archive.write(path, path.name)
                archive.writestr('manifest.json', json.dumps(manifest))
            status = cli('status', dataset, timeout=30)
            if status.returncode == 0:
                status_text = (getattr(status, 'stdout', '') + ' ' + getattr(status, 'stderr', '')).lower()
                if any(word in status_text for word in ('processing', 'pending', 'running')):
                    raise AdapterBackupError(f'Kaggle dataset is still processing: https://www.kaggle.com/datasets/{dataset}. Retry verification later; adapter is saved locally.')
                # Preserve the existing private dataset. Older uploads may have
                # converted manifest.json; publish a corrected version, never delete.
                result = cli('version', '-p', str(stage), '-m', f'Preserve adapter manifest for round {round_id}', '--keep-tabular', '--quiet')
            else:
                status_text = (getattr(status, 'stdout', '') + ' ' + getattr(status, 'stderr', '')).lower()
                if any(code in status_text for code in ('401', '403', '429')):
                    raise AdapterBackupError('Cannot inspect Kaggle dataset: ' + cli_diagnostic(status) + '. Check dataset read/write permissions.')
                result = cli('create', '-p', str(stage), '--keep-tabular', '--quiet')  # Private by default.

            if result.returncode:
                raise AdapterBackupError('Kaggle adapter upload failed (' + cli_diagnostic(result) + '). Adapter is local. Inspect adapter_backups/round_XX.pending.json for the dataset URL; rerun after correcting upload access. No completed training will be repeated.')
        if not matched:
            # A successful create may still be processing. Confirm remote access
            # before marking this round safely backed up.
            for attempt in range(12):
                remote.unlink(missing_ok=True)
                (stage/'manifest.json.zip').unlink(missing_ok=True)
                check = cli('download', '-d', dataset, '-f', 'manifest.json', '-p', str(stage), '--force', '--quiet', timeout=30)
                if check.returncode == 0 and remote_manifest(stage) == manifest:
                    matched = True
                    break
                print(f'Waiting for Kaggle dataset processing ({attempt + 1}/12)', flush=True)
                time.sleep(5)
            if not matched:
                raise AdapterBackupError(f'Kaggle upload submitted but remote manifest is not yet readable ({cli_diagnostic(check)}). Dataset: https://www.kaggle.com/datasets/{dataset}. Adapter remains local; rerun to verify without retraining.')
        saved = {**manifest, 'url': 'https://www.kaggle.com/datasets/' + dataset,
                 'status': 'remote_manifest_verified'}
        atomic_json(receipt, saved)
        pending_path.unlink(missing_ok=True)
        print('Adapter backup:', saved['url'], flush=True)
        return saved
