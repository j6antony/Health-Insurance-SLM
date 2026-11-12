"""Versioned adapter ZIPs stored in a mounted Google Drive run directory."""
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from self_instruct_pipeline.persistence import atomic_json, digest


def file_hash(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def verify_archive(archive, manifest):
    try:
        with zipfile.ZipFile(archive) as saved:
            if set(saved.namelist()) != set(manifest['files']) | {'manifest.json'}:
                return False
            if json.loads(saved.read('manifest.json')) != manifest:
                return False
            for name, expected in manifest['files'].items():
                actual=hashlib.sha256()
                with saved.open(name) as stream:
                    for block in iter(lambda: stream.read(1024*1024), b''):
                        actual.update(block)
                if actual.hexdigest() != expected:
                    return False
        return True
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError):
        return False


def save_drive_adapter(checkpoint, run_dir, round_id):
    checkpoint, run_dir=Path(checkpoint),Path(run_dir)
    if not (checkpoint/'adapter_config.json').is_file() or not list(checkpoint.glob('adapter_model*.safetensors')):
        raise ValueError('Missing completed adapter files; no Google Drive backup was recorded')
    allowed={'adapter_config.json','adapter_model.safetensors.index.json','tokenizer.json',
             'tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt',
             'added_tokens.json','chat_template.jinja','README.md'}
    files=sorted(p for p in checkpoint.iterdir() if p.is_file() and not p.is_symlink() and
                 (p.name in allowed or re.fullmatch(r'adapter_model(?:-\d+-of-\d+)?\.safetensors',p.name)))
    hashes={path.name:file_hash(path) for path in files}
    manifest={'round':round_id,'files':hashes,'format':'PEFT LoRA adapter; use original Qwen/Qwen3-8B base'}
    directory=run_dir/'drive_backups';directory.mkdir(parents=True,exist_ok=True)
    archive=directory/f'round_{round_id:02d}-{digest(hashes)[:12]}.zip'
    if not verify_archive(archive,manifest):
        temporary=archive.with_suffix('.zip.tmp')
        try:
            with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED) as output:
                # Fixed ZIP metadata keeps the checksum stable when a backup is rebuilt.
                for path in files:
                    entry = zipfile.ZipInfo(path.name)
                    entry.file_size = path.stat().st_size
                    with path.open('rb') as source, output.open(entry, 'w') as destination:
                        shutil.copyfileobj(source, destination)
                output.writestr(zipfile.ZipInfo('manifest.json'), json.dumps(manifest, sort_keys=True))
            if not verify_archive(temporary,manifest):
                raise ValueError('Google Drive adapter archive verification failed; completed training remains saved')
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    receipt={'status':'drive_archive_verified','archive':str(archive),
             'sha256':file_hash(archive),'checkpoint':str(checkpoint),'round':round_id}
    atomic_json(directory/f'round_{round_id:02d}.json',receipt)
    print('Google Drive adapter saved and archive verified:',archive,flush=True)
    return receipt
