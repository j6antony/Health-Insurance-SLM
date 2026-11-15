"""Content identities for saved adapters, cached while their file metadata is unchanged."""
from functools import lru_cache
from pathlib import Path
import re

from self_instruct_pipeline.persistence import digest

from .github_backup import file_sha256


def signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=256)
def checked_hash(path, expected):
    result = file_sha256(path)
    if signature(path) != expected:
        raise ValueError(f'Adapter file changed while hashing: {path}')
    return result


def adapter_files(checkpoint):
    root = Path(checkpoint).resolve()
    config = root / 'adapter_config.json'
    weights = sorted(path for path in root.glob('adapter_model*.safetensors')
                     if re.fullmatch(r'adapter_model(?:-\d+-of-\d+)?\.safetensors', path.name))
    index = root / 'adapter_model.safetensors.index.json'
    files = [config, *weights, *([index] if index.exists() else [])]
    if not weights or any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError(f'Adapter configuration or weights are missing or invalid: {root}')
    before = {path: signature(path) for path in files}
    hashes = {path.name: checked_hash(path, stat) for path, stat in before.items()}
    if any(signature(path) != stat for path, stat in before.items()):
        raise ValueError(f'Adapter changed while verifying files: {root}')
    return hashes


def checkpoint_binding(checkpoint):
    path = Path(checkpoint).resolve()
    return str(path), digest(adapter_files(path))
