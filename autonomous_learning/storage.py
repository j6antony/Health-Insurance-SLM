"""Fail-closed local storage: volume identity, durable writes and a single writer."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

DEFAULT_ROOT = Path('/Volumes/Extreme SSD/Codex Distillation Loop')
LAYOUT = ('datasets/master_scenarios', 'datasets/rendered_documents', 'datasets/counterfactuals',
          'datasets/task_examples', 'datasets/approved_sft', 'datasets/replay', 'datasets/quarantine',
          'experiments', 'checkpoints/best', 'checkpoints/candidates', 'checkpoints/archived',
          'evaluation/development', 'evaluation/probes', 'evaluation/release_holdout', 'rules/snapshots',
          'gpu_jobs/packages', 'gpu_jobs/imported_results', 'gpu_jobs/logs', 'state', 'api_usage',
          'manifests', 'logs', 'backups')


class StorageError(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_write(path, content):
    """Unique temporary file, fsync, atomic rename, directory fsync where supported."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.', delete=False) as f:
            temporary = Path(f.name)
            f.write(content if isinstance(content, bytes) else content.encode())
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            try:
                os.fsync(fd)
            except OSError as exc:
                # Some exFAT implementations do not expose directory fsync.
                if exc.errno not in (22, 45, 95):
                    raise
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_write(path, json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def jsonl(rows):
    return ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows)


class Storage:
    def __init__(self, root=None, minimum_free_bytes=64 * 1024 * 1024):
        if os.environ.get('SLM_STORAGE_BACKEND', 'local') != 'local':
            raise StorageError('Only the explicit local storage backend is supported')
        self.root = Path(root or os.environ.get('SLM_ARTIFACT_ROOT', DEFAULT_ROOT)).absolute()
        if not self.root.is_absolute() or self.root.is_symlink():
            raise StorageError('Artifact root must be an existing, non-symlink absolute directory')
        self.minimum_free_bytes = minimum_free_bytes
        self.device = None
        self.preflight()
        self.device = self.root.stat().st_dev

    def preflight(self):
        try:
            if not self.root.is_dir():
                raise StorageError('Artifact root unavailable; no automatic fallback or root creation')
            if self.root == DEFAULT_ROOT and not os.path.ismount('/Volumes/Extreme SSD'):
                raise StorageError('Expected external SSD is not mounted')
            if self.device is not None and self.root.stat().st_dev != self.device:
                raise StorageError('Artifact volume identity changed')
            if not os.access(self.root, os.R_OK | os.W_OK):
                raise StorageError('Artifact root requires read/write access')
            free = shutil.disk_usage(self.root).free
            if free < self.minimum_free_bytes:
                raise StorageError('Insufficient free space')
            return {'root': str(self.root), 'device': self.root.stat().st_dev, 'free_bytes': free}
        except OSError as exc:
            raise StorageError('Artifact SSD I/O unavailable') from exc

    def path(self, name):
        path = self.root / name
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise StorageError('Artifact path escapes root')
        return path

    def initialize(self):
        self.preflight()
        for name in LAYOUT:
            self.path(name).mkdir(parents=True, exist_ok=True)
        identity = self.path('manifests/storage.json')
        if identity.exists():
            saved = read_json(identity)
            if saved['device'] != self.device or saved['root'] != str(self.root):
                raise StorageError('Storage identity changed; explicit migration required')
        else:
            atomic_json(identity, {'root': str(self.root), 'device': self.device, 'version': 1})

    @contextlib.contextmanager
    def lock(self):
        self.preflight()
        with self.path('.slm-writer.lock').open('a') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StorageError('Another artifact writer is active') from exc
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def write(self, name, value):
        self.preflight()
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, value)
        self.preflight()
        return {'path': name, 'sha256': checksum(path)}

    def verify(self, receipt):
        self.preflight()
        if checksum(self.path(receipt['path'])) != receipt['sha256']:
            raise StorageError('Artifact checksum mismatch: ' + receipt['path'])
        return read_json(self.path(receipt['path']))


def atomic_copy(source, destination):
    """Bounded-memory durable copy for multi-gigabyte adapters and packages."""
    destination = Path(destination)
    temporary = None
    try:
        with Path(source).open('rb') as incoming, tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix='.' + destination.name + '.', delete=False) as outgoing:
            temporary = Path(outgoing.name)
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(temporary, destination)
        fd = os.open(destination.parent, os.O_RDONLY)
        try:
            try:
                os.fsync(fd)
            except OSError as exc:
                if exc.errno not in (22, 45, 95):
                    raise
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
