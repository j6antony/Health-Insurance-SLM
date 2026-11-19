"""Versioned, verified manual GPU transfer packages; no remote SSD assumptions."""
from pathlib import Path
import json
import shutil
import zipfile

from distillation_loop.adapter_identity import adapter_files
from . import CONTRACT_VERSION
from .storage import atomic_json, atomic_write, atomic_copy, checksum, digest, read_json

MODEL = 'Qwen/Qwen3-8B'


def prepare(storage, exported, plan, adapter=None):
    storage.preflight()
    directory = storage.path(exported['directory'])
    manifest = exported['manifest']
    for name, expected in manifest['files'].items():
        if checksum(directory / name) != expected:
            raise ValueError('Training dataset checksum mismatch')
    if plan['adapter_strategy'] == 'continued' and adapter is None:
        raise ValueError('Continued training requires a verified compatible adapter')
    if plan['adapter_strategy'] == 'fresh' and adapter is not None:
        raise ValueError('Fresh adapter must not load previous LoRA weights')
    source = Path(__file__).resolve().parents[1]
    config = read_json(source / 'config/distillation_loop.json')
    config['random_seed'] = 42 + plan['cycle_id']
    binding = adapter_files(adapter) if adapter else None
    if adapter:
        cfg = read_json(Path(adapter) / 'adapter_config.json')
        if cfg.get('base_model_name_or_path') != MODEL or cfg.get('r') != config['training']['lora_rank']:
            raise ValueError('Incompatible experimental adapter')
    job = {'version': 1, 'contract_version': CONTRACT_VERSION, 'model_name': MODEL,
           'model_revision': 'main', 'adapter_strategy': plan['adapter_strategy'],
           'adapter_files': binding, 'adapter_lineage': digest(binding) if binding else None,
           'dataset_id': manifest['dataset_id'], 'dataset_files': manifest['files'],
           'config': config, 'seed': config['random_seed'], 'planned_max_gpu_usd': plan['max_gpu_usd'], 'output_directory': 'result',
           'requirements_sha256': checksum(source / 'requirements-training.txt'),
           'source_sha256': {str(p.relative_to(source)): checksum(p)
                             for package in ('autonomous_learning', 'distillation_loop', 'self_instruct_pipeline')
                             for p in sorted((source / package).glob('*.py'))},
           'authorization': 'prepared only; remote --authorize-training required',
           'limitations': ['Model revision main is resolved and recorded at remote load time; pin a revision before a release experiment.']}
    job['job_id'] = digest(job)
    base = f'gpu_jobs/packages/{job["job_id"]}'
    root = storage.path(base)
    root.mkdir(exist_ok=True)
    contents = {'job.json': json.dumps(job, sort_keys=True, indent=2).encode(),
                'requirements-training.txt': source / 'requirements-training.txt'}
    for name in manifest['files']:
        contents[name] = directory / name
    for name in job['source_sha256']:
        contents[name] = source / name
    if adapter:
        for name in binding:
            contents['initial_adapter/' + name] = Path(adapter) / name
    package = root / 'package.zip'
    temporary = root / '.package.zip.tmp'
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for name, data in sorted(contents.items()):
                entry = zipfile.ZipInfo(name)
                if isinstance(data, Path):
                    with data.open('rb') as incoming, z.open(entry, 'w', force_zip64=True) as outgoing:
                        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                else:
                    z.writestr(entry, data)
        atomic_copy(temporary, package)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {'job': job, 'package': base + '/package.zip', 'sha256': checksum(package)}
    storage.write(base + '/manifest.json', receipt)
    return receipt


def verify_package(directory):
    root = Path(directory)
    job = read_json(root / 'job.json')
    if job['job_id'] != digest({k: v for k, v in job.items() if k != 'job_id'}):
        raise ValueError('GPU job identity mismatch')
    if checksum(root / 'requirements-training.txt') != job['requirements_sha256']:
        raise ValueError('GPU requirements changed')
    for files, prefix in ((job['dataset_files'], ''), (job['source_sha256'], ''), (job['adapter_files'] or {}, 'initial_adapter/')):
        for name, expected in files.items():
            path = root / prefix / name
            if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or checksum(path) != expected:
                raise ValueError('GPU package checksum mismatch')
    return job


def import_result(storage, directory, job_receipt):
    """Import an unpacked result only, after remote adapter reload/inference checks."""
    storage.preflight()
    source = Path(directory)
    result = read_json(source / 'result.json')
    job = job_receipt['job']
    if result['job_id'] != job['job_id'] or result['dataset_id'] != job['dataset_id']:
        raise ValueError('GPU result belongs to a different job/dataset')
    if (result.get('adapter_reload_verified') is not True or result.get('minimal_inference_verified') is not True
            or not isinstance(result.get('library_versions'), dict) or not result['library_versions']):
        raise ValueError('GPU result lacks successful remote verification')
    import math
    cost = result.get('estimated_gpu_cost_usd')
    if (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0
            or cost > job['planned_max_gpu_usd']):
        raise ValueError('GPU result lacks bounded cost accounting or exceeded its job budget')
    files = adapter_files(source / 'adapter')
    if files != result['adapter_files']:
        raise ValueError('GPU adapter checksum mismatch')
    destination = storage.path('gpu_jobs/imported_results/' + job['job_id'])
    if destination.exists():
        if read_json(destination / 'result.json') != result or adapter_files(destination / 'adapter') != files:
            raise ValueError('Immutable imported result changed')
        return {'directory': str(destination), 'result': result}
    temporary = destination.with_name('.' + destination.name + '.importing')
    if temporary.exists():
        raise ValueError('Partial import exists; inspect before manually retrying')
    temporary.mkdir()
    (temporary / 'adapter').mkdir()
    for name in files:
        atomic_copy(source / 'adapter' / name, temporary / 'adapter' / name)
    if adapter_files(temporary / 'adapter') != files:
        raise ValueError('Imported adapter failed copy verification')
    atomic_json(temporary / 'result.json', result)
    storage.preflight()
    temporary.rename(destination)
    return {'directory': str(destination), 'result': result}
