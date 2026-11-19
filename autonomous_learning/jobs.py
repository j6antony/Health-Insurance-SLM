"""Versioned, verified manual GPU transfer packages; no remote SSD assumptions."""
from pathlib import Path
import json
import shutil
import zipfile

from distillation_loop.adapter_identity import adapter_files
from . import CONTRACT_VERSION
from .storage import atomic_json, atomic_write, atomic_copy, checksum, digest, read_json

from .model import MODEL, MODEL_REVISION


def adapter_compatibility(adapter, training):
    cfg = read_json(Path(adapter) / 'adapter_config.json')
    expected = {'base_model_name_or_path': MODEL, 'r': training['lora_rank'],
                'lora_alpha': training['lora_alpha'], 'lora_dropout': training['lora_dropout'],
                'task_type': 'CAUSAL_LM', 'peft_type': 'LORA', 'bias': 'none'}
    problems = [key for key, value in expected.items() if cfg.get(key) != value]
    target = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    if set(cfg.get('target_modules', [])) != target:
        problems.append('target_modules')
    if cfg.get('use_rslora', False) != training.get('use_rslora', False):
        problems.append('use_rslora')
    if cfg.get('use_dora', False) or cfg.get('use_qalora', False) or cfg.get('rank_pattern') or cfg.get('alpha_pattern') or cfg.get('modules_to_save'):
        problems.append('unsupported_lora_variant')
    if cfg.get('revision') not in (None, MODEL_REVISION):
        problems.append('base_revision')
    return {'compatible': not problems, 'mismatches': problems, 'revision_limit':
            'An absent saved base revision cannot prove historical training used the pinned base; record this lineage limitation.'}


def prepare(storage, exported, plan, adapter=None, training_overrides=None, inference_overrides=None):
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
    if training_overrides:
        allowed = {'max_steps_per_round', 'learning_rate', 'gradient_accumulation_steps', 'per_device_train_batch_size',
                   'lora_rank', 'lora_alpha', 'lora_dropout', 'use_rslora'}
        if set(training_overrides) - allowed:
            raise ValueError('Unsupported training canary override')
        for key, value in training_overrides.items():
            if key == 'use_rslora':
                if type(value) is not bool:
                    raise ValueError('use_rslora must be boolean')
            elif key == 'lora_dropout':
                if type(value) not in (int, float) or not 0 <= value < 1:
                    raise ValueError('Invalid LoRA dropout')
            elif key == 'learning_rate':
                if type(value) not in (int, float) or not 0 < value <= .001:
                    raise ValueError('Invalid canary learning rate')
            elif type(value) is not int or value < 1:
                raise ValueError('Invalid training step/batch override')
        config['training'].update(training_overrides)
    if inference_overrides:
        if set(inference_overrides) != {'max_new_tokens'} or type(inference_overrides['max_new_tokens']) is not int or not 1 <= inference_overrides['max_new_tokens'] <= 2048:
            raise ValueError('Invalid canary inference budget')
        config['inference'].update(inference_overrides)
    config['random_seed'] = 42 + plan['cycle_id']
    binding = adapter_files(adapter) if adapter else None
    if adapter:
        compatibility = adapter_compatibility(adapter, config['training'])
        if not compatibility['compatible']:
            raise ValueError('Incompatible experimental adapter: ' + ','.join(compatibility['mismatches']))
    job = {'version': 1, 'contract_version': CONTRACT_VERSION, 'model_name': MODEL,
           'model_revision': MODEL_REVISION, 'adapter_strategy': plan['adapter_strategy'],
           'adapter_compatibility': compatibility if adapter else None,
           'adapter_files': binding, 'adapter_lineage': digest(binding) if binding else None,
           'dataset_id': manifest['dataset_id'], 'dataset_files': manifest['files'],
           'config': config, 'seed': config['random_seed'], 'planned_max_gpu_usd': plan['max_gpu_usd'], 'output_directory': 'result',
           'requirements_sha256': checksum(source / 'requirements-training.txt'),
           'source_sha256': {str(p.relative_to(source)): checksum(p)
                             for package in ('autonomous_learning', 'distillation_loop', 'self_instruct_pipeline')
                             for p in sorted((source / package).glob('*.py'))},
           'authorization': 'prepared only; remote --authorize-training required',
           'limitations': ['CUDA model loading and training require live canary approval.']}
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
