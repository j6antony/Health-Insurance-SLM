"""Observability and authorization wrapper for the immutable Strategy A package.

This file stays OUTSIDE the checksummed package. It never changes training data,
model settings, or the packaged trainer. Free resources use the same worker core
with a wall-clock watchdog because the old paid CLI rejects a zero hourly rate.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import zipfile

JOB = 'c03c14696b8ab4b394266f83d850641ae3e71f9eaa3cc749a34de12489834fc5'
ARCHIVE_SHA = '19ec9e08f6bd85eb7ee585dbb62939db7c3fbcaa588f70f853b89b82a5751e2d'
REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
DATASET = '51cb0864daa7680acd3afd5d68d31276bac2289cc24bf7921cbc850e81252810'
ADAPTER = {
    'adapter_config.json': '1fe1fdbfa6b611c63f9d7dc84d431f2b3685338d91678510f579b24ae002d5df',
    'adapter_model.safetensors': '154c508775106e8ffe15e224c9f256131852d384e4efaea6af2bdf3d35e99c6f',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def extract(archive, root):
    if sha(archive) != ARCHIVE_SHA:
        raise ValueError('Uploaded archive SHA-256 mismatch')
    if root.exists():
        raise ValueError('Extraction directory already exists; preserve previous attempts')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        for entry in z.infolist():
            target = root / entry.filename
            mode = entry.external_attr >> 16
            if not target.resolve().is_relative_to(root.resolve()) or mode & 0o170000 == 0o120000:
                raise ValueError('Unsafe ZIP member')
        z.extractall(root)
    return verify(root)


def verify(root):
    sys.path.insert(0, str(root))
    from autonomous_learning.jobs import verify_package, adapter_compatibility
    from autonomous_learning.dataset import validate_example
    job = verify_package(root)
    if (job['job_id'] != JOB or job['dataset_id'] != DATASET
            or job['model_name'] != 'Qwen/Qwen3-8B' or job['model_revision'] != REVISION
            or job['adapter_strategy'] != 'continued' or job['adapter_files'] != ADAPTER
            or job['output_directory'] != 'result' or job['planned_max_gpu_usd'] != 1):
        raise ValueError('Not the approved immutable Strategy A job')
    expected = dict(lora_rank=32, lora_alpha=32, lora_dropout=0.0, use_rslora=True,
                    learning_rate=0.00002, per_device_train_batch_size=1,
                    gradient_accumulation_steps=4, max_steps_per_round=2,
                    max_sequence_length=4096)
    if any(job['config']['training'].get(k) != v for k, v in expected.items()):
        raise ValueError('Strategy A training settings changed')
    if not adapter_compatibility(root / 'initial_adapter', job['config']['training'])['compatible']:
        raise ValueError('Starting adapter is incompatible')
    rows = [json.loads(x) for x in (root / 'examples.jsonl').read_text().splitlines()]
    if len(rows) != 85 or sum(x['replay'] is True for x in rows) != 21:
        raise ValueError('Expected 64 new examples and 21 replay examples')
    for row in rows:
        validate_example(row)
    return job


def hardware():
    import importlib.metadata
    import platform
    import subprocess
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Exactly one visible CUDA GPU is required')
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    # Accommodate reported capacity of marketed 24 GB devices, never 16 GB cards.
    if total < 22 * 1024**3 or free < 20 * 1024**3 or not 7 <= props.major <= 9:
        raise ValueError(f'Need a 24 GB-class GPU with >=20 GiB free and CC 7.x–9.x for pinned CUDA 12.6: {props.name}; free={free}, total={total}')
    versions = {p: importlib.metadata.version(p) for p in
                ('torch', 'transformers', 'trl', 'peft', 'accelerate', 'bitsandbytes', 'datasets', 'lighteval', 'safetensors')}
    return {'gpu': props.name, 'compute_capability': [props.major, props.minor],
            'total_vram_bytes': total, 'free_vram_bytes': free, 'cuda_runtime': torch.version.cuda,
            'python': platform.python_version(), 'platform': platform.platform(),
            'versions': versions, 'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True)}


def preflight(root):
    import subprocess
    job = verify(root)
    environment = hardware()
    from distillation_loop.evaluation import require_training_versions
    require_training_versions()
    # A separate process keeps validate-only's CPU Accelerate state out of training.
    completed = subprocess.run([sys.executable, '-m', 'autonomous_learning.gpu_worker', '.', '--validate-only'],
                               cwd=root, text=True, capture_output=True)
    (root.parent / 'validate-only.stdout.log').write_text(completed.stdout)
    (root.parent / 'validate-only.stderr.log').write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError('Worker validate-only failed; inspect validate-only.stderr.log')
    analysis = json.loads(completed.stdout)
    if analysis['status'] != 'tokenizer_validated_only' or analysis['examples'] != 85 or analysis['overlength_examples']:
        raise ValueError('Tokenizer preflight failed')
    receipt = {'status': 'passed', 'job_id': job['job_id'], 'environment': environment, 'tokenizer': analysis}
    save(root.parent / 'cloud-preflight.json', receipt)
    return receipt


def authorization(args, environment):
    if args.billing_mode not in ('free', 'paid'):
        raise ValueError('Confirm whether this specific allocation is free or paid')
    if not args.authorize_training or not args.provider or args.expected_gpu != environment['gpu']:
        raise ValueError('Explicit training authorization, provider and exact detected GPU name are required')
    if not math.isfinite(args.max_gpu_usd) or not 0 < args.max_gpu_usd <= 1:
        raise ValueError('Budget must be positive and at most the prepared $1 limit')
    if not math.isfinite(args.max_runtime_seconds) or not 0 < args.max_runtime_seconds <= 1200:
        raise ValueError('Choose a worker wall-clock limit of at most 1200 seconds')
    rate = args.gpu_hourly_rate_usd
    if not math.isfinite(rate) or rate < 0:
        raise ValueError('Hourly rate must be finite and nonnegative')
    if args.billing_mode == 'free' and rate != 0:
        raise ValueError('Confirmed free resources require actual hourly charge 0')
    if args.billing_mode == 'paid' and rate <= 0:
        raise ValueError('Paid mode requires the actual approved positive hourly rate')


def train(root, args):
    job = verify(root)
    receipt = json.loads((root.parent / 'cloud-preflight.json').read_text())
    if receipt['status'] != 'passed' or receipt['job_id'] != JOB:
        raise ValueError('Run cloud preflight first')
    environment = hardware()
    authorization(args, environment)
    if (root / 'result').exists() or (root.parent / 'training-observations.json').exists():
        raise ValueError('Previous attempt exists; never retrain automatically')
    import torch
    import trl
    from autonomous_learning import gpu_worker
    from autonomous_learning.storage import atomic_json
    from distillation_loop.student import StudentRuntime
    from distillation_loop.adapter_identity import adapter_files
    observations = {'status': 'started', 'provider': args.provider, 'billing_mode': args.billing_mode,
                    'environment': environment, 'job_id': JOB, 'dataset_id': DATASET,
                    'approved_max_gpu_usd': args.max_gpu_usd, 'approved_gpu_hourly_rate_usd': args.gpu_hourly_rate_usd,
                    'max_runtime_seconds': args.max_runtime_seconds, 'adapter_restores': [],
                    'lineage_limitation': 'The recovered adapter has revision=null; this continuation uses the pinned job revision.'}
    path = root.parent / 'training-observations.json'
    save(path, observations)
    original_train, original_prepare = trl.SFTTrainer.train, StudentRuntime.prepare
    original_restore, original_infer = StudentRuntime.restore, StudentRuntime.infer

    def observe_prepare(runtime, *a, **kw):
        value = original_prepare(runtime, *a, **kw)
        loaded = bool(getattr(runtime.model, 'is_loaded_in_4bit', False))
        observations['four_bit_model_loaded'] = loaded
        if not loaded:
            raise ValueError('Expected a 4-bit QLoRA base model')
        identity = runtime.identity()
        observations['model_identity'] = identity
        if identity['model_revision'] != REVISION:
            raise ValueError('Loaded model revision differs from pinned job')
        return value

    def observe_restore(runtime, checkpoint):
        value = original_restore(runtime, checkpoint)
        observations['adapter_restores'].append({'path': str(checkpoint), 'files': adapter_files(checkpoint)})
        return value

    def observe_train(trainer, *a, **kw):
        value = original_train(trainer, *a, **kw)
        observations['optimizer_steps'] = trainer.state.global_step
        observations['training_log_history'] = trainer.state.log_history
        observations['training_loss'] = float(value.training_loss)
        losses = [x['loss'] for x in trainer.state.log_history if 'loss' in x]
        if trainer.state.global_step != 2 or len(losses) != 2 or not all(math.isfinite(x) for x in [value.training_loss, *losses]):
            raise ValueError('Expected exactly two optimizer steps with two finite logged losses')
        return value

    def observe_infer(runtime, payload):
        value = original_infer(runtime, payload)
        observations['inference'] = {'payload': payload, 'prediction': value}
        return value

    StudentRuntime.prepare, StudentRuntime.restore, StudentRuntime.infer = observe_prepare, observe_restore, observe_infer
    trl.SFTTrainer.train = observe_train
    started = time.monotonic()
    previous = None
    original_setitimer = signal.setitimer
    torch.cuda.reset_peak_memory_stats(0)
    try:
        if args.billing_mode == 'paid':
            # Preserve the existing worker's own cumulative runtime-cost watchdog.
            def bounded_timer(which, seconds, interval=0):
                return original_setitimer(which, min(seconds, args.max_runtime_seconds) if seconds > 0 else 0, interval)
            signal.setitimer = bounded_timer
            result = gpu_worker.run(root, authorized=True, max_gpu_usd=args.max_gpu_usd,
                                    gpu_hourly_rate_usd=args.gpu_hourly_rate_usd)
        else:
            def timeout(signum, frame):
                raise TimeoutError('Free GPU canary wall-clock limit exhausted')
            previous = signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL, args.max_runtime_seconds)
            result = gpu_worker._run(root, authorized=True)
            elapsed = time.monotonic() - started
            if elapsed > args.max_runtime_seconds:
                raise TimeoutError('Free GPU canary exceeded wall-clock limit')
            result.update(gpu_runtime_seconds=elapsed, estimated_gpu_cost_usd=0.0,
                          gpu_budget_usd=args.max_gpu_usd, gpu_hourly_rate_usd=0.0,
                          gpu_cost_limitation='Operator confirmed free allocation; actual usage charge is zero. No equivalent commercial GPU cost was invented.')
        saved = root / job['output_directory']
        files = adapter_files(saved / 'adapter')
        if files != result['adapter_files'] or any((saved / 'adapter' / name).stat().st_size == 0 for name in files):
            raise ValueError('Saved adapter is empty or checksum verification failed')
        if len(observations['adapter_restores']) != 2 or observations['adapter_restores'][0]['files'] != ADAPTER or observations['adapter_restores'][1]['files'] != files:
            raise ValueError('Starting/saved adapter reload evidence is incomplete')
        verify(root)  # Initial adapter, dataset and packaged source remain unchanged.
        observations.update(status='completed', infrastructure_canary_passed=True)
        result['canary_observations'] = observations
        atomic_json(saved / 'result.json', result)
        return result
    except BaseException:
        observations.update(status='failed', infrastructure_canary_passed=False, failure_traceback=traceback.format_exc())
        # Never leave an importable success receipt after an observation failure.
        success = root / 'result/result.json'
        if success.exists():
            success.rename(root / 'result/unverified-worker-result.json')
        raise
    finally:
        signal.setitimer = original_setitimer
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        observations['total_wall_seconds'] = time.monotonic() - started
        try:
            observations.update(peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(0),
                                peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(0))
        except Exception as exc:
            observations['memory_measurement_error'] = str(exc)
        save(path, observations)
        if observations['status'] == 'completed':
            save(root / 'result/canary-observations.json', observations)
            result['canary_observations'] = observations
            atomic_json(root / 'result/result.json', result)
        trl.SFTTrainer.train = original_train
        StudentRuntime.prepare, StudentRuntime.restore, StudentRuntime.infer = original_prepare, original_restore, original_infer


def export(root):
    destination = root.parent / 'test-b-return.zip'
    if destination.exists():
        raise ValueError('Return archive already exists; preserve the earlier download')
    files = [p for p in (root / 'result').rglob('*') if p.is_file()]
    files += [p for p in root.parent.iterdir() if p.is_file() and p.suffix in ('.json', '.log')]
    if not files:
        raise ValueError('No results or diagnostic logs to export')
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for p in files:
            name = 'result/' + str(p.relative_to(root / 'result')) if p.is_relative_to(root / 'result') else 'diagnostics/' + p.name
            z.write(p, name)
    checksum = sha(destination)
    destination.with_suffix('.zip.sha256').write_text(checksum + '  ' + destination.name + '\n')
    return {'archive': str(destination), 'sha256': checksum,
            'import_only_if': 'result/result.json exists and training-observations.json status is completed'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('extract', 'verify', 'preflight', 'train', 'export'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--archive', type=Path)
    p.add_argument('--authorize-training', action='store_true')
    p.add_argument('--provider')
    p.add_argument('--expected-gpu')
    p.add_argument('--billing-mode', choices=('free', 'paid'))
    p.add_argument('--max-gpu-usd', type=float, default=1.0)
    p.add_argument('--gpu-hourly-rate-usd', type=float, default=float('nan'))
    p.add_argument('--max-runtime-seconds', type=float, default=1200)
    a = p.parse_args()
    root = a.root.resolve()
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    if a.phase == 'extract':
        value = extract(a.archive, root)
    elif a.phase == 'verify':
        value = verify(root)
    elif a.phase == 'preflight':
        value = preflight(root)
    elif a.phase == 'train':
        value = train(root, a)
    else:
        value = export(root)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
