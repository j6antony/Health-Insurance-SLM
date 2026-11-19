"""Run an extracted manual training package on one CUDA GPU.

python -m autonomous_learning.gpu_worker PACKAGE_DIR --authorize-training
"""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from .jobs import verify_package
from .storage import atomic_json, digest


def _run(directory, authorized=False, validate_only=False):
    if not authorized and not validate_only:
        raise ValueError('Training requires explicit --authorize-training')
    root = Path(directory).resolve()
    job = verify_package(root)
    output = root / job['output_directory']
    if output.exists():
        raise ValueError('Output already exists; never overwrite a previous training attempt')
    started = time.monotonic()
    if validate_only:
        from transformers import AutoTokenizer
        from distillation_loop.student import tokenize_teacher_pair
        tokenizer = AutoTokenizer.from_pretrained(job['model_name'], revision=job['model_revision'], token=False)
        rows = [json.loads(line) for line in (root / 'examples.jsonl').read_text().splitlines()]
        samples = [tokenize_teacher_pair(tokenizer, r['payload'], r['target'], job['config']['training']['max_sequence_length']) for r in rows]
        return {'status': 'tokenizer_validated_only', 'job_id': job['job_id'], 'examples': len(samples),
                'total_tokens_one_pass': sum(len(s['input_ids']) for s in samples),
                'completion_tokens_one_pass': sum(sum(s['completion_mask']) for s in samples),
                'max_sequence_tokens': max(len(s['input_ids']) for s in samples)}
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from distillation_loop.evaluation import require_training_versions
    from distillation_loop.student import StudentRuntime, tokenize_teacher_pair, train_tokenized
    from distillation_loop.adapter_identity import adapter_files
    require_training_versions()
    set_seed(job['seed'])
    tokenizer = AutoTokenizer.from_pretrained(job['model_name'], revision=job['model_revision'], token=False)
    model = AutoModelForCausalLM.from_pretrained(job['model_name'], revision=job['model_revision'], token=False,
        device_map={'': 0}, dtype=torch.float16, attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16), low_cpu_mem_usage=True)
    runtime = StudentRuntime(model, tokenizer, job['config'])
    runtime.prepare()
    if job['adapter_strategy'] == 'continued':
        runtime.restore(root / 'initial_adapter')
    rows = [json.loads(line) for line in (root / 'examples.jsonl').read_text().splitlines()]
    from .dataset import validate_example
    for row in rows:
        validate_example(row)
    samples = [tokenize_teacher_pair(tokenizer, r['payload'], r['target'], job['config']['training']['max_sequence_length']) for r in rows]
    checkpoint = train_tokenized(runtime, samples, output)
    # Reload the saved weights into the compatible runtime and actually infer.
    runtime.restore(checkpoint)
    prediction = runtime.infer(rows[0]['payload'])
    if not prediction['raw'].strip() or prediction.get('error'):
        raise ValueError('Minimal inference failed; adapter not completed')
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'transformers', 'peft', 'trl', 'datasets', 'bitsandbytes', 'lighteval')}
    result = {'job_id': job['job_id'], 'dataset_id': job['dataset_id'], 'adapter_files': adapter_files(checkpoint),
              'adapter_lineage': job['adapter_lineage'], 'adapter_reload_verified': True,
              'minimal_inference_verified': True, 'library_versions': versions, 'model_identity': runtime.identity(),
              'gpu_runtime_seconds': time.monotonic() - started, 'estimated_gpu_cost_usd': None,
              'training_metrics': json.loads((output / 'training_metrics.json').read_text()),
              'synthetic_smoke_inference': 'Nonempty generation only; not an evaluation score'}
    return result


def run(directory, authorized=False, validate_only=False, max_gpu_usd=None, gpu_hourly_rate_usd=None):
    if validate_only:
        return _run(directory, authorized, True)
    if not authorized:
        raise ValueError('Training requires explicit --authorize-training')
    import math
    import signal
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
           for v in (max_gpu_usd, gpu_hourly_rate_usd)):
        raise ValueError('Training requires positive --max-gpu-usd and --gpu-hourly-rate-usd')
    job = verify_package(directory)
    if max_gpu_usd > job['planned_max_gpu_usd']:
        raise ValueError('Requested GPU budget exceeds the prepared job limit; prepare a new job')
    seconds = 3600 * max_gpu_usd / gpu_hourly_rate_usd
    def timeout(signum, frame):
        raise TimeoutError('GPU worker runtime budget exhausted; result is not complete')
    previous = signal.signal(signal.SIGALRM, timeout)
    started = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        result = _run(directory, authorized, False)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    elapsed = time.monotonic() - started
    if elapsed > seconds:
        raise TimeoutError('GPU worker exceeded its runtime budget')
    result.update(gpu_runtime_seconds=elapsed, estimated_gpu_cost_usd=elapsed * gpu_hourly_rate_usd / 3600,
                  gpu_budget_usd=max_gpu_usd, gpu_hourly_rate_usd=gpu_hourly_rate_usd,
                  gpu_cost_limitation='Estimate for this worker only; external VM idle time and provider billing remain operator managed.')
    job = verify_package(directory)
    atomic_json(Path(directory) / job['output_directory'] / 'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--authorize-training', action='store_true')
    p.add_argument('--validate-only', action='store_true', help='Load only the tokenizer and validate sequence lengths; no GPU training')
    p.add_argument('--max-gpu-usd', type=float)
    p.add_argument('--gpu-hourly-rate-usd', type=float)
    a = p.parse_args()
    print(json.dumps(run(a.directory, a.authorize_training, a.validate_only, a.max_gpu_usd, a.gpu_hourly_rate_usd), indent=2))


if __name__ == '__main__':
    main()
