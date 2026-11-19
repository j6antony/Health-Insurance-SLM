"""One GPU, two adapters, frozen development cases; inference only."""
import argparse
import gc
import json
import math
from pathlib import Path
import signal
import time
import traceback

from .paired_evaluation import verify_package, analyze
from .storage import atomic_json, digest, checksum


def preflight(root, tokenizer_path=None):
    job, benchmark = verify_package(root)
    from transformers import AutoTokenizer
    from .tokenizer_preflight import analyze as token_analysis
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or job['model_name'], revision=job['model_revision'], token=False)
    receipt = token_analysis(benchmark['cases'], tokenizer, 4096, check_cpu_config=False)
    if receipt['status'] != 'passed' or any(x['prompt'] + 1536 > 4096 for x in receipt['lengths']):
        raise ValueError('Evaluation context preflight failed')
    atomic_json(Path(root).parent / 'test-c-preflight.json', receipt)
    return receipt


def run(root, *, authorized=False, hourly_rate=None, max_gpu_usd=1.0, max_seconds=3600):
    if not authorized: raise ValueError('Explicit inference authorization required')
    if (not isinstance(hourly_rate, (float, int)) or not math.isfinite(hourly_rate) or hourly_rate <= 0
            or not 0 < max_gpu_usd <= 1 or not 0 < max_seconds <= 3600):
        raise ValueError('Approved positive hourly valuation and bounded budget/runtime required')
    root = Path(root).resolve(); job, benchmark = verify_package(root)
    output = root.parent / 'results'; output.mkdir(exist_ok=True)
    if (output / 'completion.json').exists():
        raise ValueError('Completed evaluation exists; never regenerate saved predictions')
    import torch
    import importlib.metadata
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from distillation_loop.student import StudentRuntime, require_single_gpu, render_prompt, clear_cuda_cache
    from distillation_loop.evaluation import require_training_versions
    from distillation_loop.adapter_identity import adapter_files
    require_training_versions(); require_single_gpu(torch)
    if torch.cuda.get_device_properties(0).total_memory < 22 * 1024**3:
        raise ValueError('Approved 24 GB-class GPU required')
    ledger = output / 'runtime-budget.json'
    spent = json.loads(ledger.read_text()) if ledger.exists() else {'seconds': 0.0, 'cost_usd': 0.0, 'hourly_rate': hourly_rate}
    if spent['hourly_rate'] != hourly_rate: raise ValueError('Cannot change cost valuation during resume')
    remaining = min(max_seconds - spent['seconds'], (max_gpu_usd - spent['cost_usd']) * 3600 / hourly_rate)
    if remaining <= 0: raise ValueError('Evaluation runtime budget exhausted')
    started = time.monotonic()
    def timeout(signum, frame): raise TimeoutError('Paired inference worker budget exhausted')
    previous = signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, remaining)
    info = {'job_id': job['job_id'], 'status': 'started', 'gpu': torch.cuda.get_device_name(0),
            'cuda_runtime': torch.version.cuda, 'versions': {p: importlib.metadata.version(p) for p in
            ('torch', 'transformers', 'trl', 'peft', 'accelerate', 'bitsandbytes', 'datasets', 'lighteval')},
            'arms': {}, 'no_training': True, 'no_promotion': True}
    try:
        set_seed(43)
        tokenizer = AutoTokenizer.from_pretrained(job['model_name'], revision=job['model_revision'], token=False)
        from .tokenizer_preflight import analyze as token_analysis
        token_check = token_analysis(benchmark['cases'], tokenizer, 4096, check_cpu_config=False)
        if token_check['status'] != 'passed' or any(x['prompt'] + 1536 > 4096 for x in token_check['lengths']):
            raise ValueError('Inference prompt/context preflight failed')
        atomic_json(output / 'tokenizer-preflight.json', token_check)
        model = AutoModelForCausalLM.from_pretrained(job['model_name'], revision=job['model_revision'], token=False,
            device_map={'': 0}, dtype=torch.float16, attn_implementation='sdpa', low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
        if not getattr(model, 'is_loaded_in_4bit', False): raise ValueError('Expected 4-bit model')
        runtime = StudentRuntime(model, tokenizer, job['config']); runtime.prepare()
        if runtime.identity()['model_revision'] != job['model_revision']: raise ValueError('Loaded model revision mismatch')
        info['model_identity'] = runtime.identity()
        arms = {}
        for name in ('baseline', 'candidate'):
            checkpoint = root / name
            if adapter_files(checkpoint) != job['adapter_files'][name]: raise ValueError('Adapter changed')
            gc.collect(); clear_cuda_cache(torch)
            runtime.restore(checkpoint)
            torch.cuda.reset_peak_memory_stats(0)
            path = output / (name + '-raw-predictions.json')
            identity = {'benchmark_id': job['benchmark_id'], 'adapter_sha256': digest(adapter_files(checkpoint)),
                        'contract_version': 'document-verification-v1', 'model_name': job['model_name'],
                        'inference_config': job['inference_protocol']}
            saved = json.loads(path.read_text()) if path.exists() else {**identity, 'predictions': {}}
            if any(saved.get(k) != v for k, v in identity.items()): raise ValueError('Saved inference identity mismatch')
            permitted = {c['id'] for c in benchmark['cases']}
            if not set(saved['predictions']) <= permitted: raise ValueError('Unknown saved prediction')
            arm_started = time.monotonic(); generated = 0
            for case in benchmark['cases']:
                if case['id'] in saved['predictions']: continue
                case_started = time.monotonic()
                prompt_tokens = len(tokenizer(render_prompt(tokenizer, case['payload']), add_special_tokens=False)['input_ids'])
                prediction = runtime.infer_many([case['payload']], batch_size=1)[0]
                prediction.update(input_tokens=prompt_tokens, generation_seconds=time.monotonic() - case_started)
                saved['predictions'][case['id']] = prediction
                atomic_json(path, saved)
                generated += 1
                print(f'{name}: {len(saved["predictions"])}/28 saved; task={case["task"]}; tokens={prediction.get("generated_tokens")}; truncated={prediction.get("truncated")}', flush=True)
            if adapter_files(checkpoint) != job['adapter_files'][name]: raise ValueError('Adapter changed during inference')
            arms[name] = {'generations': saved, 'checkpoint': str(checkpoint)}
            info['arms'][name] = {'new_generations_this_attempt': generated, 'wall_seconds_this_attempt': time.monotonic() - arm_started,
                'input_tokens': sum(p['input_tokens'] for p in saved['predictions'].values()),
                'output_tokens': sum(p.get('generated_tokens', 0) for p in saved['predictions'].values()),
                'generation_seconds_sum': sum(p['generation_seconds'] for p in saved['predictions'].values()),
                'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated(0), 'peak_reserved_vram_bytes': torch.cuda.max_memory_reserved(0)}
        verify_package(root)
        summary = analyze(benchmark, arms['baseline'], arms['candidate'], output)
        info.update(status='completed', comparison=summary['comparison'])
        atomic_json(output / 'completion.json', {'job_id': job['job_id'], 'benchmark_id': job['benchmark_id'],
                    'status': 'completed', 'outputs': {p.name: checksum(p) for p in output.glob('*.json') if p.name not in ('runtime-budget.json', 'completion.json', 'runtime-report.json')},
                    'no_promotion': True, 'no_training': True})
        return info
    except BaseException:
        info.update(status='failed', error_traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
        elapsed = time.monotonic() - started
        spent['seconds'] += elapsed; spent['cost_usd'] += elapsed * hourly_rate / 3600
        atomic_json(ledger, spent)
        info.update(worker_seconds_this_attempt=elapsed, cumulative_worker_seconds=spent['seconds'],
                    estimated_credit_value_usd=spent['cost_usd'], max_gpu_usd=max_gpu_usd,
                    cost_limitation='Retail valuation of existing credits; external setup/transfer/idle charges are outside this worker watchdog.')
        atomic_json(output / 'runtime-report.json', info)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--tokenizer')
    p.add_argument('--authorize-inference', action='store_true')
    p.add_argument('--gpu-hourly-rate-usd', type=float)
    p.add_argument('--max-gpu-usd', type=float, default=1)
    p.add_argument('--max-seconds', type=float, default=3600)
    a = p.parse_args()
    result = preflight(a.root, a.tokenizer) if a.validate_only else run(a.root, authorized=a.authorize_inference,
            hourly_rate=a.gpu_hourly_rate_usd, max_gpu_usd=a.max_gpu_usd, max_seconds=a.max_seconds)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
