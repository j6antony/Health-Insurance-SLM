"""Run the single-GPU 1,000-pair distillation profile outside a notebook."""
import argparse
import json
import os
from datetime import datetime, timezone
from time import monotonic
from pathlib import Path

from .colab import UnslothStudentRuntime, distillation_1000_configs, save_colab_subset
from .controller import run_loop
from .core import load_loop_config
from .evaluation import require_training_versions
from .github_backup import restore_latest_release
from .preflight import readiness
from .student import require_single_gpu
from self_instruct_pipeline.budget import APIBudget, BudgetExceeded
from self_instruct_pipeline.config import ROOT, load_config
from self_instruct_pipeline.persistence import atomic_json



def status(message):
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f'[{timestamp}] {message}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--api-max-cost-usd', required=True, help='Cumulative teacher API spending cap')
    parser.add_argument('--api-max-calls', type=int, default=10000)
    parser.add_argument('--inference-batch-size', type=int, default=4)
    args = parser.parse_args()

    parent, config = distillation_1000_configs(load_config(), load_loop_config(), args.run_dir.resolve())
    model_name = parent['student']['model']
    key = parent['parent']['api_key_env']
    if not os.environ.get(key, '').strip():
        parser.error(f'Set {key} in the shell before running')
    if args.inference_batch_size < 1:
        parser.error('--inference-batch-size must be positive')

    started = monotonic()
    status(f'Run directory: {config["output_dir"]}')
    status('Loading CUDA and Unsloth')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    # Unsloth must patch Transformers before the latter is imported.
    import unsloth
    import torch

    require_training_versions()
    require_single_gpu(torch)
    memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    status(f'GPU: {torch.cuda.get_device_name(0)} ({memory_gib:.1f} GiB)')
    status(f'Student: {model_name} | inference batch: {args.inference_batch_size}')

    run_dir = args.run_dir.resolve()
    seed = restore_latest_release(run_dir, model_name)
    if seed:
        config['training'].update(seed['training'])
        status(f'Pinned GitHub adapter: {seed["tag"]}')
    if (run_dir / 'loop_state.json').exists():
        status('Resuming saved local run')
    elif not seed:
        status('No compatible GitHub release found. Starting from the base model.')
    config['inference']['batch_size'] = args.inference_batch_size
    profile = run_dir / 'requested_loop_config.json'
    if (run_dir / 'loop_state.json').exists() and profile.exists():
        if json.loads(profile.read_text()) != config:
            raise ValueError('Resume with the same run path and inference batch size')
    atomic_json(profile, config)
    config = load_loop_config(profile)
    report = readiness(parent, config, check_runtime=False)
    if not report['ready']:
        raise ValueError(f'Resolve data/configuration issues before training: {report}')
    schemas = json.loads((ROOT / parent['paths']['schemas']).read_text())['tasks']
    subset = save_colab_subset(config, schemas, run_dir)
    status(f'{subset["total"]} development cases | {config["max_rounds"]} rounds | '
           f'{config["scenarios_per_round"]} new pairs per round ({config["adaptive_pairs_per_round"]} adaptive)')

    budget = APIBudget(run_dir / 'api_budget', max_calls=args.api_max_calls,
                       max_cost_usd=args.api_max_cost_usd)
    try:
        with budget.activate(allow_increase=True):
            status(f'Teacher budget: ${budget.state["accounted_usd"]} used of '
                   f'${budget.limits["max_cost_usd"]} estimated USD | '
                   f'{budget.state["calls"]}/{budget.limits["max_calls"]} calls')
            status('Loading student weights (first run may download the model)')
            runtime = UnslothStudentRuntime.load(model_name, config)
            result = run_loop(runtime, parent, config, initial_adapter=seed['checkpoint'] if seed else None)
    except BudgetExceeded as exc:
        status(f'Paused at teacher budget: {exc}')
        status('Resume with the same run directory and a higher cumulative budget cap.')
        return 2
    status(f'Run finished in {(monotonic() - started) / 60:.1f} min | '
           f'{result["completed_rounds"]} rounds | {result["status"]}')
    status(f'Best development macro score: {result["best_report"]["macro_task_score"]:.3f}')
    status(f'Selected checkpoint: {result["best_checkpoint"]}')
    status(f'Teacher API: {budget.state["calls"]} calls, ${budget.state["accounted_usd"]} estimated USD')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
