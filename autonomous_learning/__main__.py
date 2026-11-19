"""Command line interface for the local autonomous learning workflow."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from self_instruct_pipeline.budget import BudgetExceeded
from self_instruct_pipeline.parent_backend import ParentError
from .storage import Storage, StorageError, digest, read_json, checksum
from .scenarios import generate, expand, OPERATIONS
from .rendering import render, verify
from .dataset import build, export
from .tasks import project
from .orchestrator import Controller, State, DEFAULTS
from .planner import validate_plan


def required(args, *names):
    for name in names:
        if getattr(args, name) is None:
            raise ValueError('--' + name.replace('_', '-') + ' is required for ' + args.command)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['storage-check', 'init', 'generate', 'verify', 'expand', 'build',
        'stats', 'dry-run', 'start', 'resume', 'status', 'pause', 'cancel', 'prepare-training',
        'import-results', 'import-benchmark', 'evaluate', 'report', 'compare', 'planner-input',
        'render-api', 'batch-export', 'batch-import'])
    parser.add_argument('--root', type=Path, help='Explicit artifact root override; must already exist')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config/autonomous_learning.json')
    parser.add_argument('--run-id', default='cycle-demo')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--benchmark', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--count', type=int, default=24)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--operation', choices=OPERATIONS, default='remove_document')
    parser.add_argument('--authorize-api', action='store_true')
    parser.add_argument('--authorize-release-evaluation', action='store_true')
    args = parser.parse_args(argv)
    try:
        storage = Storage(args.root)
        if args.command == 'storage-check':
            result = storage.preflight()
            with tempfile.NamedTemporaryFile(dir=storage.root, prefix='.slm-probe-') as f:
                f.write(b'SSD write verification\n')
                f.flush()
                os.fsync(f.fileno())
            result['write_probe'] = 'verified and removed'
        elif args.command == 'init':
            storage.initialize()
            result = storage.preflight()
        else:
            config = read_json(args.config)
            controller = Controller(storage, config)
            state = controller.state
            if args.command in ('start', 'dry-run', 'resume'):
                if args.command == 'resume':
                    saved = state.status(args.run_id)['run']
                    if saved is None:
                        raise ValueError('Unknown run')
                    state.control(args.run_id, 'resume')
                    plan = json.loads(saved['config'])['proposed_plan']
                    dry_run = bool(saved['dry_run'])
                else:
                    plan = read_json(args.plan) if args.plan else None
                    dry_run = args.command == 'dry-run'
                status = controller.run(args.run_id, dry_run=dry_run, proposed_plan=plan, authorize_api=args.authorize_api)
                result = {'run_id': args.run_id, 'status': status['run']['status'],
                          'stages': [{k: row[k] for k in ('cycle', 'name', 'status', 'attempts', 'seconds', 'error')}
                                     for row in status['stages']]}
            elif args.command in ('status', 'report'):
                result = state.status(args.run_id)
                # Raw envelopes can contain documents in stage artifact files,
                # but status/report exposes receipts and aggregate fields only.
                if args.command == 'report':
                    with storage.lock():
                        receipt = storage.write('experiments/' + args.run_id + '/report.json', result)
                    result = {'report': receipt, 'status': result['run']['status'] if result['run'] else 'unknown'}
            elif args.command in ('pause', 'cancel'):
                state.control(args.run_id, args.command)
                result = {'requested': args.command, 'effective': 'between major stages'}
            elif args.command == 'stats':
                required(args, 'input')
                result = read_json(args.input)['statistics']
            elif args.command == 'planner-input':
                from .planner import planner_input
                result = planner_input()
                if args.output:
                    from .storage import atomic_json
                    atomic_json(args.output, result)
            else:
                with storage.lock():
                    if args.command == 'generate':
                        output = generate(args.count, args.seed)
                        result = storage.write('datasets/master_scenarios/' + digest(output) + '.json', output)
                    elif args.command in ('verify', 'expand', 'build'):
                        required(args, 'input')
                        input_data = read_json(args.input)
                        sources = input_data.get('scenarios')
                        if sources is None:
                            sources = [row['scenario'] for row in input_data['verified']]
                        verified, quarantine = [], []
                        for s in sources:
                            if args.command == 'expand':
                                s = expand(s, args.operation)
                            try:
                                r = render(s)
                                gate = verify(s, r)
                            except ValueError as exc:
                                quarantine.append({'scenario': s, 'reason': str(exc)})
                            else:
                                verified.append({'scenario': s, 'rendered': r, 'gate': gate})
                        if args.command == 'build':
                            dataset = build([ex for row in verified for ex in project(row['scenario'], row['rendered'])])
                            result = export(storage, dataset)
                        else:
                            output = {'verified': verified, 'quarantine': quarantine}
                            result = storage.write('datasets/counterfactuals/' + digest(output) + '.json', output)
                    elif args.command == 'prepare-training':
                        required(args, 'input')
                        from .jobs import prepare
                        manifest = read_json(args.input)
                        exported = {'directory': str(args.input.parent.resolve().relative_to(storage.root.resolve())), 'manifest': manifest}
                        plan = read_json(args.plan) if args.plan else __import__('autonomous_learning.planner', fromlist=['heuristic']).heuristic()
                        validate_plan(plan, {'max_api_usd': config['max_api_usd'], 'max_gpu_usd': config['max_gpu_usd']})
                        result = prepare(storage, exported, plan, args.checkpoint)
                    elif args.command == 'import-results':
                        required(args, 'input', 'job')
                        from .jobs import import_result
                        result = import_result(storage, args.input, read_json(args.job))
                    elif args.command == 'import-benchmark':
                        required(args, 'input')
                        from .evaluation import import_benchmark
                        training = []
                        for path in storage.path('datasets/approved_sft').glob('*/examples.jsonl'):
                            training.extend(json.loads(line) for line in path.read_text().splitlines())
                        result = import_benchmark(storage, read_json(args.input), training)
                    elif args.command == 'evaluate':
                        required(args, 'input', 'benchmark', 'checkpoint')
                        from .evaluation import evaluate
                        # A sealed benchmark is read only after access reservation.
                        if not args.benchmark.resolve().is_relative_to(storage.path('evaluation').resolve()):
                            raise ValueError('Import benchmark into evaluation storage first')
                        if args.benchmark.parent.name == 'release_holdout':
                            from distillation_loop.adapter_identity import adapter_files
                            state.reserve_holdout(args.benchmark.stem, digest(adapter_files(args.checkpoint)), args.authorize_release_evaluation)
                        benchmark = read_json(args.benchmark)
                        if digest(benchmark) != args.benchmark.stem:
                            raise ValueError('Benchmark identity changed')
                        result = evaluate(benchmark, read_json(args.input), args.checkpoint, holdout_state=state,
                                          release_authorized=args.authorize_release_evaluation)
                        receipt = storage.write('evaluation/' + benchmark['tier'] + '/reports/' + result['report_hash'] + '.json', result)
                        result = {'report': receipt, 'macro_task_score': result['macro_task_score'], 'release_ready': False}
                    elif args.command == 'compare':
                        required(args, 'input')
                        from .experiments import comparison_plan
                        spec = read_json(args.input)
                        result = comparison_plan(spec['mode'], spec['arms'])
                        storage.write('experiments/comparison-' + result['experiment_id'] + '.json', result)
                    else:
                        from .teacher import RenderQueue
                        queue = RenderQueue(storage)
                        settings = read_json(Path(__file__).resolve().parents[1] / 'config/self_instruct.json')['parent']
                        if args.command == 'batch-export':
                            required(args, 'output')
                            result = queue.export_batch(settings, args.output)
                        elif args.command == 'batch-import':
                            required(args, 'input')
                            queue.import_batch([json.loads(line) for line in args.input.read_text().splitlines()])
                            result = {'status': 'imported; per-request verification retained in queue'}
                        else:
                            required(args, 'input')
                            for s in read_json(args.input)['scenarios']:
                                for d in s['documents']:
                                    if d['type'] in ('discharge_summary', 'investigation_report') and d['present'] and d['readability'] == 'readable':
                                        queue.enqueue(d, settings)
                            result = queue.run(settings, paid_authorized=args.authorize_api,
                                               max_calls=config['max_api_calls'], max_cost_usd=config['max_api_usd'])
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, KeyError, OSError, StorageError, BudgetExceeded, ParentError) as exc:
        # Error details from user-controlled document text are deliberately not echoed.
        print(f'Autonomous learning stopped ({type(exc).__name__}): {str(exc) if isinstance(exc, StorageError) else "inspect command inputs and persisted stage status"}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
