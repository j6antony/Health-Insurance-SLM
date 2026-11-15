"""Compare the initial, SFT and SFT+ORPO checkpoints from a saved Colab run."""
import argparse
import json
import os
import statistics
from collections import Counter
from pathlib import Path

from self_instruct_pipeline.config import ROOT
from self_instruct_pipeline.persistence import Store, atomic_json, atomic_jsonl, digest, read_jsonl
from self_instruct_pipeline.prompt_builder import target_payload

from .core import aggregate_scores, score_response
from .github_backup import file_sha256

STAGES = {
    'initial': 'initial_adapter',
    'sft': 'round_01/sft/adapter',
    'sft_orpo': 'round_01/preference/adapter',
}
MIN_REPEAT_WIDTH = 8
MAX_REPEAT_WIDTH = 64
REPEATS = 3
MAX_NEW_TOKENS = 1024


class RepetitionStopping:
    """Stop a row after three identical consecutive blocks of completion tokens."""

    def __init__(self, prompt_width, budget, tokenizer):
        self.prompt_width = prompt_width
        self.budget = budget
        self.eos = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id
        self.lengths = None

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        length = input_ids.shape[1] - self.prompt_width
        last = input_ids[:, -1]
        if self.lengths is None:
            self.lengths = torch.zeros_like(last)
            self.finished = torch.zeros_like(last, dtype=torch.bool)
            self.widths = torch.arange(MIN_REPEAT_WIDTH, MAX_REPEAT_WIDTH + 1, device=last.device)[:, None]
            self.offsets = torch.arange((REPEATS - 1) * MAX_REPEAT_WIDTH, device=last.device)
        self.finished |= (last == self.eos) | (last == self.pad)
        if REPEATS * MIN_REPEAT_WIDTH <= length < self.budget:
            # A suffix repeats if each token equals the token one block earlier.
            # Compare all candidate block widths on-device without syncing each row.
            end = input_ids.shape[1] - 1
            current = input_ids[:, (end - self.offsets).clamp_min(self.prompt_width)]
            earlier = input_ids[:, (end - self.offsets - self.widths).clamp_min(self.prompt_width)]
            ignored = self.offsets >= (REPEATS - 1) * self.widths
            matches = ((current[:, None, :] == earlier) | ignored).all(dim=2)
            repeated = (matches & (REPEATS * self.widths[:, 0] <= length)).any(dim=1)
            self.lengths.masked_fill_(repeated & ~self.finished & (self.lengths == 0), length)
        return self.lengths != 0

    def stopped_at(self):
        return self.lengths.tolist()


def frozen_cases(run_dir):
    historical = run_dir / 'round_01/lighteval'
    scores = json.loads((historical / 'scores.json').read_text())
    index = {item['record']['id']: item for item in read_jsonl(ROOT / 'data/evaluation/fixed_validation_v1.jsonl')}
    ids = [row['scenario_id'] for row in read_jsonl(historical / 'predictions.jsonl')]
    if len(ids) != 360 or len(set(ids)) != 360 or set(ids) - index.keys():
        raise ValueError('Expected all 360 unique Round 1 benchmark cases')
    cases = [index[case_id] for case_id in ids]
    if digest(cases) != scores['benchmark_hash']:
        raise ValueError('The benchmark hash differs from the saved Round 1 evaluation')
    return cases


def saved_predictions(folder, cases):
    scores = json.loads((folder / 'scores.json').read_text())
    generation = json.loads((folder / 'generation.json').read_text())
    rows = read_jsonl(folder / 'predictions.jsonl')
    predictions = generation['predictions']
    if (scores['benchmark_hash'] != digest(cases) or generation['key'] != scores['generation_key']
            or len(rows) != len(cases) or len(predictions) != len(cases)
            or any(row['scenario_id'] != item['record']['id']
                   or row['task'] != item['record']['task']
                   or any(row[field] != pred[field] for field in ('raw', 'truncated'))
                   for item, row, pred in zip(cases, rows, predictions))):
        raise ValueError(f'Saved predictions do not match the frozen benchmark at {folder}')
    return predictions, rows, scores['scoring']


def generation_summary(predictions):
    reasons = Counter(p.get('stop_reason') or ('context_budget' if p.get('error') else
                      'token_limit' if p['truncated'] else 'eos') for p in predictions)
    tokens = [p.get('generated_tokens', 0) for p in predictions]
    return {
        'truncated_rate': sum(p['truncated'] for p in predictions) / len(predictions),
        'token_limit_rate': reasons['token_limit'] / len(predictions),
        'repetition_stop_rate': reasons['repetition'] / len(predictions),
        'stop_counts': dict(reasons),
        'generated_tokens': {'total': sum(tokens), 'mean': statistics.mean(tokens),
                             'median': statistics.median(tokens), 'max': max(tokens)},
    }


def evaluate_predictions(cases, predictions, schemas, judge=None, saved_rows=None):
    if len(predictions) != len(cases):
        raise ValueError('Incomplete generation')
    rows = []
    for index, (item, prediction) in enumerate(zip(cases, predictions)):
        record = item['record']
        if saved_rows is not None:
            scores = saved_rows[index]
        elif judge is not None:
            if index % 20 == 0:
                print(f'Semantic scoring: {index}/{len(cases)}', flush=True)
            scores = judge.grade(item, prediction)
        else:
            scores = score_response(record, item['reference'], prediction['raw'],
                                    schemas, prediction['truncated'])
        rows.append({**scores, **prediction, 'scenario_id': record['id'], 'task': record['task']})
    tasks = dict.fromkeys(row['task'] for row in rows)
    return rows, {**aggregate_scores(rows, tasks, 1), **generation_summary(predictions)}


def comparison(reports):
    comparisons = {}
    for left, right in (
        ('A/initial', 'A/sft'), ('A/sft', 'A/sft_orpo'),
        ('A/initial', 'A/sft_orpo'),
        *((f'A/{stage}', f'B/{stage}') for stage in STAGES),
    ):
        if left not in reports or right not in reports:
            continue
        first, second = reports[left], reports[right]
        comparisons[f'{right} minus {left}'] = {
            **{field: second[field] - first[field] for field in
               ('macro_task_score', 'schema_evidence_valid_rate', 'truncated_rate', 'token_limit_rate')},
            'mean_generated_tokens': second['generated_tokens']['mean'] - first['generated_tokens']['mean'],
            'per_task': {task: second['per_task'][task] - first['per_task'][task]
                         for task in first['per_task']},
        }
    return comparisons


def checkpoint_config(checkpoints, config):
    """Match the saved rsLoRA shape before using the original Unsloth loader."""
    fields = ('base_model_name_or_path', 'r', 'lora_alpha', 'lora_dropout', 'use_rslora',
              'peft_type', 'task_type', 'target_modules', 'bias')
    expected_modules = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    signatures = []
    fingerprints = {}
    for stage, path in checkpoints.items():
        metadata = path / 'adapter_config.json'
        adapter = json.loads(metadata.read_text())
        if (adapter['peft_type'] != 'LORA' or adapter['task_type'] != 'CAUSAL_LM'
                or adapter['bias'] != 'none' or set(adapter['target_modules']) != expected_modules
                or any(adapter.get(key) for key in ('revision', 'rank_pattern', 'alpha_pattern',
                                                   'modules_to_save', 'use_dora', 'lora_bias', 'use_qalora'))):
            raise ValueError(f'Unsupported Unsloth adapter configuration at {path}')
        signatures.append({key: adapter[key] for key in fields if key != 'target_modules'})
        fingerprints[stage] = {
            'config': file_sha256(metadata),
            'weights': file_sha256(path / 'adapter_model.safetensors'),
        }
    first = signatures[0]
    if any(signature != first for signature in signatures):
        raise ValueError('Checkpoints must use the same base model and LoRA settings')
    config['training'].update(lora_rank=first['r'], lora_alpha=first['lora_alpha'],
                              lora_dropout=first['lora_dropout'], use_rslora=first['use_rslora'])
    return first['base_model_name_or_path'], fingerprints


def run(run_dir, output, *, smoke=False, reuse_original=False, judge_budget=None, greedy_only=False):
    if judge_budget is not None:
        raise ValueError('Use distillation_loop.semantic_eval to preflight and authorize paid judging of saved predictions')
    cases = frozen_cases(run_dir)
    benchmark_hash = digest(cases)
    if smoke:
        # One case for each task, in the order from the frozen benchmark.
        first_per_task = {}
        for item in cases:
            first_per_task.setdefault(item['record']['task'], item)
        cases = list(first_per_task.values())
    schemas = json.loads((ROOT / 'data/self_instruct/task_schemas.json').read_text())['tasks']
    config = json.loads((run_dir / 'colab_config.json').read_text())
    if config['inference']['max_new_tokens'] != MAX_NEW_TOKENS:
        raise ValueError('Original Colab configuration must use the 1,024-token generation limit')
    # The original notebook loaded this evaluation with its L4 execution override.
    config['inference']['batch_size'] = 8
    checkpoints = {stage: run_dir / STAGES[stage] for stage in (('sft',) if smoke else STAGES)}
    model, fingerprints = checkpoint_config(checkpoints, config)
    payloads = [target_payload(item['record'], schemas) for item in cases]
    modes = ('A',) if greedy_only else ('A', 'B')
    jobs = [(mode, stage) for mode in modes for stage in checkpoints]
    replay = {}
    if reuse_original:
        replay = {('A', 'initial'): saved_predictions(run_dir / 'baseline', cases),
                  ('A', 'sft_orpo'): saved_predictions(run_dir / 'round_01/lighteval', cases)}

    if Path('/content/drive/MyDrive').is_dir():
        os.environ.setdefault('DISTILLATION_LOCAL_LOCK_DIR', '/content/distillation-eval-locks')
    store = Store(output)
    with store.lock():
        manifest = {'benchmark_hash': benchmark_hash, 'case_ids': [x['record']['id'] for x in cases],
                    'model': model, 'checkpoints': fingerprints, 'inference': config['inference'],
                    'prompt': {'evidence_first': config['training'].get('evidence_first', False),
                               'context_length': config['training']['max_sequence_length']},
                    'repetition': {'min_block_width': MIN_REPEAT_WIDTH,
                                   'max_block_width': MAX_REPEAT_WIDTH, 'copies': REPEATS},
                    'reuse_original': {f'{mode}/{stage}': digest(predictions)
                                       for (mode, stage), (predictions, _, _) in replay.items()}}
        manifest = json.loads(json.dumps(manifest))
        path = store.path('manifest.json')
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError('Evaluation inputs changed. Use a fresh output directory.')
        if not path.exists():
            atomic_json(path, manifest)

        runtime = None
        for mode, stage in jobs:
            path = store.path(f'{mode}_{stage}_generation.json')
            if path.exists():
                print(f'Reusing {mode}/{stage} ({len(cases)} cases)', flush=True)
                continue
            if (mode, stage) in replay:
                print(f'Reusing original {mode}/{stage} ({len(cases)} cases)', flush=True)
                predictions = replay[mode, stage][0]
            else:
                if runtime is None:
                    from .colab import UnslothStudentRuntime
                    runtime = UnslothStudentRuntime.load(model, config)
                    atomic_json(store.path('runtime.json'), runtime.identity())
                partial_path = store.path(f'{mode}_{stage}_partial.json')
                partial = json.loads(partial_path.read_text()) if partial_path.exists() else {}
                completed = partial.get('predictions')
                done = sum(p is not None for p in completed) if completed is not None else 0
                print(f'Generating {mode}/{stage}: {done}/{len(cases)} already saved', flush=True)
                runtime.restore(checkpoints[stage])
                runtime.repetition_stopping = RepetitionStopping if mode == 'B' else None
                def save_batch(rows, batch_size):
                    atomic_json(partial_path, {'predictions': rows, 'batch_size': batch_size})

                predictions = runtime.infer_many(payloads, batch_size=partial.get('batch_size'),
                                                 completed=completed, on_batch=save_batch)
            if len(predictions) != len(cases) or any(p is None for p in predictions):
                raise ValueError(f'Incomplete generations for {mode}/{stage}')
            atomic_json(path, predictions)
            store.path(f'{mode}_{stage}_partial.json').unlink(missing_ok=True)
        if runtime is not None:
            del runtime

        return score_jobs(store, jobs, cases, schemas, replay)


def score_jobs(store, jobs, cases, schemas, replay, judge=None):
    reports = {}
    scoring = judge.version if judge else 'strict-reference'
    for mode, stage in jobs:
        predictions = json.loads(store.path(f'{mode}_{stage}_generation.json').read_text())
        saved_rows = None
        if judge and (mode, stage) in replay:
            _, saved_rows, saved_scoring = replay[mode, stage]
            if saved_scoring != scoring:
                raise ValueError('The original reports used a different semantic grading rubric')
        rows, report = evaluate_predictions(cases, predictions, schemas, judge, saved_rows)
        atomic_jsonl(store.path(f'{mode}_{stage}_{scoring}.jsonl'), rows)
        reports[f'{mode}/{stage}'] = report
    result = {'scoring': scoring, 'cases': len(cases), 'runs': reports, 'deltas': comparison(reports)}
    prefix = 'comparison' if any(mode == 'B' for mode, _ in jobs) else 'comparison_greedy'
    atomic_json(store.path(f'{prefix}_{scoring}.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Original Colab run directory')
    parser.add_argument('--output', type=Path, required=True, help='Separate directory for diagnostic results')
    parser.add_argument('--smoke', action='store_true', help='SFT only, one frozen case per task')
    parser.add_argument('--reuse-original', action='store_true',
                        help='Reuse original Pass A predictions for initial and SFT+ORPO')
    parser.add_argument('--greedy-only', action='store_true', help='Run Pass A and save its comparison first')
    parser.add_argument('--judge-budget-usd', help='Legacy flag. Paid judging has moved to distillation_loop.semantic_eval')
    args = parser.parse_args()
    if args.output.resolve() == args.run_dir.resolve():
        parser.error('Use a separate output directory')
    if args.smoke and args.reuse_original:
        parser.error('--reuse-original is only available for the full 360-case comparison')
    result = run(args.run_dir, args.output, smoke=args.smoke, reuse_original=args.reuse_original,
                 judge_budget=args.judge_budget_usd, greedy_only=args.greedy_only)
    print(f'{result["cases"]} cases per run. Reports saved under {args.output}.')


if __name__ == '__main__':
    main()
