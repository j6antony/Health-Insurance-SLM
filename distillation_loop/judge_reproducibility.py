"""Prepare and analyze independent judge repetitions without making API calls."""
import argparse
import json
import math
from collections import Counter, defaultdict
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from statistics import mean

from self_instruct_pipeline.budget import reservation_usd
from self_instruct_pipeline.config import load_config
from self_instruct_pipeline.parent_backend import OpenAIParent
from self_instruct_pipeline.persistence import atomic_json, digest, read_jsonl
from self_instruct_pipeline.validation import validate_schema

from .benchmarks import load_saved_split
from .correction import judge_messages
from .evaluation import incomplete_prediction
from .review import DIMENSIONS, grade_reviewed
from .semantic import JUDGMENT_INSTRUCTION, JUDGMENT_SCHEMA, SemanticJudge, judgment_payload
from .semantic_eval import CORPUS, SCHEMAS, load_comparison

ARMS = ('medium-1', 'medium-2', 'none-1', 'none-2')


def prepare(comparison, split, *, checkpoint='sft', include_none=True, corpus=CORPUS, schemas_path=SCHEMAS):
    schemas = json.loads(schemas_path.read_text())['tasks']
    cases, _ = load_saved_split(split, corpus, schemas)
    manifest, generations = load_comparison(comparison, cases)
    parent = load_config()['parent']
    if parent['provider'] != 'openai' or parent.get('reasoning_effort') != 'medium':
        raise ValueError('Reproducibility plan requires the configured medium-effort OpenAI judge')
    arms = ARMS if include_none else ARMS[:2]
    experiment = {'benchmark_sha256': digest(cases), 'comparison_sha256': digest(manifest),
                  'generation_sha256': digest(generations[checkpoint]), 'checkpoint': checkpoint,
                  'scoring': SemanticJudge.version, 'model': parent['model']}
    targets = {}
    costs = dict.fromkeys(arms, Decimal(0))
    for item, prediction in zip(cases, generations[checkpoint], strict=True):
        if incomplete_prediction(prediction) or grade_reviewed(item, prediction, schemas, {})[0] is not None:
            continue
        payload = judgment_payload(item, prediction, schemas)
        record = item['record']
        keys = {}
        for arm in arms:
            settings = {**parent, 'reasoning_effort': arm.split('-')[0]}
            body = OpenAIParent.request_body(settings, judge_messages(payload, JUDGMENT_INSTRUCTION),
                                             JUDGMENT_SCHEMA, temperature=0)
            keys[arm] = digest([experiment, arm, record['id'], body])
            costs[arm] += reservation_usd(settings, body) * (settings['max_retries'] + 1)
        targets[record['id']] = {'task': record['task'], 'prediction_sha256': digest(prediction), 'keys': keys,
                                 'review_flag_correct': payload['candidate'].get('requires_review') ==
                                 item['reference'].get('requires_review')}
    return {'experiment': experiment, 'arms': list(arms), 'targets': targets,
            'max_usd_by_arm': {arm: str(cost) for arm, cost in costs.items()},
            'scope': 'development cases that require paid judging',
            'execution': 'offline plan only; verify none support before scheduling any paid work'}


def analyze(plan, rows):
    arms = plan['arms']
    targets = plan['targets']
    if not targets:
        raise ValueError('No judge-required cases for reproducibility analysis')
    result = {arm: {} for arm in arms}
    for row in rows:
        arm, case = row['arm'], row['scenario_id']
        if (arm not in result or case not in targets or case in result[arm]
                or row.get('cache_key') != targets[case]['keys'][arm]):
            raise ValueError('Duplicate, unknown, or mismatched judge repetition')
        validate_schema(row['judgment'], JUDGMENT_SCHEMA)
        cost = Decimal(str(row['cost_usd']))
        latency = row['latency_seconds']
        if (not cost.is_finite() or cost < 0 or type(latency) not in (int, float)
                or not math.isfinite(latency) or latency < 0):
            raise ValueError('Invalid judge cost or latency')
        result[arm][case] = (row['judgment'], cost, latency)
    if any(entries.keys() != targets.keys() for entries in result.values()):
        raise ValueError('Reproducibility analysis requires all planned judgments')

    def passed(assessment):
        return all(assessment[name] for name in DIMENSIONS[:3]) and not assessment['unsupported_claims']

    comparisons = {}
    for left, right in combinations(arms, 2):
        totals = defaultdict(Counter)
        for case, spec in targets.items():
            a, b = result[left][case][0], result[right][case][0]
            for key in ('overall', spec['task']):
                counts = totals[key]
                counts['cases'] += 1
                counts['exact_agreement'] += a == b
                counts['disagreement'] += a != b
                counts['decision_correct_flips'] += a['decision_correct'] != b['decision_correct']
                counts['abstention_flips'] += a['abstention_correct'] != b['abstention_correct']
                counts['semantic_pass_flips'] += passed(a) != passed(b)
                counts['task_score_change_sum'] += (int(passed(b)) - int(passed(a))) if spec['review_flag_correct'] else 0
        comparisons[f'{right} vs {left}'] = {
            task: {**counts, 'task_score_change': counts['task_score_change_sum'] / counts['cases']}
            for task, counts in totals.items()}
    return {'experiment': plan['experiment'], 'per_arm': {
        arm: {'cost_usd': str(sum(row[1] for row in entries.values())),
              'mean_latency_seconds': mean(row[2] for row in entries.values())}
        for arm, entries in result.items()}, 'comparisons': comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    planner, analyzer = commands.add_parser('plan'), commands.add_parser('analyze')
    planner.add_argument('--comparison-dir', type=Path, required=True)
    planner.add_argument('--split', type=Path, required=True)
    planner.add_argument('--medium-only', action='store_true')
    analyzer.add_argument('--plan', type=Path, required=True)
    analyzer.add_argument('--judgments', type=Path, required=True)
    for command in (planner, analyzer):
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'plan':
        report = prepare(args.comparison_dir, args.split, include_none=not args.medium_only)
    else:
        report = analyze(json.loads(args.plan.read_text()), read_jsonl(args.judgments))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)


if __name__ == '__main__':
    main()
