"""Verify audit inputs offline before reading case annotations."""
import argparse
import json
from collections import Counter
from pathlib import Path

from distillation_loop.benchmarks import load_saved_split
from distillation_loop.leakage import ExclusionRegistry, historical_records
from distillation_loop.semantic_eval import CORPUS, SCHEMAS, load_comparison
from self_instruct_pipeline.persistence import digest, read_jsonl


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reconcile(packet_path, diagnostic, split, comparison, plan):
    packet = read_jsonl(packet_path)
    schemas = json.loads(SCHEMAS.read_text())['tasks']
    development, holdout = load_saved_split(split, CORPUS, schemas)
    manifest, generations = load_comparison(comparison, development)
    identity = diagnostic['identity']
    expected = {
        'comparison_manifest_sha256': digest(manifest),
        'split_sha256': digest(json.loads(split.read_text())),
        'development_sha256': digest(development),
        'holdout_sha256': digest(holdout),
        'generations': {stage: digest(rows) for stage, rows in generations.items()},
    }
    for field, value in expected.items():
        require(identity[field] == value, f'Diagnostic identity mismatch: {field}')
    reasons = diagnostic['diagnostic_decision']['operational_safety']['reasons']
    flags = {}
    for reason in reasons:
        for case_id in reason['cases']:
            flags.setdefault(case_id, set()).add(reason['code'])
    require(len(packet) == len(flags) == 46 and
            {row['scenario_id'] for row in packet} == flags.keys(),
            'Packet must contain exactly the 46 unique diagnostic cases')
    positions = {row['record']['id']: i for i, row in enumerate(development)}
    for row in packet:
        case_id = row['scenario_id']
        i = positions[case_id]
        source = development[i]
        require(set(row['flag_types']) == flags[case_id], f'Flag mismatch: {case_id}')
        for field in ('record', 'reference', 'source_group', 'accepted_references'):
            require(row.get(field, []) == source.get(field, []), f'{field} mismatch: {case_id}')
        for stage in ('initial', 'sft'):
            require(row[f'{stage}_prediction'] == generations[stage][i],
                    f'{stage} generation mismatch: {case_id}')
            scores = diagnostic['runs'][stage]['case_scores'][case_id]
            require(row[f'{stage}_case_score'] == scores, f'{stage} score mismatch: {case_id}')
    history, plan_hash = historical_records(plan, schemas)
    benchmark = ExclusionRegistry(development + holdout)
    benchmark.require_clean(history, 'historical Round 1 audit')
    return {'cases': len(packet), 'development': len(development), 'holdout': len(holdout),
            'training_plan_hash': plan_hash, 'training_tasks': dict(Counter(r['task'] for r in history)),
            'training_case_types': dict(Counter(r['case_type'] for r in history))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'diagnostic', 'split', 'comparison', 'plan'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.diagnostic = json.loads(args.diagnostic.read_text())
    print(json.dumps(reconcile(
        args.packet, args.diagnostic, args.split, args.comparison, args.plan), indent=2))


if __name__ == '__main__':
    main()
