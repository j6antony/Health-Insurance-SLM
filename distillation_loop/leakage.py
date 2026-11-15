"""Keep frozen evaluation examples out of every training and screening path."""
import json
from collections import Counter

from self_instruct_pipeline.deduplicator import Deduplicator, canonical_scenario
from self_instruct_pipeline.persistence import atomic_json, digest, read_jsonl
from self_instruct_pipeline.validation import validate_answer, validate_audit


def exclusion_record(item):
    record = item.get('record', item)
    if record is item:
        return record
    metadata = {key: item[key] for key in ('source_group', 'source_family', 'source_family_id', 'template_family')
                if item.get(key)}
    group = item.get('review', {}).get('source_group')
    if group:
        metadata.setdefault('source_group', group)
    return {**metadata, **record}


def source_groups(record):
    return {record[key] for key in ('source_group', 'source_family', 'source_family_id', 'template_family')
            if isinstance(record.get(key), str) and record[key]}


def scenario_content(record):
    def normalize(value):
        if isinstance(value, str):
            return ' '.join(value.casefold().split())
        if isinstance(value, list):
            return [normalize(v) for v in value]
        if isinstance(value, dict):
            return {key: normalize(v) for key, v in value.items()}
        return value

    scenario = normalize(canonical_scenario(record))
    if isinstance(scenario, dict) and 'documents' in scenario:
        scenario = {'documents': sorted(scenario['documents'], key=digest),
                    'context': scenario.get('context', {})}
    return digest(scenario)


class ExclusionRegistry:
    def __init__(self, rows, *, near_duplicates=True):
        self.ids, self.content, self.groups = set(), set(), set()
        self.dedup = Deduplicator([]) if near_duplicates else None
        for item in rows:
            self.add(exclusion_record(item))

    def add(self, record):
        self.ids.add(record['id'])
        self.content.add(scenario_content(record))
        self.groups.update(source_groups(record))
        if self.dedup is not None:
            self.dedup.add(record)

    def match(self, record):
        if record['id'] in self.ids:
            return 'scenario_id'
        if self.groups & source_groups(record):
            return 'source_family'
        if scenario_content(record) in self.content:
            return 'canonical_content'
        if self.dedup is not None:
            return self.dedup.check(record)
        return None

    def require_clean(self, rows, operation):
        for record in rows:
            reason = self.match(record)
            if reason:
                raise ValueError(f'Evaluation leakage in {operation}: {record["id"]} matches by {reason}')


def historical_records(path, schemas):
    plan = json.loads(path.read_text())
    pairs = plan['pairs']
    inputs = [pairs, plan['diagnostics']] if 'diagnostics' in plan else pairs
    ids = [row['record']['id'] for row in pairs]
    if len(ids) != 200 or len(set(ids)) != 200 or digest(inputs) != plan['hash']:
        raise ValueError('Historical Round 1 training plan must verify all 200 original pairs')
    for row in pairs:
        record = row['record']
        validate_answer(record, row['answer'], schemas[record['task']])
    return [exclusion_record(row) for row in pairs], plan['hash']


def freeze_loss_split(output, historical_plan, benchmarks, schemas, *, candidates_path=None, target=50):
    historical, history_hash = historical_records(historical_plan, schemas)
    historical_registry = ExclusionRegistry(historical)
    benchmark_registry = ExclusionRegistry(benchmarks)
    if type(target) is not int or target < 1:
        raise ValueError('Held-out loss target must be a positive integer')
    if candidates_path is not None and not candidates_path.is_file():
        raise ValueError(f'Missing verified teacher candidates at {candidates_path}')
    candidates = read_jsonl(candidates_path) if candidates_path is not None else []
    identity = {'historical_plan_sha256': history_hash, 'benchmark_sha256': digest(benchmarks),
                'candidates_sha256': digest(candidates), 'schema_sha256': digest(schemas), 'target': target}
    if output.exists():
        saved = json.loads(output.read_text())
        if saved['identity'] != identity or saved['split_sha256'] != digest(saved['rows']):
            raise ValueError('Frozen teacher-loss split identity changed')
        for registry in (historical_registry, benchmark_registry):
            registry.require_clean((row['record'] for row in saved['rows']), 'frozen teacher-loss split')
        return saved
    rejected = Counter()
    pools = {task: [] for task in sorted(schemas)}
    for row in candidates:
        try:
            record = exclusion_record(row)
            if (row.get('verification_status') != 'automatically_checked_silver'
                    or row.get('scenario_hash') != digest(row['record'])):
                raise ValueError('Candidate lacks bound teacher verification')
            validate_answer(record, row['answer'], schemas[record['task']])
            validate_audit(row['audit'])
        except (KeyError, TypeError, ValueError):
            rejected['unverified_or_invalid'] += 1
            continue
        match = historical_registry.match(record)
        overlap = benchmark_registry.match(record)
        if match or overlap:
            rejected[f'historical_{match}' if match else f'benchmark_{overlap}'] += 1
        else:
            pools[record['task']].append({'record': record, 'answer': row['answer']})
    for pool in pools.values():
        pool.sort(key=lambda row: digest(row['record']), reverse=True)
    rows, selected = [], ExclusionRegistry([])
    while len(rows) < target and any(pools.values()):
        for pool in pools.values():
            while pool and len(rows) < target:
                row = pool.pop()
                reason = selected.match(row['record'])
                if reason:
                    rejected[f'duplicate_{reason}'] += 1
                    continue
                selected.add(row['record'])
                rows.append(row)
                break
    saved = {'identity': identity, 'rows': rows, 'split_sha256': digest(rows),
             'per_task': dict(Counter(row['record']['task'] for row in rows)),
             'excluded': dict(rejected), 'limitations': []}
    if len(rows) < target:
        saved['limitations'].append(f'Only {len(rows)} independent verified examples available for a target of {target}')
    missing = sorted(set(schemas) - saved['per_task'].keys())
    if missing:
        saved['limitations'].append(f'Missing task coverage: {", ".join(missing)}')
    atomic_json(output, saved)
    return saved


def validate_frozen_training_records(store, registry):
    for path in sorted(store.directory.glob('round_*/training_plan.json')):
        plan = json.loads(path.read_text())
        rows = plan['pairs']
        inputs = [rows, plan['diagnostics']] if 'diagnostics' in plan else rows
        if digest(inputs) != plan['hash']:
            raise ValueError(f'Frozen training plan hash mismatch at {path}')
        registry.require_clean((exclusion_record(row) for row in rows), path.name)
    for path in sorted(store.directory.glob('round_*/corrections/*.json')):
        row = json.loads(path.read_text())
        registry.require_clean([row['scenario']], 'cached student correction')


def main():
    """Validate historical exclusions and freeze independently sourced loss examples."""
    import argparse
    from pathlib import Path
    from self_instruct_pipeline.persistence import Store
    from .benchmarks import load_saved_split
    from .semantic_eval import CORPUS, SCHEMAS

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--training-plan', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidates', type=Path)
    parser.add_argument('--corpus', type=Path, default=CORPUS)
    parser.add_argument('--schemas', type=Path, default=SCHEMAS)
    args = parser.parse_args()
    schemas = json.loads(args.schemas.read_text())['tasks']
    development, holdout = load_saved_split(args.split, args.corpus, schemas)
    store = Store(args.output)
    with store.lock():
        result = freeze_loss_split(store.path('heldout_teacher_loss_split.json'),
                                   args.training_plan, development + holdout, schemas,
                                   candidates_path=args.candidates)
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
