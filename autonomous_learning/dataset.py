"""Dataset QA, quarantine, family caps, replay and content-bound exports."""
from collections import Counter, defaultdict
import json
import re
from difflib import SequenceMatcher

from self_instruct_pipeline.validation import validate_schema, check_hidden_fields
from . import CONTRACT_VERSION
from .storage import digest, jsonl, atomic_write, checksum
from .tasks import TASKS, SCHEMAS, INSTRUCTIONS, target_from_visible


def validate_example(row):
    if row['contract_version'] != CONTRACT_VERSION or row['task'] not in TASKS:
        raise ValueError('Unsupported task contract')
    if row.get('verification') != 'synthetic_silver':
        raise ValueError('Only verified synthetic silver rows are exportable')
    payload = row['payload']
    if (set(payload) != {'instruction', 'scenario', 'response_schema'}
            or payload['response_schema'] != SCHEMAS[row['task']]
            or payload['instruction'] != INSTRUCTIONS[row['task']]):
        raise ValueError('Student input contract changed')
    if set(payload['scenario']) != {'synthetic', 'claim_category', 'claim_facts', 'inventory', 'documents', 'fictional_rule_pack'}:
        raise ValueError('Hidden generation metadata in student input')
    if payload['scenario']['synthetic'] is not True:
        raise ValueError('Unattended data must be synthetic')
    check_hidden_fields(payload['scenario'])
    if row['task'] == 'document_classification' and any(item['present'] and item['document_type'] is not None for item in payload['scenario']['inventory']):
        raise ValueError('Classification label leaked through typed inventory')
    validate_schema(row['target'], SCHEMAS[row['task']])
    pages = {(p['document_id'], p['page_number']): p['text'] for p in payload['scenario']['documents']}
    for e in row['target']['evidence']:
        text = pages.get((e['document_id'], e['page_number']), '')
        if not e['quote'] or text[e['start']:e['end']] != e['quote']:
            raise ValueError('Invalid evidence offsets')
    if row['target'] != target_from_visible(row):
        raise ValueError('Target inconsistent with independently parsed visible evidence')
    from .rules import rule_pack
    if payload['scenario']['fictional_rule_pack'] != rule_pack():
        raise ValueError('Fictional source rule pack changed')
    from .rendering import parse_text
    for text in pages.values():
        parse_text(text)
    from .evidence import check_offsets
    check_offsets(payload, row['target'])


def content(row):
    """Normalize synthetic IDs for duplicate checks; never use hidden targets."""
    text = json.dumps(row['payload']['scenario']['documents'], sort_keys=True)
    return re.sub(r'SYN-[PC]-[a-f0-9]+', 'SYN-ID', text).casefold()


def training_content(row):
    visible = row['payload']['scenario']
    value = json.dumps({k: visible[k] for k in ('documents', 'inventory', 'claim_facts', 'claim_category')}, sort_keys=True)
    return re.sub(r'SYN-[PC]-[a-f0-9]+', 'SYN-ID', value).casefold()


def reject_leakage(training, evaluation):
    families = {r['family_id'] for r in evaluation}
    ids = {r['id'] for r in evaluation}
    templates = {r['template_family'] for r in evaluation}
    texts = [content(r) for r in evaluation]
    for row in training:
        if row['family_id'] in families or row['id'] in ids:
            raise ValueError('Training/evaluation source-family or ID leakage')
        if row['template_family'] in templates:
            raise ValueError('Training/evaluation template-family leakage')
        text = content(row)
        if any(text == other or SequenceMatcher(None, text, other, autojunk=False).ratio() >= .98 for other in texts):
            raise ValueError('Training/evaluation near-duplicate leakage')


def build(rows, family_cap=8, evaluation=(), replay=(), replay_fraction=.25, initialization_strategy='continued', selection_seed=42):
    if initialization_strategy not in ('fresh', 'continued'):
        raise ValueError('Unsupported adapter initialization strategy')
    if type(family_cap) is not int or family_cap < 1 or not 0 <= replay_fraction < 1:
        raise ValueError('Invalid family cap/replay allocation')
    historical = {}
    for row in replay:
        previous = historical.get(row['id'])
        if previous is not None and (previous['payload'] != row['payload'] or previous['target'] != row['target']):
            raise ValueError('Historical example identity conflict')
        historical.setdefault(row['id'], row)
    replay = list(historical.values())
    reject_leakage(list(rows) + replay, evaluation)
    counts, seen_ids, seen_content = Counter(), set(), set()
    near_content = defaultdict(list)
    accepted, rejected = [], []

    def add(row, replay_row=False):
        try:
            validate_example(row)
            key = (row['task'], training_content(row))
            outcome = re.sub(r'SYN-[PC]-[a-f0-9]+', 'SYN-ID', json.dumps(row['target']['result'], sort_keys=True))
            near_bucket = (row['task'], outcome, row['target']['requires_review'])
            if row['id'] in seen_ids or key in seen_content:
                raise ValueError('duplicate')
            if any(SequenceMatcher(None, key[1], old).quick_ratio() >= .995
                   and SequenceMatcher(None, key[1], old, autojunk=False).ratio() >= .995
                   for old in near_content[near_bucket]):
                raise ValueError('near_duplicate')
            if counts[row['family_id']] >= family_cap:
                raise ValueError('family_cap')
            counts[row['family_id']] += 1
            seen_ids.add(row['id'])
            seen_content.add(key)
            near_content[near_bucket].append(key[1])
            accepted.append({**row, 'replay': replay_row})
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({'id': row.get('id'), 'reason': str(exc), 'row': row})
    # Greedy set coverage, with local task coverage and global deficit pressure.
    # Tie breaking uses a stable seed and content identity, never list order.
    families = defaultdict(list)
    for row in rows:
        families[row['family_id']].append(row)
    covered, task_counts = set(), Counter()

    def features(row):
        metadata = row.get('selection', {})
        result = row['target']['result']
        statuses = tuple(sorted({v['status'] for v in result if isinstance(v, dict) and 'status' in v}))
        outcome = (statuses, row['target']['requires_review'], bool(result))
        values = {('task', row['task']), ('task_outcome', row['task'], str(outcome)),
                  ('variant', row['variant_id']), ('layout', str(row['layout']))}
        for key in ('category', 'intervention', 'control'):
            values.add((key, str(metadata.get(key, 'unspecified'))))
        for key in ('states', 'findings'):
            values.update((key, value) for value in metadata.get(key, []))
        return values

    for family in sorted(families, key=lambda f: digest([selection_seed, f])):
        pool, local = list(families[family]), set()
        while pool:
            def priority(row):
                f = features(row)
                return (int(('task', row['task']) not in local) * 100,
                        sum((8 if v[0] == 'task_outcome' else 3 if v[0] in ('variant', 'intervention', 'control') else 1)
                            for v in f - local),
                        len(f - covered), -task_counts[row['task']], digest([selection_seed, row['id']]))
            row = max(pool, key=priority)
            pool.remove(row)
            before_count = len(accepted)
            add(row)
            if len(accepted) != before_count:
                f = features(row)
                local.update(f)
                covered.update(f)
                task_counts[row['task']] += 1
    replay_target = (len({r['id'] for r in replay} - seen_ids) if initialization_strategy == 'fresh'
                     else round(len(accepted) * replay_fraction / (1 - replay_fraction)))
    pools = {t: sorted([r for r in replay if r['task'] == t], key=lambda r: r['id']) for t in TASKS}
    before = len(accepted)
    while len(accepted) - before < replay_target and any(pools.values()):
        for task in TASKS:
            if pools[task] and len(accepted) - before < replay_target:
                add(pools[task].pop(0), True)
    stats = {'candidates_generated': len(rows), 'selection_seed': selection_seed,
             'distinct_variants_retained': len({r['variant_id'] for r in accepted}),
             'distinct_findings_retained': sorted({v for r in accepted for v in r.get('selection', {}).get('findings', [])}),
             'intervention_distribution': dict(Counter(r.get('selection', {}).get('intervention', 'legacy') for r in accepted)),
             'control_distribution': dict(Counter(r.get('selection', {}).get('control', 'legacy') for r in accepted)),
             'evidence_spans': sum(len(r['target']['evidence']) for r in accepted),
             'token_totals': None, 'token_totals_status': 'requires real tokenizer preflight',
             'total_examples': len(accepted), 'unique_master_families': len(counts),
             'task_distribution': dict(Counter(r['task'] for r in accepted)),
             'variants_per_family': {f: len({r['variant_id'] for r in accepted if r['family_id'] == f}) for f in counts},
             'rule_coverage': dict(Counter(rule for r in accepted for rule in r['rule_ids'])),
             'layouts': dict(Counter(str(r['layout']) for r in accepted)),
             'rejected': len(rejected), 'rejection_reasons': dict(Counter(r['reason'] for r in rejected)),
             'duplicate_rate': sum(r['reason'] in ('duplicate', 'near_duplicate') for r in rejected) / max(1, len(accepted) + len(rejected)),
             'verification_rejection_rate': sum(r['reason'] not in ('duplicate', 'near_duplicate', 'family_cap') for r in rejected) / max(1, len(accepted) + len(rejected)),
             'replay_policy': 'cumulative' if initialization_strategy == 'fresh' else 'fixed_stratified',
             'replay_pool_size': len(replay), 'replay_target': replay_target, 'replay_actual': len(accepted) - before,
             'replay_shortfall': replay_target - (len(accepted) - before),
             'quality': 'automatically verified synthetic silver; not expert verified'}
    from .diversity import measure
    stats['surface_diversity'] = measure([p for r in accepted for p in r['payload']['scenario']['documents']])
    return {'rows': accepted, 'quarantine': rejected, 'statistics': stats}


def export(storage, dataset):
    rows = dataset['rows']
    if not rows:
        raise ValueError('Refusing empty training dataset')
    identity = digest({'rows': rows, 'statistics': dataset['statistics'], 'contract_version': CONTRACT_VERSION})
    base = f'datasets/approved_sft/{identity}'
    storage.preflight()
    directory = storage.path(base)
    directory.mkdir(parents=True, exist_ok=True)
    # Immutable content-addressed dataset exports.
    chats = [{'messages': [{'role': 'user', 'content': json.dumps(r['payload'], ensure_ascii=False)},
                            {'role': 'assistant', 'content': json.dumps(r['target'], ensure_ascii=False)}]} for r in rows]
    files = {'examples.jsonl': jsonl(rows), 'qwen3_8b_chat.jsonl': jsonl(chats)}
    hashes = {}
    for name, data in files.items():
        path = directory / name
        if path.exists() and path.read_text() != data:
            raise ValueError('Immutable dataset export changed')
        atomic_write(path, data)
        hashes[name] = checksum(path)
    manifest = {'contract_version': CONTRACT_VERSION, 'dataset_id': identity, 'examples_content_hash': digest(rows), 'files': hashes,
                'statistics': dataset['statistics'], 'family_ids': sorted({r['family_id'] for r in rows}),
                'split': 'training', 'template_families': sorted({r['template_family'] for r in rows})}
    if (directory / 'manifest.json').exists():
        from .storage import read_json
        if read_json(directory / 'manifest.json') != manifest:
            raise ValueError('Immutable dataset manifest changed')
    storage.write(base + '/manifest.json', manifest)
    return {'directory': base, 'manifest': manifest}


def sealed_index(cases):
    """One-way source fingerprints permit exclusions without exposing holdout text."""
    def shingles(text):
        tokens = re.findall(r'\w+', text)
        return sorted({digest(tokens[i:i + 5]) for i in range(max(1, len(tokens) - 4))})
    return {'families': sorted({r['family_id'] for r in cases}),
            'templates': sorted({r['template_family'] for r in cases}),
            'ids': sorted({r['id'] for r in cases}),
            'fingerprints': [{'hash': digest(content(r)), 'shingles': shingles(content(r))} for r in cases]}


def reject_sealed_leakage(training, index):
    families, templates, ids = map(set, (index['families'], index['templates'], index['ids']))
    for row in training:
        if row['family_id'] in families or row['template_family'] in templates or row['id'] in ids:
            raise ValueError('Sealed evaluation family/template/ID leakage')
        token = sealed_index([row])['fingerprints'][0]
        mine = set(token['shingles'])
        for candidate in index['fingerprints']:
            other = set(candidate['shingles'])
            if token['hash'] == candidate['hash'] or len(mine & other) / max(1, len(mine | other)) >= .98:
                raise ValueError('Sealed evaluation content/near-duplicate leakage')



def migrate_verified_replay(rows):
    """Recompute legacy synthetic silver targets; never mutate the original export."""
    from .evidence import EVIDENCE_POLICY
    migrated = []
    for row in rows:
        if row.get('verification') != 'synthetic_silver' or row['payload']['scenario'].get('synthetic') is not True:
            raise ValueError('Replay migration requires explicitly verified synthetic sources')
        target = target_from_visible(row)
        new = {**row, 'target': target, 'evidence_policy': EVIDENCE_POLICY,
               'id': digest([row['id'], EVIDENCE_POLICY]),
               'migration': {'source_example_id': row['id'], 'source_sha256': digest(row),
                             'method': 'final-text deterministic target reconstruction'},
               'selection': {**row.get('selection', {}), 'category': row['payload']['scenario']['claim_category'],
                             'intervention': 'revalidated_replay'}}
        validate_example(new)
        migrated.append(new)
    return migrated
