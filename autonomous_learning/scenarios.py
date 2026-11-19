"""Seeded feasible states, pairwise coverage, and lineage-preserving interventions."""
from copy import deepcopy
from datetime import date, timedelta
from itertools import combinations, product
import random

from .rules import CATEGORIES, REQUIRED, VERSION, evaluate, validate_state
from .storage import digest

CASES = ('valid', 'missing', 'unreadable', 'unsigned', 'incomplete', 'date_conflict', 'unknown_signature')


def seal(s):
    s['findings'] = evaluate(s)
    s['relevant_rule_ids'] = [f['rule_id'] for f in s['findings']]
    s.pop('content_hash', None)
    s['content_hash'] = digest(s)
    return s


def verify_scenario(s):
    validate_state(s)
    body = {k: v for k, v in s.items() if k != 'content_hash'}
    if digest(body) != s['content_hash'] or evaluate(s) != s['findings']:
        raise ValueError('Scenario identity or deterministic findings changed')


def make_scenario(seed, index, category, condition, case, layout):
    rng = random.Random(f'{seed}:{index}')
    identity = digest([VERSION, seed, index])[:20]
    patient = 'SYN-P-' + identity[:10]
    day = (date(2025, 1, 1) + timedelta(days=rng.randrange(500))).isoformat()
    types = list(REQUIRED[category]) + (['referral'] if condition is True else [])
    docs = []
    for n, kind in enumerate(types):
        charges = [f'{rng.randrange(100, 10000) / 100:.2f}' for _ in range(2)]
        from decimal import Decimal
        fields = {'patient_id': patient, 'claim_id': 'SYN-C-' + identity,
                  'service_date': day}
        if kind in ('hospital_bill', 'invoice', 'pharmacy_invoice'):
            fields.update(charges=charges, total=str(sum(map(Decimal, charges))))
        d = {'id': f'DOC-{n + 1}', 'type': kind, 'present': True,
             'readability': 'readable', 'signature': True, 'fields': fields}
        docs.append(d)
    target = docs[-1]
    if case == 'missing':
        target.update(present=False, readability='not_available', signature=None, fields={})
    elif case == 'unreadable':
        target['readability'] = 'unreadable'
    elif case == 'unsigned':
        target['signature'] = False
    elif case == 'incomplete':
        target['fields'].pop('service_date')
    elif case == 'date_conflict':
        target['fields']['service_date'] = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    elif case == 'unknown_signature':
        target['signature'] = None
    return seal({'version': 1, 'synthetic': True, 'family_id': 'F-' + identity,
                 'variant_id': 'V-' + identity, 'parent_variant_id': None, 'category': category,
                 'rule_pack_version': VERSION, 'facts': {'patient_id': patient,
                 'claim_id': 'SYN-C-' + identity, 'service_date': day, 'referral_required': condition},
                 'documents': docs, 'seed': seed, 'layout': layout,
                 'provenance': {'generator': 'python-feasible-pairwise-v1', 'case': case,
                                'dimensions': {'category': category, 'condition': str(condition), 'case': case, 'layout': str(layout)}}})


def pairs(dimensions):
    return {tuple(sorted(pair)) for pair in combinations(dimensions.items(), 2)}


def generate(count, seed=42, prior_coverage=(), focus=None):
    if type(count) is not int or not 1 <= count <= 10000:
        raise ValueError('Scenario count must be in 1..10000')
    candidates = list(product(CATEGORIES, (True, False, None), CASES, range(3)))
    random.Random(seed).shuffle(candidates)
    seen = {tuple(tuple(x) for x in item) for item in prior_coverage}
    rows = []
    for index in range(count):
        def priority(c):
            dims = dict(zip(('category', 'condition', 'case', 'layout'), map(str, c)))
            weight = (focus or {}).get(c[2], 0)
            return len(pairs(dims) - seen), weight
        pool = candidates
        represented = {r['category'] for r in rows}
        if len(represented) < len(CATEGORIES):
            pool = [c for c in pool if c[0] not in represented]
        if index == 0:
            pool = [c for c in pool if c[2] == 'valid']
        choice = max(pool, key=priority)
        candidates.remove(choice)
        if not candidates:
            candidates = list(product(CATEGORIES, (True, False, None), CASES, range(3)))
        row = make_scenario(seed, index, *choice)
        rows.append(row)
        seen.update(pairs(row['provenance']['dimensions']))
    return {'scenarios': rows, 'coverage_pairs': sorted(seen),
            'coverage_basis': 'observed feasible dimension pairs; not a claim of exhaustive coverage'}


OPERATIONS = ('remove_document', 'restore_document', 'make_unreadable', 'repair_readability',
              'remove_signature', 'introduce_date_mismatch', 'fix_dates', 'remove_field',
              'satisfy_condition', 'make_valid')


def expand(s, operation):
    verify_scenario(s)
    if operation not in OPERATIONS:
        raise ValueError('Unknown counterfactual operation')
    new = deepcopy(s)
    new['parent_variant_id'] = s['variant_id']
    new['variant_id'] = 'V-' + digest([s['content_hash'], operation])[:20]
    new['provenance'] = {**s['provenance'], 'intervention': operation}
    target = new['documents'][-1]

    def restore(d):
        d.update(present=True, readability='readable', signature=True,
                 fields={'patient_id': new['facts']['patient_id'], 'claim_id': new['facts']['claim_id'],
                         'service_date': new['facts']['service_date']})

    if operation == 'remove_document':
        target.update(present=False, readability='not_available', signature=None, fields={})
    elif operation == 'restore_document':
        restore(target)
    elif operation == 'make_unreadable':
        if not target['present']:
            restore(target)
        target['readability'] = 'unreadable'
    elif operation == 'repair_readability':
        if not target['present']:
            restore(target)
        target['readability'] = 'readable'
    elif operation in ('remove_signature', 'introduce_date_mismatch', 'remove_field'):
        if not target['present']:
            restore(target)
        target['readability'] = 'readable'
        if operation == 'remove_signature':
            target['signature'] = False
        elif operation == 'remove_field':
            target['fields'].pop('service_date', None)
        else:
            target['fields']['service_date'] = (date.fromisoformat(new['facts']['service_date']) + timedelta(days=1)).isoformat()
    elif operation == 'fix_dates':
        for d in new['documents']:
            if d['present']:
                d['fields']['service_date'] = new['facts']['service_date']
    elif operation == 'satisfy_condition':
        new['facts']['referral_required'] = True
        referral = next((d for d in new['documents'] if d['type'] == 'referral'), None)
        if referral is None:
            referral = {'id': 'DOC-REFERRAL', 'type': 'referral'}
            new['documents'].append(referral)
        restore(referral)
    elif operation == 'make_valid':
        for d in new['documents']:
            restore(d)
        if new['facts']['referral_required'] is None:
            new['facts']['referral_required'] = False
    return seal(new)
