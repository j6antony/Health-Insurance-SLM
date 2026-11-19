"""Seeded feasible states, pairwise coverage, and lineage-preserving interventions."""
from copy import deepcopy
from datetime import date, timedelta
from itertools import combinations, product
import random

from .rules import CATEGORIES, REQUIRED, VERSION, evaluate, validate_state
from .storage import digest

GENERATOR_VERSION = 'python-feasible-pairwise-v2'

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
    if case == 'valid' and condition is None:
        case = 'unknown_applicability'
    identity = digest([VERSION, GENERATOR_VERSION, seed, index, category, condition, case])[:20]
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
    target = next(d for d in docs if d['type'] == REQUIRED[category][-1])
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
                 'variant_id': 'V-' + digest([identity, layout])[:20], 'parent_variant_id': None, 'category': category,
                 'rule_pack_version': VERSION, 'facts': {'patient_id': patient,
                 'claim_id': 'SYN-C-' + identity, 'service_date': day, 'referral_required': condition},
                 'documents': docs, 'seed': seed, 'layout': layout,
                 'provenance': {'generator': GENERATOR_VERSION, 'case': case,
                                'dimensions': {'category': category, 'condition': str(condition), 'case': case, 'layout': str(layout)}}})


def pairs(dimensions):
    return {tuple(sorted(pair)) for pair in combinations(dimensions.items(), 2)}


def generate(count, seed=42, prior_coverage=(), focus=None):
    if type(count) is not int or not 1 <= count <= 10000:
        raise ValueError('Scenario count must be in 1..10000')
    candidates = [c for c in product(CATEGORIES, (True, False, None), CASES, range(9))
                  if not (c[2] == 'valid' and c[1] is None)]
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
        if index < 4:
            desired = ('valid', 'unreadable', 'date_conflict', 'missing')[index]
            pool = [c for c in pool if c[2] == desired]
        choice = max(pool, key=priority)
        candidates.remove(choice)
        if not candidates:
            candidates = [c for c in product(CATEGORIES, (True, False, None), CASES, range(9))
                  if not (c[2] == 'valid' and c[1] is None)]
        row = make_scenario(seed, index, *choice)
        rows.append(row)
        seen.update(pairs(row['provenance']['dimensions']))
    return {'scenarios': rows, 'coverage_pairs': sorted(seen),
            'coverage_basis': 'observed feasible dimension pairs; not a claim of exhaustive coverage'}


OPERATIONS = ('remove_document', 'restore_document', 'make_unreadable', 'repair_readability',
              'remove_signature', 'introduce_date_mismatch', 'fix_dates', 'remove_field',
              'satisfy_condition', 'make_valid')


def expand(s, operation, target_document_id=None, comparison_document_id=None):
    verify_scenario(s)
    if operation not in OPERATIONS:
        raise ValueError('Unknown counterfactual operation')
    new = deepcopy(s)
    by_id = {d['id']: d for d in new['documents']}
    default_type = REQUIRED[s['category']][-1]
    default = next(d for d in new['documents'] if d['type'] == default_type)
    target = by_id.get(target_document_id or default['id'])
    if target is None:
        raise ValueError('Intervention target is not in the inventory')
    required = target['type'] in REQUIRED[s['category']] or (target['type'] == 'referral' and s['facts']['referral_required'] is True)
    if operation == 'remove_document' and not required:
        raise ValueError('Removal requires an applicable required document')
    new['parent_variant_id'] = s['variant_id']
    new['variant_id'] = 'V-' + digest([s['content_hash'], operation, target['id'], comparison_document_id])[:20]
    new['provenance'] = {**s['provenance'], 'intervention': operation, 'target_document_id': target['id'],
                         'comparison_document_id': comparison_document_id}

    def restore(d):
        d.update(present=True, readability='readable', signature=True,
                 fields={'patient_id': new['facts']['patient_id'], 'claim_id': new['facts']['claim_id'],
                         'service_date': new['facts']['service_date'], **d.get('fields', {})})

    if operation == 'remove_document':
        if not target['present']:
            raise ValueError('Removal requires a present document')
        target.update(present=False, readability='not_available', signature=None, fields={})
    elif operation == 'restore_document':
        if target['present']:
            raise ValueError('Restore requires an absent document')
        restore(target)
    elif operation == 'make_unreadable':
        if not target['present'] or target['readability'] != 'readable':
            raise ValueError('Unreadability intervention requires a readable present document')
        target['readability'] = 'unreadable'
    elif operation == 'repair_readability':
        if not target['present'] or target['readability'] == 'readable':
            raise ValueError('Readability repair requires present unreadable evidence')
        target['readability'] = 'readable'
    elif operation in ('remove_signature', 'introduce_date_mismatch', 'remove_field', 'fix_dates'):
        if not target['present'] or target['readability'] != 'readable':
            raise ValueError('Content intervention requires readable present evidence')
        if operation == 'remove_signature':
            if target['signature'] is False:
                raise ValueError('Signature already absent')
            target['signature'] = False
        elif operation == 'remove_field':
            if 'service_date' not in target['fields']:
                raise ValueError('Service date already absent')
            target['fields'].pop('service_date')
        else:
            other = by_id.get(comparison_document_id) if comparison_document_id else next(
                (d for d in new['documents'] if d['id'] != target['id'] and d['present']
                 and d['readability'] == 'readable' and d['fields'].get('service_date')), None)
            if other is None or other['id'] == target['id'] or not other['present'] or other['readability'] != 'readable' or not other['fields'].get('service_date'):
                raise ValueError('Date intervention needs a distinct readable comparison document')
            new['provenance']['comparison_document_id'] = other['id']
            day = other['fields']['service_date']
            target['fields']['service_date'] = ((date.fromisoformat(day) + timedelta(days=1)).isoformat()
                                               if operation == 'introduce_date_mismatch' else day)
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
            d['fields']['patient_id'] = new['facts']['patient_id']
            d['fields']['service_date'] = new['facts']['service_date']
        if new['facts']['referral_required'] is None:
            new['facts']['referral_required'] = False
    if new['documents'] == s['documents'] and new['facts'] == s['facts']:
        raise ValueError('Intervention did not change source facts')
    return seal(new)


def meaningful_variants(s, count):
    """Only distinct, feasible interventions; rotate negative skills by family."""
    negative = ['remove_document', 'make_unreadable', 'introduce_date_mismatch', 'remove_signature', 'remove_field']
    offset = int(digest(s['family_id'])[:8], 16) % len(negative)
    ops = ['make_valid'] + negative[offset:] + negative[:offset] + ['restore_document', 'repair_readability', 'fix_dates', 'satisfy_condition']
    rows, seen = [], {digest([s['facts'], s['documents']])}
    for op in ops:
        try:
            variant = expand(s, op)
        except ValueError:
            continue
        key = digest([variant['facts'], variant['documents']])
        if key not in seen:
            rows.append(variant)
            seen.add(key)
        if len(rows) >= count:
            break
    return rows if count else []
