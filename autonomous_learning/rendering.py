"""Text rendering with an independently parsed, closed-world faithfulness gate.

The accepted grammar deliberately rejects unconstrained prose. Rich clinical prose
requires a separately validated semantic verifier; it is not silently trusted.
"""
from collections import Counter
import re
from .rules import HEADINGS, VERSION, rule_pack
from .scenarios import verify_scenario
from .storage import digest

ALIASES = {'patient_id': ('Patient identifier', 'Patient ID', 'Subject'),
           'claim_id': ('Claim identifier', 'Claim ID', 'File'),
           'service_date': ('Service date', 'Date of service', 'Encounter date'),
           'charges': ('Charges', 'Line items', 'Item amounts'),
           'total': ('Total', 'Invoice total', 'Total amount')}
REVERSE = {label: key for key, labels in ALIASES.items() for label in labels}
SIGNATURE = {True: 'present', False: 'absent', None: 'unknown'}


def format_value(key, value, layout):
    if key == 'charges':
        return ', '.join(value)
    if key == 'service_date' and layout == 1:
        y, m, d = value.split('-')
        return f'{d}/{m}/{y}'
    return str(value)


def render_document(d, layout=0):
    if not d['present']:
        return None
    lines = [HEADINGS[d['type']], 'Synthetic fictional training document']
    if d['readability'] != 'readable':
        lines.append('Contents: ' + d['readability'])
    else:
        entries = list(d['fields'].items())
        if layout == 2:
            entries.reverse()
        lines += [f'{ALIASES[key][layout]}: {format_value(key, value, layout)}' for key, value in entries]
        lines.append('Signature: ' + SIGNATURE[d['signature']])
    return {'document_id': d['id'], 'page_number': 1, 'text': '\n'.join(lines)}


def render(s):
    verify_scenario(s)
    pages = [render_document(d, s['layout']) for d in s['documents'] if d['present']]
    return {'scenario_hash': s['content_hash'], 'pages': pages, 'renderer': 'python-v1',
            'render_hash': digest(pages)}


def parse_text(text):
    lines = text.splitlines()
    if len(lines) < 3 or lines[0] not in HEADINGS.values() or lines[1] != 'Synthetic fictional training document':
        raise ValueError('Unrecognized document heading or unverified narrative')
    kind = next(k for k, v in HEADINGS.items() if v == lines[0])
    fields, signature, readability = {}, None, 'readable'
    seen = set()
    for line in lines[2:]:
        label, sep, value = line.partition(': ')
        key = REVERSE.get(label, label)
        if not sep or key in seen:
            raise ValueError('Unrecognized, duplicate or conflicting text field')
        seen.add(key)
        if key == 'Signature':
            if value not in SIGNATURE.values():
                raise ValueError('Unverifiable signature')
            signature = next(k for k, v in SIGNATURE.items() if v == value)
        elif key == 'Contents':
            if value not in ('unreadable', 'unknown'):
                raise ValueError('Unknown content state')
            readability = value
        elif label in REVERSE:
            if key == 'service_date' and re.fullmatch(r'\d{2}/\d{2}/\d{4}', value):
                d, m, y = value.split('/')
                value = f'{y}-{m}-{d}'
            fields[key] = value.split(', ') if key == 'charges' else value
        else:
            raise ValueError('Unverified additional statement')
    if readability == 'readable' and 'Signature' not in seen:
        raise ValueError('Signature condition not stated')
    if readability != 'readable' and (fields or 'Signature' in seen):
        raise ValueError('Unreadable evidence cannot also supply legible fields')
    return {'type': kind, 'fields': fields, 'signature': signature, 'readability': readability}


def verify(s, rendered):
    verify_scenario(s)
    if rendered['scenario_hash'] != s['content_hash'] or digest(rendered['pages']) != rendered['render_hash']:
        raise ValueError('Rendered identity changed')
    expected = {d['id']: d for d in s['documents'] if d['present']}
    pages = rendered['pages']
    if Counter(p['document_id'] for p in pages) != Counter(expected.keys()):
        raise ValueError('Rendered document presence/identity mismatch')
    evidence = []
    for p in pages:
        if p['page_number'] != 1:
            raise ValueError('Unsupported page number')
        d = expected[p['document_id']]
        parsed = parse_text(p['text'])
        wanted = {k: d[k] for k in ('type', 'signature', 'readability', 'fields')}
        if d['readability'] != 'readable':
            wanted.update(fields={}, signature=None)
        if parsed != wanted:
            raise ValueError('Final text does not preserve source facts')
        offset = 0
        for line in p['text'].splitlines(keepends=True):
            quote = line.rstrip('\n')
            evidence.append({'document_id': p['document_id'], 'page_number': 1,
                             'start': offset, 'end': offset + len(quote), 'quote': quote})
            offset += len(line)
    return {'status': 'verified', 'scenario_hash': s['content_hash'], 'render_hash': rendered['render_hash'],
            'evidence': evidence, 'verification': 'closed-grammar final-text parse', 'version': VERSION}


def visible_package(s, rendered):
    """Explicit inventory and fictional rules are source evidence, not answer keys."""
    inventory = [{'document_id': d['id'], 'document_type': d['type'] if not d['present'] else None,
                  'present': d['present']} for d in s['documents']]
    # Type labels for present documents are inferred from the actual heading.
    return {'synthetic': True, 'claim_category': s['category'],
            'claim_facts': s['facts'], 'inventory': inventory, 'documents': rendered['pages'],
            'fictional_rule_pack': rule_pack()}
