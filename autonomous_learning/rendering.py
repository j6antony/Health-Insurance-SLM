"""Controlled surface variation with a closed, independently parsed fact grammar."""
from collections import Counter
from datetime import datetime
import re
from .rules import HEADINGS, VERSION, rule_pack
from .scenarios import verify_scenario
from .storage import digest

LAYOUT_COUNT = 9
ALIASES = {
    'patient_id': ('Patient identifier', 'Patient ID', 'Subject', 'Pt ID', 'Patient ref.', 'Patlent ID'),
    'claim_id': ('Claim identifier', 'Claim ID', 'File', 'Clm ref.', 'Claim reference', 'C l a i m  I D'),
    'service_date': ('Service date', 'Date of service', 'Encounter date', 'Svc. dt.', 'Service dt', 'S e r v i c e  d a t e'),
    'charges': ('Charges', 'Line items', 'Item amounts', 'Charge lines', 'Charges (USD)', 'Itemised charges'),
    'total': ('Total', 'Invoice total', 'Total amount', 'Amt total', 'Total (USD)', 'T o t a l'),
}
REVERSE = {label: key for key, labels in ALIASES.items() for label in labels}
SIGNATURE = {True: 'present', False: 'absent', None: 'unknown'}
HEADING_TYPES = {v: k for k, v in HEADINGS.items()}
HEADING_TYPES.update({v.upper(): k for k, v in HEADINGS.items()})
HEADING_TYPES.update({'D/C summary': 'discharge_summary', 'Diagnostic findings report': 'investigation_report'})
DECORATIONS = {'Synthetic fictional training document', '[Identifiers]', '[Document details]', '[Attestation]',
               '| Field | Value |', '| --- | --- |', 'Processing desk: SYNTHETIC ADMIN ONLY'}


def format_value(key, value, layout):
    if key == 'charges':
        return ', '.join(value)
    if key == 'service_date':
        day = datetime.strptime(value, '%Y-%m-%d')
        return day.strftime(('%Y-%m-%d', '%d/%m/%Y', '%Y-%m-%d', '%b %d, %Y',
                             '%Y/%m/%d', '%Y-%m-%d', '%d %b %Y', '%Y-%m-%d', '%Y-%m-%d')[layout])
    return str(value)


def render_document(d, layout=0):
    if type(layout) is not int or not 0 <= layout < LAYOUT_COUNT:
        raise ValueError('Unsupported rendering style')
    if not d['present']:
        return None
    heading = HEADINGS[d['type']]
    if layout in (3, 6):
        heading = heading.upper()
    if layout == 7 and d['type'] == 'discharge_summary':
        heading = 'D/C summary'
    lines = [heading, 'Synthetic fictional training document']
    if d['readability'] != 'readable':
        lines.append('Contents: ' + d['readability'])
    else:
        entries = list(d['fields'].items())
        if layout in (2, 5, 8):
            entries.reverse()
        if layout == 3:
            lines.append('[Document details]')
        if layout == 4:
            lines += ['| Field | Value |', '| --- | --- |']
        for key, value in entries:
            label = ALIASES[key][layout % len(ALIASES[key])]
            value = format_value(key, value, layout)
            if layout == 4:
                lines.append(f'| {label} | {value} |')
            elif layout == 6:
                lines.append(f'{label} = {value}')
            elif layout == 7:
                lines.append(f'{label} records [{value}].')
            else:
                lines.append(f'{label}: {value}')
        if layout == 3:
            lines.append('[Attestation]')
        lines.append('Signature: ' + SIGNATURE[d['signature']])
        if layout in (5, 8):
            lines.append('Processing desk: SYNTHETIC ADMIN ONLY')
    return {'document_id': d['id'], 'page_number': 1, 'text': '\n'.join(lines)}


def line_fact(line):
    """Return a canonical field/value for a complete controlled source statement."""
    if line in DECORATIONS:
        return None
    if line.startswith('| ') and line.endswith(' |'):
        parts = line[2:-2].split(' | ')
        if len(parts) != 2:
            raise ValueError('Malformed source table')
        label, value = parts
    elif ' records [' in line and line.endswith('].'):
        label, value = line[:-2].split(' records [', 1)
    else:
        separator = ' = ' if ' = ' in line else ': '
        label, sep, value = line.partition(separator)
        if not sep:
            raise ValueError('Unverified source statement')
    key = REVERSE.get(label, label)
    if key == 'Signature':
        if value not in SIGNATURE.values():
            raise ValueError('Unverifiable signature')
        return 'signature', next(k for k, v in SIGNATURE.items() if v == value)
    if key == 'Contents':
        if value not in ('unreadable', 'unknown'):
            raise ValueError('Unknown content state')
        return 'readability', value
    if label not in REVERSE:
        raise ValueError('Unverified additional statement')
    if key == 'service_date':
        for form in ('%Y-%m-%d', '%d/%m/%Y', '%b %d, %Y', '%Y/%m/%d', '%d %b %Y'):
            try:
                value = datetime.strptime(value, form).strftime('%Y-%m-%d')
                break
            except ValueError:
                continue
        else:
            raise ValueError('Unsupported source date')
    return key, value.split(', ') if key == 'charges' else value


def text_units(page):
    units, offset = [], 0
    for i, line in enumerate(page['text'].splitlines(keepends=True)):
        quote = line.rstrip('\n')
        fact = ('type', HEADING_TYPES.get(quote)) if i == 0 else line_fact(quote)
        if fact is not None:
            units.append({'key': fact[0], 'value': fact[1], 'span': {
                'document_id': page['document_id'], 'page_number': page['page_number'],
                'start': offset, 'end': offset + len(quote), 'quote': quote}})
        offset += len(line)
    return units


def parse_text(text):
    lines = text.splitlines()
    if len(lines) < 3 or lines[0] not in HEADING_TYPES or lines[1] != 'Synthetic fictional training document':
        raise ValueError('Unrecognized document heading or unverified narrative')
    fields, signature, readability, seen = {}, None, 'readable', set()
    for line in lines[2:]:
        fact = line_fact(line)
        if fact is None:
            continue
        key, value = fact
        if key in seen:
            raise ValueError('Duplicate or conflicting text field')
        seen.add(key)
        if key == 'signature':
            signature = value
        elif key == 'readability':
            readability = value
        else:
            fields[key] = value
    if readability == 'readable' and 'signature' not in seen:
        raise ValueError('Signature condition not stated')
    if readability != 'readable' and (fields or 'signature' in seen):
        raise ValueError('Unreadable evidence cannot also supply legible fields')
    return {'type': HEADING_TYPES[lines[0]], 'fields': fields, 'signature': signature, 'readability': readability}


def render(s):
    verify_scenario(s)
    pages = [render_document(d, s['layout']) for d in s['documents'] if d['present']]
    return {'scenario_hash': s['content_hash'], 'pages': pages, 'renderer': 'python-v2', 'render_hash': digest(pages)}


def verify(s, rendered):
    verify_scenario(s)
    if rendered['scenario_hash'] != s['content_hash'] or digest(rendered['pages']) != rendered['render_hash']:
        raise ValueError('Rendered identity changed')
    expected = {d['id']: d for d in s['documents'] if d['present']}
    if Counter(p['document_id'] for p in rendered['pages']) != Counter(expected.keys()):
        raise ValueError('Rendered document presence/identity mismatch')
    evidence = []
    for p in rendered['pages']:
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
    inventory = [{'document_id': d['id'], 'document_type': d['type'] if not d['present'] else None,
                  'present': d['present']} for d in s['documents']]
    return {'synthetic': True, 'claim_category': s['category'], 'claim_facts': s['facts'],
            'inventory': inventory, 'documents': rendered['pages'], 'fictional_rule_pack': rule_pack()}
