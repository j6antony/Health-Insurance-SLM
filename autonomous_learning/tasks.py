"""Versioned task contracts. Student payloads are constructed from an allowlist."""
from . import CONTRACT_VERSION
from .rendering import parse_text, verify, visible_package
from .storage import digest

TASKS = ('document_classification', 'field_extraction', 'missing_documents', 'evidence_quality',
         'conditional_requirements', 'cross_document_consistency', 'finding_explanation')
INSTRUCTIONS = {
    'document_classification': 'Classify each submitted document from its heading; use unknown for unreadable contents only when the heading is also unavailable.',
    'field_extraction': 'Extract all legible supplied fields. Do not infer fields from unreadable documents.',
    'missing_documents': 'List absent required document types under the supplied fictional checklist. Do not treat unreadable evidence as missing.',
    'evidence_quality': 'Report the readability of submitted evidence.',
    'conditional_requirements': 'Apply the fictional referral requirement; preserve unknown applicability.',
    'cross_document_consistency': 'Check service-date consistency using legible evidence only.',
    'finding_explanation': 'Give short findings grounded in the supplied fictional checklist and evidence. Do not adjudicate coverage or payment.',
}
STRING = {'type': 'string'}
EVIDENCE = {'type': 'object', 'additionalProperties': False,
            'properties': {'document_id': STRING, 'page_number': {'type': 'integer', 'minimum': 1},
                           'start': {'type': 'integer', 'minimum': 0}, 'end': {'type': 'integer', 'minimum': 0}, 'quote': STRING},
            'required': ['document_id', 'page_number', 'start', 'end', 'quote']}


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def array(item):
    return {'type': 'array', 'items': item}


FINDING = obj({'rule_id': STRING, 'status': {'type': 'string', 'enum': [
    'passed', 'issue_detected', 'requires_review', 'not_applicable', 'not_evaluated']}, 'reason': STRING})
RESULTS = {
    'document_classification': array(obj({'document_id': STRING, 'document_type': STRING})),
    'field_extraction': array(obj({'document_id': STRING, 'field_name': STRING, 'value': STRING})),
    'missing_documents': array(STRING),
    'evidence_quality': array(obj({'document_id': STRING, 'readability': STRING})),
    'conditional_requirements': array(FINDING),
    'cross_document_consistency': array(FINDING),
    'finding_explanation': array(FINDING),
}
SCHEMAS = {task: obj({'result': RESULTS[task], 'evidence': array(EVIDENCE),
                      'source_references': array(STRING), 'requires_review': {'type': 'boolean'}}) for task in TASKS}


def project(s, rendered):
    gate = verify(s, rendered)
    visible = visible_package(s, rendered)
    parsed = [(p['document_id'], parse_text(p['text'])) for p in rendered['pages']]
    findings = s['findings']
    results = {
        'document_classification': [{'document_id': key, 'document_type': p['type']} for key, p in parsed],
        'field_extraction': [{'document_id': key, 'field_name': field, 'value': ', '.join(v) if isinstance(v, list) else v}
                             for key, p in parsed for field, v in p['fields'].items()],
        'missing_documents': [f['input_facts']['document_type'] for f in findings
                              if f['rule_id'].startswith('required.') and f['input_facts']['applicable'] is True
                              and not f['input_facts']['present']],
        'evidence_quality': [{'document_id': key, 'readability': p['readability']} for key, p in parsed],
    }
    selections = {'conditional_requirements': [f for f in findings if f['rule_id'] == 'required.referral'],
                  'cross_document_consistency': [f for f in findings if f['rule_id'].startswith('consistency.')],
                  'finding_explanation': findings}
    for task, selected in selections.items():
        results[task] = [{k: f[k] for k in ('rule_id', 'status', 'reason')} for f in selected]
    rows = []
    for task in TASKS:
        if task in selections:
            review = any(f['requires_review'] for f in selections[task])
        elif task == 'missing_documents':
            review = bool(results[task]) or s['facts']['referral_required'] is None
        elif task in ('field_extraction', 'evidence_quality'):
            review = any(p['readability'] != 'readable' for _, p in parsed)
        else:
            review = False
        payload = {'instruction': INSTRUCTIONS[task], 'scenario': visible, 'response_schema': SCHEMAS[task]}
        target = {'result': results[task], 'evidence': gate['evidence'],
                  'source_references': ['inventory', 'claim_facts', 'fictional_rule_pack'], 'requires_review': review}
        rows.append({'id': digest([s['variant_id'], task]), 'contract_version': CONTRACT_VERSION,
                     'family_id': s['family_id'], 'variant_id': s['variant_id'], 'task': task,
                     'payload': payload, 'target': target, 'scenario_hash': s['content_hash'],
                     'render_hash': rendered['render_hash'], 'verification': 'synthetic_silver',
                     'template_family': 'closed-grammar-v1', 'layout': s['layout'],
                     'rule_ids': s['relevant_rule_ids']})
    return rows


def target_from_visible(row):
    """Independently reconstruct truth from student-visible text and inventory."""
    from .scenarios import seal
    visible = row['payload']['scenario']
    by_id = {p['document_id']: parse_text(p['text']) for p in visible['documents']}
    documents = []
    for item in visible['inventory']:
        key = item['document_id']
        if item['present']:
            parsed = by_id[key]
            documents.append({'id': key, 'present': True, **parsed})
        else:
            if key in by_id:
                raise ValueError('Absent document unexpectedly has text')
            documents.append({'id': key, 'type': item['document_type'], 'present': False,
                              'readability': 'not_available', 'signature': None, 'fields': {}})
    if set(by_id) != {d['id'] for d in documents if d['present']}:
        raise ValueError('Inventory and final text disagree')
    s = seal({'synthetic': True, 'category': visible['claim_category'], 'facts': visible['claim_facts'],
              'rule_pack_version': visible['fictional_rule_pack']['version'], 'documents': documents,
              'family_id': row['family_id'], 'variant_id': row['variant_id'], 'layout': row['layout']})
    r = {'scenario_hash': s['content_hash'], 'pages': visible['documents'], 'render_hash': digest(visible['documents'])}
    return next(example['target'] for example in project(s, r) if example['task'] == row['task'])
