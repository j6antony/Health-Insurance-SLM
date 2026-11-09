"""Strict contract and evidence validation. Semantic checks are separate silver review."""
import json
import re
from collections import Counter


class ValidationError(ValueError):
    pass


def fail(message):
    raise ValidationError(message)


def validate_schema(value, schema, path='$'):
    supported = {'type', 'properties', 'required', 'additionalProperties', 'items', 'enum',
                 'minLength', 'maxLength', 'minimum', 'maximum', 'minItems', 'maxItems', 'description', 'title'}
    if set(schema) - supported:
        fail(f'{path}: unsupported schema keyword')
    kind = schema.get('type')
    matches = {'object': isinstance(value, dict), 'array': isinstance(value, list),
               'string': isinstance(value, str), 'boolean': type(value) is bool,
               'integer': type(value) is int, 'number': type(value) in (int, float), None: True}
    if kind not in matches or not matches[kind]:
        fail(f'{path}: expected {kind}')
    if 'enum' in schema and value not in schema['enum']:
        fail(f'{path}: invalid enum')
    if kind == 'object':
        if set(schema.get('required', [])) - set(value):
            fail(f'{path}: missing required keys')
        properties = schema.get('properties', {})
        if schema.get('additionalProperties') is False and set(value) - set(properties):
            fail(f'{path}: unexpected keys')
        for key, item in value.items():
            if key in properties:
                validate_schema(item, properties[key], f'{path}.{key}')
    elif kind == 'array':
        if not schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', float('inf')):
            fail(f'{path}: wrong array length')
        for i, item in enumerate(value):
            validate_schema(item, schema.get('items', {}), f'{path}[{i}]')
    elif kind == 'string':
        if not schema.get('minLength', 0) <= len(value) <= schema.get('maxLength', float('inf')):
            fail(f'{path}: invalid string length')
    elif kind in ('integer', 'number'):
        if not schema.get('minimum', -float('inf')) <= value <= schema.get('maximum', float('inf')):
            fail(f'{path}: number outside bounds')


CASE_TYPES = ['supported', 'no_issue', 'insufficient_evidence', 'ambiguous', 'unreadable',
              'contradiction', 'conditional', 'date_mismatch', 'prompt_injection', 'unsupported_request']


def scenario_schema(tasks):
    string = {'type': 'string', 'minLength': 1}
    properties = {
        'generation_id': string, 'task': {'type': 'string', 'enum': list(tasks)},
        'difficulty': {'type': 'string', 'enum': ['easy', 'medium', 'hard']},
        'instruction': string,
        'scenario': {'type': 'object', 'required': ['synthetic', 'documents', 'context'],
                     'additionalProperties': False, 'properties': {
                         'synthetic': {'type': 'boolean', 'enum': [True]},
                         'documents': {'type': 'array', 'minItems': 1, 'items': {
                             'type': 'object', 'required': ['document_id', 'page_number', 'text'],
                             'additionalProperties': False, 'properties': {
                                 'document_id': string, 'page_number': {'type': 'integer', 'minimum': 1}, 'text': string}}},
                         'context': {'type': 'object'}}},
        'response_schema_key': {'type': 'string', 'enum': list(tasks)},
        'skills': {'type': 'array', 'minItems': 1, 'items': string},
        'case_type': {'type': 'string', 'enum': CASE_TYPES},
        'source_mode': {'type': 'string', 'enum': ['synthetic_rules', 'approved_references', 'mixed']},
        'reference_ids': {'type': 'array', 'items': string},
    }
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def batch_schema(tasks, count):
    return {'type': 'array', 'items': scenario_schema(tasks), 'minItems': count, 'maxItems': count}


def check_hidden_fields(value):
    forbidden = {'answer', 'output', 'expected_answer', 'answer_key', 'hidden_answer', 'rationale',
                 'solution', 'reference_answer', 'expected_conclusion', 'completion',
                 'expected_output', 'proposed_answer', 'case_type', 'skills', 'difficulty'}
    if isinstance(value, dict):
        if set(value) & forbidden:
            fail('Hidden answer or generation fields in target-facing context')
        for item in value.values():
            check_hidden_fields(item)
    elif isinstance(value, list):
        for item in value:
            check_hidden_fields(item)


def validate_scenario(record, tasks, references):
    validate_schema(record, scenario_schema(tasks))
    if record['response_schema_key'] != record['task']:
        fail('response_schema_key must equal task')
    documents = record['scenario']['documents']
    ids = [(d['document_id'], d['page_number']) for d in documents]
    if len(ids) != len(set(ids)) or any(not d['text'].strip() for d in documents):
        fail('Duplicate document/page pairs or empty source text')
    check_hidden_fields(record['scenario']['context'])
    lookup = {r['id']: r for r in references}
    reference_ids = record['reference_ids']
    if len(reference_ids) != len(set(reference_ids)) or set(reference_ids) - set(lookup):
        fail('Invalid reference IDs')
    if record['source_mode'] == 'synthetic_rules' and reference_ids:
        fail('Synthetic-only record cites approved references')
    if record['source_mode'] != 'synthetic_rules' and not reference_ids:
        fail('Approved-reference source mode needs references')
    serialized = json.dumps(record['scenario'], ensure_ascii=False)
    for key in reference_ids:
        source = lookup[key]
        if source['text'] not in '\n'.join(d['text'] for d in documents):
            fail('Approved reference excerpt missing or modified')
        for field in ('version', 'jurisdiction', 'effective_from', 'effective_to'):
            if source.get(field) and str(source[field]) not in serialized:
                fail(f'Approved reference missing applicability metadata: {field}')


def validate_batch(records, quotas, difficulty, tasks, references):
    validate_schema(records, batch_schema(tasks, sum(quotas.values())))
    if Counter(r['task'] for r in records) != +Counter(quotas):
        fail('Task quota mismatch')
    if Counter(r['difficulty'] for r in records) != +Counter(difficulty):
        fail('Difficulty quota mismatch')
    ids = [r['generation_id'] for r in records]
    if len(ids) != len(set(ids)):
        fail('Duplicate generation IDs')
    for record in records:
        validate_scenario(record, tasks, references)


def validate_answer(record, answer, schema):
    validate_schema(answer, schema)
    pages = record['scenario']['documents']
    docs = {}
    for page in pages:
        docs.setdefault(page['document_id'], []).append(page)
    for evidence in answer.get('evidence', []):
        if isinstance(evidence, str):
            if not evidence.strip() or not any(evidence in d['text'] for d in pages):
                fail('Classification evidence quote not in source')
        else:
            doc_id = evidence['document_id']
            if doc_id not in docs:
                fail(f'Evidence document ID {doc_id!r} not found; use one of {sorted(docs)}')
            if not evidence['quote'].strip() or not any(evidence['quote'] in d['text'] for d in docs[doc_id]):
                fail(f'Evidence quote for {doc_id!r} must be a nonempty exact contiguous substring of its source text')
    fields = answer.get('fields', [])
    names = [f['field_name'] for f in fields]
    if len(names) != len(set(names)):
        fail('Duplicate extracted fields: emit at most one entry per field_name; mark a directly supported source value requires_review=true when conflicting')
    target_id = record['scenario']['context'].get('target_document_id')
    for field in fields:
        matches = [d for d in pages if field['evidence'] in d['text'] and d['page_number'] == field['page_number']]
        if target_id:
            matches = [d for d in matches if d['document_id'] == target_id]
        if not field['evidence'].strip() or not matches:
            fail('Extraction evidence/page/document mismatch')
        # Values may be normalized, but identifiers, names and amounts must remain
        # traceable to the quoted source; allow removal of thousands separators.
        normalize = lambda s: re.sub(r'[^\w]', '', s.casefold())
        if normalize(field['value']) not in normalize(field['evidence']):
            fail('Extracted value unsupported by its exact quote')
    if answer.get('missing_information') and answer.get('requires_review') is not True:
        fail('Missing information requires review flag')
    all_source = '\n'.join(d['text'] for d in pages)
    if 'suggested_codes' in answer:
        for code in answer['suggested_codes']:
            if not re.search(r'(?<!\w)' + re.escape(code) + r'(?!\w)', all_source):
                fail('Suggested code absent from source catalog')
    # Optional machine-checkable source arithmetic assertions; the target also
    # sees these constraints, so they are evidence, never hidden expected answers.
    for equation in record['scenario']['context'].get('numeric_constraints', []):
        if set(equation) != {'operands', 'operation', 'claimed_result'}:
            fail('Invalid numeric constraint')
        from decimal import Decimal, InvalidOperation
        try:
            values = [Decimal(str(v)) for v in equation['operands']]
            if not values:
                fail('Empty numeric operands')
            if equation['operation'] == 'sum':
                result = sum(values)
            elif equation['operation'] == 'subtract':
                result = values[0] - sum(values[1:])
            else:
                fail('Unsupported numeric operation')
            if result != Decimal(str(equation['claimed_result'])):
                fail('Contradictory numeric source constraint')
        except (InvalidOperation, TypeError):
            fail('Malformed numeric constraint')


AUDIT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['grounded', 'calculations_checked', 'uncertainty_handled', 'unsupported_actions', 'safe_for_synthetic_sft', 'reasons'],
    'properties': {**{key: {'type': 'boolean'} for key in (
        'grounded', 'calculations_checked', 'uncertainty_handled', 'unsupported_actions', 'safe_for_synthetic_sft')},
        'reasons': {'type': 'array', 'items': {'type': 'string'}}},
}


def validate_audit(audit):
    validate_schema(audit, AUDIT_SCHEMA)
    if not all(audit[k] for k in ('grounded', 'calculations_checked', 'uncertainty_handled', 'safe_for_synthetic_sft')) or audit['unsupported_actions']:
        fail('Answer failed automated grounding audit')


def scenario_audit_schema(ids):
    return {'type': 'array', 'minItems': len(ids), 'maxItems': len(ids), 'items': {
        'type': 'object', 'additionalProperties': False,
        'required': ['generation_id', 'usable', 'coverage_dimensions', 'duplicate_of', 'reasons'],
        'properties': {
            'generation_id': {'type': 'string', 'enum': list(ids)},
            'usable': {'type': 'boolean'},
            'duplicate_of': {'type': 'string'},
            'coverage_dimensions': {'type': 'array', 'items': {'type': 'string', 'enum': [
                'supported_or_no_issue', 'insufficient_ambiguous_unreadable', 'conflicting_or_misleading',
                'conditional_logic_or_date', 'injection_or_unsupported_action']}},
            'reasons': {'type': 'array', 'items': {'type': 'string'}},
        }}}
