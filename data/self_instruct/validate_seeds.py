"""Validate this pack's schema subset and source fidelity, not domain correctness."""
import collections
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent


def check(value, schema, path):
    kind = schema.get('type')
    valid = {
        'object': isinstance(value, dict),
        'array': isinstance(value, list),
        'string': isinstance(value, str),
        'boolean': type(value) is bool,
        'integer': type(value) is int,
        'number': type(value) in (int, float),
        None: True,
    }
    assert kind in valid and valid[kind], f'{path}: wrong type {kind}'
    supported = {'type', 'properties', 'required', 'additionalProperties', 'items',
                 'enum', 'minLength', 'maxLength', 'minimum', 'maximum',
                 'minItems', 'maxItems', 'description', 'title'}
    assert not set(schema) - supported, f'{path}: unsupported schema keywords'
    if 'enum' in schema:
        assert value in schema['enum'], f'{path}: invalid enum'
    if kind == 'object':
        assert set(schema.get('required', [])) <= set(value), f'{path}: missing keys'
        props = schema.get('properties', {})
        if schema.get('additionalProperties') is False:
            assert set(value) <= set(props), f'{path}: extra keys'
        for key, item in value.items():
            if key in props:
                check(item, props[key], f'{path}.{key}')
    if kind == 'array':
        assert len(value) >= schema.get('minItems', 0), f'{path}: too few items'
        assert len(value) <= schema.get('maxItems', float('inf')), f'{path}: too many items'
        for i, item in enumerate(value):
            check(item, schema.get('items', {}), f'{path}[{i}]')
    if kind == 'string':
        assert len(value) >= schema.get('minLength', 0), f'{path}: short string'
        assert len(value) <= schema.get('maxLength', float('inf')), f'{path}: long string'
    if kind in ('integer', 'number'):
        assert value >= schema.get('minimum', -float('inf')), f'{path}: below minimum'
        assert value <= schema.get('maximum', float('inf')), f'{path}: above maximum'


def main():
    schemas = json.loads((ROOT / 'task_schemas.json').read_text())['tasks']
    rich = [json.loads(line) for line in (ROOT / 'seed_tasks.annotated.jsonl').read_text().splitlines()]
    core = [json.loads(line) for line in (ROOT / 'seed_tasks.jsonl').read_text().splitlines()]
    assert len(core) == len(rich) == 72
    assert {row['task'] for row in rich} == set(schemas)
    assert {row['task'] for row in core} == set(schemas)
    assert len({row['id'] for row in rich}) == len(rich)
    assert len({row['name'] for row in rich}) == len(rich)
    assert len({row['instruction'] for row in rich}) == len(rich)
    seen = set()
    for row, export in zip(rich, core):
        tag = row['id']
        check(row['output'], schemas[row['task']], tag)
        assert row['input']['synthetic'] is True
        assert row['review_status'] == 'ai_drafted_unreviewed'
        assert set(export) == {'id', 'name', 'task', 'instruction', 'instances', 'is_classification'}
        assert export['id'] == tag and export['name'] == row['name']
        assert export['task'] == row['task'], f'{tag}: task labels disagree'
        assert export['task'] in schemas, f'{tag}: unsupported task'
        assert export['instruction'] == row['instruction']
        assert export['is_classification'] == (row['task'] == 'document_classification')
        assert len(export['instances']) == 1
        instance = export['instances'][0]
        assert json.loads(instance['output']) == row['output']
        input_text, schema_text = instance['input'].split('\nResponse schema:\n')
        assert json.loads(input_text) == row['input']
        assert json.loads(schema_text) == schemas[row['task']]
        fingerprint = hashlib.sha256(input_text.encode()).hexdigest()
        assert fingerprint not in seen, f'{tag}: duplicate input'
        seen.add(fingerprint)
        docs = {d['document_id']: d for d in row['input']['documents']}
        assert len(docs) == len(row['input']['documents'])
        for evidence in row['output'].get('evidence', []):
            if isinstance(evidence, str):
                assert any(evidence in d['text'] for d in docs.values()), f'{tag}: missing quote'
            else:
                assert evidence['document_id'] in docs, f'{tag}: invalid document'
                assert evidence['quote'] in docs[evidence['document_id']]['text'], f'{tag}: missing quote'
        for field in row['output'].get('fields', []):
            matches = [d for d in docs.values() if field['evidence'] in d['text']]
            assert matches, f'{tag}: extraction quote missing'
            if 'target_document_id' in row['input']:
                target = docs[row['input']['target_document_id']]
                assert field['evidence'] in target['text'], f'{tag}: wrong document'
            page_match = False
            for doc in matches:
                marked_pages = re.split(r'\[Page (\d+)\]', doc['text'])
                if len(marked_pages) > 1:
                    for i in range(1, len(marked_pages), 2):
                        if int(marked_pages[i]) == field['page_number'] and field['evidence'] in marked_pages[i+1]:
                            page_match = True
                elif field['page_number'] == doc['page_number']:
                    page_match = True
            assert page_match, f'{tag}: wrong source page'
    report = {
        'status': 'passed',
        'seed_count': len(rich),
        'task_count': len(schemas),
        'counts_by_task': dict(collections.Counter(r['task'] for r in rich)),
        'counts_by_case_type': dict(collections.Counter(r['case_type'] for r in rich)),
        'counts_by_difficulty': dict(collections.Counter(r['difficulty'] for r in rich)),
        'checks': ['JSON parsing', 'unique IDs/names/instructions/inputs', '18-task coverage',
                   'explicit supported task label on every record in both exports',
                   'supported response-schema constraints', 'core/annotated parity',
                   'exact source quotes and document references', 'extraction page attribution'],
        'not_verified': ['independent domain correctness', 'clinical/legal validity',
                         'semantic novelty', 'legacy repository runtime compatibility',
                         'model quality or performance'],
        'sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                   for name in ('seed_tasks.jsonl', 'seed_tasks.annotated.jsonl', 'task_schemas.json')},
    }
    (ROOT / 'validation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"PASS: {len(rich)} seeds, {len(schemas)} tasks; structural/schema-subset, parity, and evidence checks.")


if __name__ == '__main__':
    main()
