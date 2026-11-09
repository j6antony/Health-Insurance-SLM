import json
from .persistence import atomic_jsonl, digest
from .prompt_builder import target_payload
from .validation import validate_answer, validate_audit


def export_dataset(store, schemas):
    scenarios = store.unique_records('accepted_scenarios.jsonl', 'id')
    answers = store.unique_records('validated_answers.jsonl', 'scenario_id')
    by_id = {r['id']: r for r in scenarios}
    rows, chats, metadata = [], [], []
    for record in answers:
        if record.get('verification_status') != 'automatically_checked_silver':
            continue
        scenario = by_id.get(record['scenario_id'])
        if scenario is None or record.get('scenario_hash') != digest(scenario):
            raise ValueError('Answer has no matching unchanged scenario')
        validate_answer(scenario, record['answer'], schemas[scenario['task']])
        validate_audit(record['audit'])
        payload = target_payload(scenario, schemas)
        output = json.dumps(record['answer'], ensure_ascii=False)
        rows.append({'task': scenario['task'], 'instruction': payload['instruction'],
                     'input': json.dumps({'scenario': payload['scenario'], 'response_schema': payload['response_schema']}, ensure_ascii=False),
                     'output': output})
        chats.append({'messages': [{'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)},
                                   {'role': 'assistant', 'content': output}]})
        metadata.append({'row': len(rows) - 1, 'scenario_id': scenario['id'], 'task': scenario['task'],
                         'verification_status': record['verification_status'], 'parent': record['parent'],
                         'scenario_hash': record['scenario_hash']})
    atomic_jsonl(store.path('qwen3_8b_sft.jsonl'), rows)
    atomic_jsonl(store.path('qwen3_8b_chat.jsonl'), chats)
    atomic_jsonl(store.path('export_metadata.jsonl'), metadata)
    return len(rows)
