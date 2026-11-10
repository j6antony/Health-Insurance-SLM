import json
import math
import re
from collections import Counter
from pathlib import Path

from self_instruct_pipeline.config import resolve_path
from self_instruct_pipeline.deduplicator import rouge_l
from self_instruct_pipeline.validation import ValidationError, validate_answer


class ResponseError(ValueError):
    pass


def clean_response(raw, truncated=False):
    """Remove only complete wrappers; never repair JSON or invent missing fields."""
    if truncated:
        raise ResponseError('Student generation hit the token limit without EOS')
    if not isinstance(raw, str) or not raw.strip():
        raise ResponseError('Empty student response')
    text = raw.strip()
    if text.startswith('<think>'):
        end = text.find('</think>')
        if end < 0:
            raise ResponseError('Unclosed thinking block')
        text = text[end + len('</think>'):].strip()
    if '<think>' in text or '</think>' in text:
        raise ResponseError('Unexpected thinking markup')
    fenced = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        def unique_keys(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ResponseError('Duplicate JSON keys in student response')
                result[key] = item
            return result
        value = json.loads(text, object_pairs_hook=unique_keys)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ResponseError('Student response is not a complete JSON object') from exc
    if not isinstance(value, dict):
        raise ResponseError('Student response must be a JSON object')
    return value


def normalize(value):
    return ' '.join(str(value).casefold().split())


def set_f1(predicted, reference):
    predicted, reference = set(predicted), set(reference)
    if not predicted and not reference:
        return 1.0
    return 2 * len(predicted & reference) / max(1, len(predicted) + len(reference))


def score_response(record, reference, raw, schemas, truncated=False):
    """Conservative task-aware reference metric with schema/evidence gates."""
    try:
        answer = clean_response(raw, truncated)
        validate_answer(record, answer, schemas[record['task']])
    except (ResponseError, ValidationError, KeyError, TypeError):
        return {'task_score': 0.0, 'schema_evidence_valid': 0.0}
    if 'document_type' in reference:
        score = float(answer['document_type'] == reference['document_type'])
    elif answer.get('requires_review') != reference.get('requires_review'):
        score = 0.0
    elif 'fields' in reference:
        key = lambda field: (field['field_name'], normalize(field['value']), field['requires_review'])
        score = set_f1(map(key, answer['fields']), map(key, reference['fields']))
    elif answer.get('requires_review') != reference.get('requires_review'):
        score = 0.0
    elif 'suggested_codes' in reference:
        score = set_f1(answer['suggested_codes'], reference['suggested_codes'])
    else:
        text = lambda value: '\n'.join([value['answer'], value.get('draft', ''),
                                       '\n'.join(value.get('criteria_assessment', []))])
        # Exact normalized reference agreement is deliberately conservative.
        # Lexical overlap must never reward a reversed conclusion.
        score = float(normalize(text(answer)) == normalize(text(reference)))
        # These gates preserve important no-issue / missing-information distinctions
        # while allowing the natural-language wording to differ.
        for field in ('issues', 'missing_information'):
            if {normalize(x) for x in answer.get(field, [])} != {normalize(x) for x in reference.get(field, [])}:
                score = 0.0
    return {'task_score': float(score), 'schema_evidence_valid': 1.0}


def aggregate_scores(rows, tasks, minimum_examples):
    grouped = {task: [] for task in tasks}
    for row in rows:
        if row['task'] not in grouped:
            raise ValueError('Unknown evaluation task')
        for field in ('task_score', 'schema_evidence_valid'):
            if type(row[field]) not in (int, float) or not math.isfinite(row[field]) or not 0 <= row[field] <= 1:
                raise ValueError('Evaluation scores must be finite values in [0, 1]')
        grouped[row['task']].append(row)
    if any(len(samples) < minimum_examples for samples in grouped.values()):
        raise ValueError('Incomplete evaluation: insufficient examples for one or more tasks')
    scores = {task: sum(r['task_score'] for r in samples) / len(samples) for task, samples in grouped.items()}
    return {'macro_task_score': sum(scores.values()) / len(scores), 'per_task': scores,
            'minimum_task_score': min(scores.values()),
            'schema_evidence_valid_rate': sum(r['schema_evidence_valid'] for r in rows) / len(rows),
            'examples_per_task': {task: len(samples) for task, samples in grouped.items()}}


def satisfactory(report, thresholds, tasks):
    if set(report.get('per_task', {})) != set(tasks):
        return False
    numbers = [report.get('macro_task_score'), report.get('minimum_task_score'),
               report.get('schema_evidence_valid_rate'), *report['per_task'].values()]
    if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in numbers):
        return False
    # Recompute aggregates so an inconsistent summary cannot bypass the gate.
    per_task = list(report['per_task'].values())
    return (sum(per_task) / len(per_task) >= thresholds['macro_task_score']
            and min(per_task) >= thresholds['minimum_task_score']
            and report['schema_evidence_valid_rate'] >= thresholds['schema_evidence_valid_rate'])


def feedback_weights(base_weights, report, tasks, minimum_score, strength):
    return {task: base_weights.get(task, 1) * (1 + strength * max(0, minimum_score - report['per_task'][task]))
            for task in tasks}


def load_loop_config(path='config/distillation_loop.json'):
    value = json.loads(resolve_path(path).read_text())
    for key in ('max_rounds', 'scenarios_per_round', 'development_examples_per_task', 'development_fill_attempts'):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if set(value['thresholds']) != {'macro_task_score', 'minimum_task_score', 'schema_evidence_valid_rate'}:
        raise ValueError('Invalid threshold names')
    for threshold in value['thresholds'].values():
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError('Thresholds must be finite in (0, 1]')
    if type(value['feedback_strength']) not in (int, float) or not math.isfinite(value['feedback_strength']) or value['feedback_strength'] < 0:
        raise ValueError('feedback_strength must be nonnegative and finite')
    evaluation = value.get('evaluation', {'mode': 'synthetic_smoke'})
    if evaluation.get('mode') not in ('reviewed', 'synthetic_smoke'):
        raise ValueError('Invalid evaluation mode')
    if evaluation['mode'] == 'reviewed':
        minimum = evaluation.get('minimum_test_examples_per_task')
        if type(minimum) is not int or minimum < 30 or value['development_examples_per_task'] < 30:
            raise ValueError('Reviewed evaluation requires at least 30 examples per task per split')
        for split in ('validation', 'test'):
            if not isinstance(evaluation.get(f'{split}_path'), str) or not evaluation[f'{split}_path']:
                raise ValueError('Reviewed evaluation requires benchmark paths')
    t = value['training']
    for key in ('max_sequence_length','epochs_per_round','max_steps_per_round','gradient_accumulation_steps','lora_rank','lora_alpha'):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(t['learning_rate']) not in (int, float) or not 0 < t['learning_rate'] < 1:
        raise ValueError('Invalid learning rate')
    if type(t['lora_dropout']) not in (int, float) or not 0 <= t['lora_dropout'] < 1:
        raise ValueError('Invalid LoRA dropout')
    if type(t['replay_previous_rounds']) is not bool:
        raise ValueError('replay_previous_rounds must be boolean')
    n = value['inference']['max_new_tokens']
    if type(n) is not int or not 0 < n < t['max_sequence_length']:
        raise ValueError('Completion budget must fit within the sequence length')
    if type(value['random_seed']) is not int or value['random_seed'] < 0:
        raise ValueError('random_seed must be nonnegative')
    return value


def validate_report(report, tasks, minimum_examples):
    if set(report.get('per_task', {})) != set(tasks):
        raise ValueError('Lighteval report is missing tasks')
    values = list(report['per_task'].values())
    for value in [report.get('macro_task_score'), report.get('minimum_task_score'),
                  report.get('schema_evidence_valid_rate'), *values]:
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Lighteval report contains invalid scores')
    if any(report.get('examples_per_task', {}).get(task, 0) < minimum_examples for task in tasks):
        raise ValueError('Lighteval report has insufficient task coverage')
    if (not math.isclose(report['macro_task_score'], sum(values) / len(values), abs_tol=1e-9)
            or not math.isclose(report['minimum_task_score'], min(values), abs_tol=1e-9)):
        raise ValueError('Lighteval summary disagrees with per-task scores')
