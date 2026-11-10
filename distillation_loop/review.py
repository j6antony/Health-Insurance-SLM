"""Conservative grading with independently reviewed variants and bound adjudications."""
from self_instruct_pipeline.persistence import digest
from .core import clean_response, normalize, score_response
from self_instruct_pipeline.validation import validate_answer, ValidationError

SCORING_VERSION = 'reviewed-v2'
DIMENSIONS = ('decision_correct', 'facts_correct', 'abstention_correct', 'unsupported_claims')


class ReviewRequired(ValueError):
    pass


def semantic_payload(answer):
    # Evidence may use any valid source span. All substantive fields still count.
    unordered = {'fields', 'suggested_codes', 'missing_information', 'issues'}
    prose = {'answer', 'draft', 'criteria_assessment', 'missing_information', 'issues'}
    def canonical(value, field=None):
        if isinstance(value, str):
            return normalize(value) if field in prose else ' '.join(value.split())
        if isinstance(value, list):
            values = [canonical(v, field) for v in value]
            return sorted(values, key=lambda v: str(v)) if field in unordered else values
        if isinstance(value, dict):
            return {k: canonical(v, k) for k, v in value.items() if k not in ('evidence', 'page_number')}
        return value
    return canonical(answer)


def review_key(item, prediction):
    return digest({'benchmark_item': item, 'prediction': prediction, 'scoring_version': SCORING_VERSION})


def grade_reviewed(item, prediction, schemas, judgments):
    key = review_key(item, prediction)
    record = item['record']
    try:
        answer = clean_response(prediction['raw'], prediction['truncated'])
        # Separate schema/evidence validity from independent answer correctness.
        visible = {k: v for k, v in record.items() if k != 'answer_checks'}
        validate_answer(visible, answer, schemas[record['task']])
    except (ValueError, KeyError, TypeError):
        return {'task_score': 0.0, 'schema_evidence_valid': 0.0, 'grading_basis': 'invalid_schema_or_evidence'}, None
    try:
        validate_answer(record, answer, schemas[record['task']])
    except ValidationError:
        return {'task_score': 0.0, 'schema_evidence_valid': 1.0, 'grading_basis': 'failed_deterministic_check'}, None
    variants = [item['reference'], *item.get('accepted_references', [])]
    if any(semantic_payload(answer) == semantic_payload(v) for v in variants):
        return {'task_score': 1.0, 'schema_evidence_valid': 1.0, 'grading_basis': 'reviewed_reference_agreement'}, None
    # Deterministic schemas have no ungraded prose. Other answers need full review:
    # matching codes alone must not approve a contradictory explanation.
    if 'document_type' in answer or 'fields' in answer:
        return {**score_response(record, item['reference'], prediction['raw'], schemas, prediction['truncated']),
                'grading_basis': 'deterministic_fields'}, None
    judgment = judgments.get(key)
    if judgment is not None:
        if (not isinstance(judgment.get('reviewer'), str) or not judgment['reviewer'].strip()
                or not isinstance(judgment.get('reason'), str) or not judgment['reason'].strip()
                or any(type(judgment.get(d)) is not bool for d in DIMENSIONS)):
            raise ValueError('Adjudication requires reviewer, reason, and all Boolean correctness dimensions')
        correct = all(judgment[d] for d in DIMENSIONS[:3]) and not judgment['unsupported_claims']
        return {'task_score': float(correct), 'schema_evidence_valid': 1.0,
                'grading_basis': 'independent_adjudication',
                'correctness_review': {d: judgment[d] for d in DIMENSIONS}}, None
    return None, {'review_key': key, 'record': record, 'reference': item['reference'],
                  'rubric': item.get('rubric'), 'prediction': prediction,
                  'required_dimensions': list(DIMENSIONS)}
