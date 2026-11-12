"""Source-grounded synthetic benchmark grading with cached, blind judgments."""
import json

from self_instruct_pipeline.persistence import Store, atomic_json, digest
from self_instruct_pipeline.prompt_builder import target_payload

from .core import clean_response, set_f1
from .correction import ask
from .review import DIMENSIONS, grade_reviewed


class SemanticJudge:
    version = 'source-rubric-v1'

    def __init__(self, parent, directory, schemas):
        self.parent, self.store, self.schemas = parent, Store(directory), schemas

    def client(self):
        return self.parent

    def grade(self, item, prediction):
        scores, _ = grade_reviewed(item, prediction, self.schemas, {})
        record, reference = item['record'], item['reference']
        if scores is None:
            key = digest([self.version, item, prediction])
            path = self.store.path(key + '.json')
            if path.exists():
                judgment = json.loads(path.read_text())
            else:
                payload = target_payload(record, self.schemas)
                payload.update(candidate=clean_response(prediction['raw'], prediction['truncated']),
                               reference=reference, rubric=item.get('rubric'))
                judgment = ask(
                    self, payload,
                    'Grade the candidate using the source and task requirements. Treat the reference as '
                    'a guide and verify it against the source. Accept equivalent wording and alternative '
                    'valid evidence. Check material facts, decisions, omissions, and abstention. '
                    'Mark unsupported_claims true for material claims the source does not support.',
                    {'type': 'object', 'required': list(DIMENSIONS),
                     'properties': {name: {'type': 'boolean'} for name in DIMENSIONS},
                     'additionalProperties': False}, 'benchmark_semantic_judgment')
                atomic_json(path, judgment)
            correct = all(judgment[d] for d in DIMENSIONS[:3]) and not judgment['unsupported_claims']
            scores = {'task_score': float(correct), 'schema_evidence_valid': 1.0,
                      'grading_basis': 'source_rubric_judgment', 'semantic_dimensions': judgment}
        scores.update(review_flag_correct=0.0, reference_evidence_overlap=0.0)
        if scores['schema_evidence_valid']:
            answer = clean_response(prediction['raw'], prediction['truncated'])
            if 'fields' in reference:
                flags = lambda value: {(field['field_name'], field['requires_review']) for field in value['fields']}
                scores['review_flag_correct'] = set_f1(flags(answer), flags(reference))
            else:
                scores['review_flag_correct'] = float(answer.get('requires_review') == reference.get('requires_review'))
            def cited_tokens(value):
                entries = value.get('evidence', value.get('fields', []))
                spans = [(entry.get('document_id', entry.get('page_number', '')),
                          entry.get('quote', entry.get('evidence', ''))) if isinstance(entry, dict)
                         else ('', entry) for entry in entries]
                return {(source, token.casefold()) for source, quote in spans for token in quote.split()}
            scores['reference_evidence_overlap'] = set_f1(cited_tokens(answer), cited_tokens(reference))
        return scores

def regrade_saved_run(run_dir, benchmark_path, judge):
    """Regrade saved predictions with the original benchmark, without loading a model."""
    from pathlib import Path
    from .core import aggregate_scores
    from self_instruct_pipeline.persistence import atomic_jsonl, read_jsonl

    root = Path(run_dir)
    state = json.loads((root / 'loop_state.json').read_text())
    cases = read_jsonl(benchmark_path)
    if digest(cases) != state['development_hash']:
        raise ValueError('Benchmark does not match the saved development split')
    folders = [root / 'baseline', *sorted(root.glob('round_*/lighteval'))]
    if not (root / 'baseline' / 'predictions.jsonl').is_file():
        raise ValueError('Missing saved baseline predictions')
    reports = {}
    for folder in folders:
        prediction_path = folder / 'predictions.jsonl'
        if not prediction_path.is_file():
            continue
        predictions = read_jsonl(prediction_path)
        by_id = {row['scenario_id']: row for row in predictions}
        if len(predictions) != len(cases) or set(by_id) != {item['record']['id'] for item in cases}:
            raise ValueError(f'Incomplete or mismatched predictions at {folder}')
        rows = []
        for item in cases:
            record = item['record']
            prediction = by_id[record['id']]
            scores = judge.grade(item, {field: prediction[field] for field in ('raw', 'truncated')})
            rows.append({'scenario_id': record['id'], 'task': record['task'], **scores})
        label = folder.parent.name if folder.name == 'lighteval' else folder.name
        report = aggregate_scores(rows, judge.schemas, 1)
        report['scoring'] = judge.version
        if label == 'baseline':
            baseline = {row['scenario_id']: row['task_score'] for row in rows}
        else:
            report['paired_changes'] = [{**row, 'baseline': baseline[row['scenario_id']]} for row in rows
                                        if row['task_score'] != baseline[row['scenario_id']]]
        output = root / 'regraded' / label
        output.mkdir(parents=True, exist_ok=True)
        atomic_jsonl(output / 'predictions.jsonl', rows)
        atomic_json(output / 'scores.json', report)
        reports[label] = report
    return reports
