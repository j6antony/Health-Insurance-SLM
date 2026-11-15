"""Compare semantic grading with a frozen independently reviewed development sample."""
import argparse
import json
from collections import Counter
from pathlib import Path

from self_instruct_pipeline.persistence import atomic_json, digest, read_jsonl
from self_instruct_pipeline.validation import validate_schema

from .review import DIMENSIONS
from .semantic import JUDGMENT_SCHEMA, SemanticJudge


def calibrate(report_path, labels_path):
    report = json.loads(report_path.read_text())
    if report['scope'] != 'development synthetic silver' or report['scoring'] != SemanticJudge.version:
        raise ValueError('Calibration requires the saved development semantic report')
    identity = digest(report['identity'])
    labels = read_jsonl(labels_path)
    if not labels:
        raise ValueError('Supply a nonempty, independently reviewed calibration file')
    cases = {}
    for stage in report['runs']:
        for row in read_jsonl(report_path.parent / f'{stage}_semantic_cases.jsonl'):
            cases[stage, row['scenario_id']] = row
    seen = set()
    per_task, per_checkpoint = Counter(), Counter()
    tables = {name: Counter() for name in (*DIMENSIONS, 'pass')}
    bias = []
    disagreements = []
    for label in labels:
        key = label['checkpoint'], label['scenario_id']
        row = cases.get(key)
        if (key in seen or row is None or label.get('comparison_identity_sha256') != identity
                or not isinstance(label.get('prediction_sha256'), str)
                or label['prediction_sha256'] != row.get('prediction_sha256')
                or not isinstance(label.get('reviewer'), str) or not label['reviewer'].strip()):
            raise ValueError('Duplicate, unreviewed, or mismatched development calibration label')
        seen.add(key)
        human = label['judgment']
        validate_schema(human, JUDGMENT_SCHEMA)
        machine = row.get('semantic_dimensions')
        per_task[row['task']] += 1
        per_checkpoint[key[0]] += 1
        expected = all(human[name] for name in DIMENSIONS[:3]) and not human['unsupported_claims']
        expected = expected and label.get('review_flag_correct') is True and label.get('schema_evidence_valid') is True
        if any(type(label.get(field)) is not bool for field in ('review_flag_correct', 'schema_evidence_valid')):
            raise ValueError('Calibration labels need independent review-flag and validity decisions')
        observed = row['task_score'] == 1
        tables['pass'][f'{int(expected)}/{int(observed)}'] += 1
        bias.append(row['task_score'] - int(expected))
        fields = []
        if machine is not None:
            for name in DIMENSIONS:
                tables[name][f'{int(human[name])}/{int(machine[name])}'] += 1
                if human[name] != machine[name]:
                    fields.append(name)
        if expected != observed or fields:
            disagreements.append({'checkpoint': key[0], 'scenario_id': key[1], 'task': row['task'],
                                  'human_pass': expected, 'semantic_pass': observed,
                                  'judge_dimensions_disagree': fields})
    return {
        'comparison_identity_sha256': identity, 'labels_sha256': digest(labels), 'reviewed_cases': len(labels),
        'reviewed_per_task': dict(per_task), 'reviewed_per_checkpoint': dict(per_checkpoint),
        'human_vs_semantic_confusion': {name: dict(counts) for name, counts in tables.items()},
        'mean_score_bias': sum(bias) / len(bias), 'disagreements': disagreements,
        'scope': 'descriptive development calibration; coverage and sampling govern generalizability',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('report', 'labels', 'output'):
        parser.add_argument(f'--{arg}', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, calibrate(args.report, args.labels))


if __name__ == '__main__':
    main()
