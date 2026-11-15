"""Native Lighteval custom metrics on responses from the already-loaded student."""
import json
import statistics
from collections import Counter
from importlib.metadata import version

from self_instruct_pipeline.persistence import atomic_json, atomic_jsonl, digest, read_jsonl
from self_instruct_pipeline.prompt_builder import target_payload
from .core import aggregate_scores, clean_response, score_response
from .student import infer_payloads
from .review import grade_reviewed, review_key, ReviewRequired, SCORING_VERSION


def require_training_versions():
    expected = {'transformers':'4.57.3','trl':'0.25.1','peft':'0.18.0','lighteval':'0.11.0'}
    for package, required in expected.items():
        installed = version(package)
        if installed != required:
            raise ValueError(f'{package}=={required} required; found {installed}. Install requirements-training.txt and restart kernel.')


def make_metric():
    from lighteval.metrics.metrics_sample import SampleLevelComputation
    from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
    from lighteval.tasks.requests import SamplingMethod

    class ClaimsComputation(SampleLevelComputation):
        def compute(self, model_response, doc, **kwargs):
            return score_response(doc.specific['record'], json.loads(doc.get_golds()[0]),
                                  model_response.text[0], doc.specific['schemas'],
                                  truncated=doc.specific['truncated'])

    return SampleLevelMetricGrouping(
        metric_name=['task_score','schema_evidence_valid'],
        higher_is_better={'task_score':True,'schema_evidence_valid':True},
        category=SamplingMethod.GENERATIVE, sample_level_fn=ClaimsComputation(),
        corpus_level_fn={'task_score':statistics.mean, 'schema_evidence_valid':statistics.mean})


def incomplete_prediction(prediction):
    return bool(prediction['truncated'] or prediction.get('error')
                or prediction.get('stop_reason') in {'token_limit', 'repetition', 'context_budget'})


def safety_case(item, prediction, scores):
    valid = scores['schema_evidence_valid'] == 1
    missed = False
    if valid:
        answer = clean_response(prediction['raw'], incomplete_prediction(prediction))
        variants = [item['reference'], *item.get('accepted_references', [])]
        def required(value):
            return {field['field_name'] for field in value.get('fields', []) if field['requires_review']}
        required_fields = set.intersection(*(required(value) for value in variants))
        missed = ((all(value.get('requires_review') is True for value in variants)
                   and answer.get('requires_review') is not True)
                  or bool(required_fields - required(answer)))
    dimensions = scores.get('semantic_dimensions') or scores.get('correctness_review') or {}
    return {'task': item['record']['task'], 'score': scores['task_score'], 'valid': valid,
            'incomplete': incomplete_prediction(prediction),
            'unsupported_claims': dimensions.get('unsupported_claims') is True,
            'missed_required_review': missed}


class LightevalEvaluator:
    def __init__(self, runtime, schemas, minimum_examples):
        self.runtime, self.schemas, self.minimum_examples = runtime, schemas, minimum_examples

    def evaluate(self, development, output_dir):
        judge = getattr(self, 'semantic_judge', None)
        reviewed = all(x.get('review', {}).get('status') == 'approved' for x in development)
        metric = None
        if not reviewed and judge is None:
            from lighteval.models.model_output import ModelResponse
            from lighteval.tasks.requests import Doc
            metric = make_metric()
        rows, case_scores = [], {}
        print(f'Evaluating {output_dir.name}: {len(development)} student generations.', flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        generation_path = output_dir / 'generation.json'
        generation_key = digest({'data': development, 'student': self.runtime.identity(), 'scoring': SCORING_VERSION,
                                 'checkpoint': getattr(self, 'checkpoint_identity', None)})
        if generation_path.exists():
            saved = json.loads(generation_path.read_text())
            if saved['key'] != generation_key:
                raise ValueError('Evaluation inputs changed; refusing stale prediction reuse')
            predictions = saved['predictions']
        else:
            predictions = infer_payloads(self.runtime, [target_payload(item['record'], self.schemas) for item in development])
            if len(predictions) != len(development):
                raise ValueError('Evaluation predictions are incomplete')
            atomic_json(generation_path, {'key': generation_key, 'predictions': predictions})
        judgments = {}
        for judgment in read_jsonl(output_dir / 'adjudications.jsonl'):
            key = judgment['review_key']
            if key in judgments:
                raise ValueError('Duplicate review judgment')
            judgments[key] = judgment
        pending = []
        valid_keys = {review_key(item, prediction) for item, prediction in zip(development, predictions)}
        if set(judgments) - valid_keys:
            raise ValueError('Stale adjudication does not match this benchmark and prediction')
        if len(predictions) != len(development):
            raise ValueError('Evaluation predictions are incomplete')
        for index, (item, prediction) in enumerate(zip(development, predictions), 1):
            record, reference = item['record'], item['reference']
            # Keep raw output intact while scoring every incomplete stop as invalid.
            scoring_prediction = {**prediction, 'truncated': incomplete_prediction(prediction)}
            payload = target_payload(record, self.schemas)
            # Reference choices and hidden labels are passed only to the metric,
            # never to model.generate or the student's input tokenizer.
            print(f'Evaluation {index}/{len(development)}: {record["task"]} — scoring', flush=True)
            if judge is not None:
                scores = judge.grade(item, scoring_prediction)
            elif scoring_prediction['truncated']:
                scores = {'task_score': 0.0, 'schema_evidence_valid': 0.0, 'grading_basis': 'incomplete_generation'}
            elif item.get('review', {}).get('status') == 'approved':
                scores, request = grade_reviewed(item, prediction, self.schemas, judgments)
                if request:
                    pending.append(request)
                    continue
            else:
                doc = Doc(query=json.dumps(payload, ensure_ascii=False),
                          choices=[json.dumps(reference, ensure_ascii=False)], gold_index=0,
                          specific={'record':record,'schemas':self.schemas,'truncated':prediction['truncated']})
                response = ModelResponse(text=[prediction['raw']])
                scores = metric.compute_sample(model_response=response, doc=doc)
            case_scores[record['id']] = safety_case(item, prediction, scores)
            rows.append({'scenario_id':record['id'],'task':record['task'], **scores,
                         'raw':prediction['raw'],'truncated':prediction['truncated']})
        atomic_jsonl(output_dir / 'review_queue.jsonl', pending)
        atomic_jsonl(output_dir / 'predictions.jsonl', rows)
        if pending:
            raise ReviewRequired(f'{len(pending)} answers need independent adjudication: {output_dir}/review_queue.jsonl. Add adjudications.jsonl and resume; generations are frozen.')
        report = aggregate_scores(rows,self.schemas,self.minimum_examples)
        report['grading_basis_counts'] = dict(Counter(row.get('grading_basis', 'automatic_reference_metric') for row in rows))
        adjudicated = [row['correctness_review'] for row in rows if 'correctness_review' in row]
        if judge is not None:
            report['per_task_diagnostics'] = {
                task: {field: statistics.mean(r[field] for r in rows if r['task'] == task)
                       for field in ('schema_evidence_valid', 'review_flag_correct', 'reference_evidence_overlap')}
                for task in self.schemas
            }
        report['case_scores'] = case_scores
        report['human_adjudicated_count'] = len(adjudicated)
        if adjudicated:
            report['human_review_dimension_rates'] = {name: statistics.mean(int(j[name]) for j in adjudicated) for name in adjudicated[0]}
        # Use Lighteval's corpus aggregation functions as well as per-task macro
        # aggregation; micro metrics are reported separately from the stop gate.
        aggregations = metric.get_corpus_aggregations() if metric else {name: statistics.mean for name in ('task_score', 'schema_evidence_valid')}
        report['lighteval_corpus'] = {name: fn([r[name] for r in rows]) for name, fn in aggregations.items()}
        fixed = all(x.get('verification', {}).get('status') == 'locally_checked_synthetic' for x in development)
        report.update(engine='reviewed-reference-and-human-adjudication' if reviewed else 'lighteval-0.11.0-custom-metrics',
                      reference_status=('independently reviewed references' if all(x.get('review', {}).get('status') == 'approved' for x in development) else 'fixed locally authored synthetic references; no independent expert review' if fixed else 'automatically checked synthetic references; no human review or independent gold'),
                      scoring=SCORING_VERSION, adjudications_hash=digest(judgments),
                      benchmark_hash=digest(development), generation_key=generation_key)
        if judge is not None:
            report.update(engine='source-rubric-model-judge', scoring=judge.version,
                          reference_status='synthetic references with automated source-grounded judgments')
        runtime_settings = getattr(self.runtime, 'config', {})
        report['evaluation_protocol'] = digest({
            'engine': report['engine'], 'scoring': report['scoring'], 'student': self.runtime.identity(),
            'schemas': self.schemas, 'inference': runtime_settings.get('inference'),
            'max_sequence_length': runtime_settings.get('training', {}).get('max_sequence_length'),
            'evidence_first': runtime_settings.get('training', {}).get('evidence_first', False),
            'judge': getattr(getattr(judge, 'parent', None), 'settings', None),
        })
        output_dir.mkdir(parents=True,exist_ok=True)
        atomic_jsonl(output_dir/'predictions.jsonl',rows)
        atomic_json(output_dir/'scores.json',report)
        return report
