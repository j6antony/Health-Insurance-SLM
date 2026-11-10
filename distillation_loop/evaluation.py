"""Native Lighteval custom metrics on responses from the already-loaded student."""
import json
import statistics
from collections import Counter
from importlib.metadata import version

from self_instruct_pipeline.persistence import atomic_json, atomic_jsonl, digest, read_jsonl
from self_instruct_pipeline.prompt_builder import target_payload
from .core import aggregate_scores, score_response
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


class LightevalEvaluator:
    def __init__(self, runtime, schemas, minimum_examples):
        self.runtime, self.schemas, self.minimum_examples = runtime, schemas, minimum_examples

    def evaluate(self, development, output_dir):
        reviewed = all(x.get('review', {}).get('status') == 'approved' for x in development)
        metric = None
        if not reviewed:
            from lighteval.models.model_output import ModelResponse
            from lighteval.tasks.requests import Doc
            metric = make_metric()
        rows = []
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
            payload = target_payload(record, self.schemas)
            # Reference choices and hidden labels are passed only to the metric,
            # never to model.generate or the student's input tokenizer.
            print(f'Evaluation {index}/{len(development)}: {record["task"]} — scoring', flush=True)
            if item.get('review', {}).get('status') == 'approved':
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
            rows.append({'scenario_id':record['id'],'task':record['task'], **scores,
                         'raw':prediction['raw'],'truncated':prediction['truncated']})
        atomic_jsonl(output_dir / 'review_queue.jsonl', pending)
        atomic_jsonl(output_dir / 'predictions.jsonl', rows)
        if pending:
            raise ReviewRequired(f'{len(pending)} answers need independent adjudication: {output_dir}/review_queue.jsonl. Add adjudications.jsonl and resume; generations are frozen.')
        report = aggregate_scores(rows,self.schemas,self.minimum_examples)
        report['grading_basis_counts'] = dict(Counter(row.get('grading_basis', 'synthetic_smoke_metric') for row in rows))
        adjudicated = [row['correctness_review'] for row in rows if 'correctness_review' in row]
        report['human_adjudicated_count'] = len(adjudicated)
        if adjudicated:
            report['human_review_dimension_rates'] = {name: statistics.mean(int(j[name]) for j in adjudicated) for name in adjudicated[0]}
        # Use Lighteval's corpus aggregation functions as well as per-task macro
        # aggregation; micro metrics are reported separately from the stop gate.
        aggregations = metric.get_corpus_aggregations() if metric else {name: statistics.mean for name in ('task_score', 'schema_evidence_valid')}
        report['lighteval_corpus'] = {name: fn([r[name] for r in rows]) for name, fn in aggregations.items()}
        report.update(engine='reviewed-reference-and-human-adjudication' if reviewed else 'lighteval-0.11.0-custom-metrics',
                      reference_status=('independently reviewed references' if all(x.get('review', {}).get('status') == 'approved' for x in development) else 'synthetic smoke references; not independent gold'),
                      scoring=SCORING_VERSION, adjudications_hash=digest(judgments),
                      benchmark_hash=digest(development), generation_key=generation_key)
        output_dir.mkdir(parents=True,exist_ok=True)
        atomic_jsonl(output_dir/'predictions.jsonl',rows)
        atomic_json(output_dir/'scores.json',report)
        return report
