"""Native Lighteval custom metrics on responses from the already-loaded student."""
import json
import statistics
from importlib.metadata import version

from self_instruct_pipeline.persistence import atomic_json, atomic_jsonl
from self_instruct_pipeline.prompt_builder import target_payload
from .core import aggregate_scores, score_response
from .student import infer_payloads


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
        from lighteval.models.model_output import ModelResponse
        from lighteval.tasks.requests import Doc
        metric = make_metric()
        rows = []
        print(f'Evaluating {output_dir.name}: {len(development)} student generations.', flush=True)
        predictions = infer_payloads(self.runtime, [target_payload(item['record'], self.schemas) for item in development])
        if len(predictions) != len(development):
            raise ValueError('Evaluation predictions are incomplete')
        for index, (item, prediction) in enumerate(zip(development, predictions), 1):
            record, reference = item['record'], item['reference']
            payload = target_payload(record, self.schemas)
            # Reference choices and hidden labels are passed only to the metric,
            # never to model.generate or the student's input tokenizer.
            print(f'Evaluation {index}/{len(development)}: {record["task"]} — scoring', flush=True)
            doc = Doc(query=json.dumps(payload, ensure_ascii=False),
                      choices=[json.dumps(reference, ensure_ascii=False)], gold_index=0,
                      specific={'record':record,'schemas':self.schemas,'truncated':prediction['truncated']})
            response = ModelResponse(text=[prediction['raw']])
            scores = metric.compute_sample(model_response=response, doc=doc)
            rows.append({'scenario_id':record['id'],'task':record['task'], **scores,
                         'raw':prediction['raw'],'truncated':prediction['truncated']})
        report = aggregate_scores(rows,self.schemas,self.minimum_examples)
        # Use Lighteval's corpus aggregation functions as well as per-task macro
        # aggregation; micro metrics are reported separately from the stop gate.
        report['lighteval_corpus'] = {name:fn([r[name] for r in rows])
                                     for name,fn in metric.get_corpus_aggregations().items()}
        report.update(engine='lighteval-0.11.0-custom-metrics',
                      reference_status=('independently reviewed references' if all(x.get('review', {}).get('status') == 'approved' for x in development) else 'synthetic smoke references; not independent gold'),
                      scoring='strict normalized reference agreement; ROUGE is not a correctness gate')
        output_dir.mkdir(parents=True,exist_ok=True)
        atomic_jsonl(output_dir/'predictions.jsonl',rows)
        atomic_json(output_dir/'scores.json',report)
        return report
