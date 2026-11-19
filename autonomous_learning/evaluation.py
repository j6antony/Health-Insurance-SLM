"""Family-separated evaluation import, frozen predictions and promotion evidence."""
from collections import Counter, defaultdict
from statistics import mean
import json
from pathlib import Path

from self_instruct_pipeline.validation import validate_schema
from distillation_loop.adapter_identity import adapter_files
from distillation_loop.paired import paired_summary
from distillation_loop.quality import promotion_gate
from . import CONTRACT_VERSION
from .dataset import validate_example, reject_leakage
from .storage import digest, read_json
from .tasks import TASKS, SCHEMAS

SCORING = 'document-verification-grounded-v3'
TIERS = ('development', 'probes', 'release_holdout')


def import_benchmark(storage, benchmark, training=()):
    if set(benchmark) != {'version', 'tier', 'process', 'independently_reviewed', 'cases'} or benchmark['version'] != 1:
        raise ValueError('Invalid benchmark import contract')
    if benchmark['tier'] not in TIERS or benchmark['process'] not in ('in_generator', 'independent'):
        raise ValueError('Unknown benchmark tier/process')
    if type(benchmark['independently_reviewed']) is not bool or not benchmark['cases']:
        raise ValueError('Invalid benchmark review status or empty cases')
    if benchmark['tier'] in ('probes', 'release_holdout') and benchmark['process'] != 'independent':
        raise ValueError('Transfer and holdout require independently authored cases')
    if benchmark['tier'] == 'release_holdout' and not benchmark['independently_reviewed']:
        raise ValueError('Release holdout requires independent review')
    ids = [r['id'] for r in benchmark['cases']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate evaluation case ID')
    # Independent authors supply the same task contracts and evidence locations;
    # they need not use the training renderer grammar or silver metadata.
    for row in benchmark['cases']:
        if row['task'] not in TASKS or row['contract_version'] != CONTRACT_VERSION:
            raise ValueError('Evaluation task contract mismatch')
        if not all(row.get(k) for k in ('family_id', 'template_family', 'payload', 'target')):
            raise ValueError('Evaluation source provenance missing')
        if row['payload']['response_schema'] != SCHEMAS[row['task']]:
            raise ValueError('Evaluation schema mismatch')
        validate_schema(row['target'], SCHEMAS[row['task']])
        validate_evidence(row['payload'], row['target'])
    reject_leakage(training, benchmark['cases'])
    identity = digest(benchmark)
    receipt = storage.write(f'evaluation/{benchmark["tier"]}/{identity}.json', benchmark)
    if benchmark['tier'] == 'release_holdout':
        from .dataset import sealed_index
        storage.write('evaluation/release_holdout/exclusions/' + identity + '.json', sealed_index(benchmark['cases']))
    return {**receipt, 'benchmark_id': identity, 'tier': benchmark['tier'], 'case_count': len(ids)}


def validate_evidence(payload, answer):
    from .evidence import check_offsets
    check_offsets(payload, answer)


def score_case(case, prediction, reference_reviewed=False):
    from .evidence import assess_evidence
    answer = None
    schema_valid = offsets_valid = False
    grounding = {'available': False, 'relevant': False, 'sufficient': False, 'basis': 'malformed_or_incomplete'}
    incomplete = bool(prediction.get('truncated') or prediction.get('error'))
    try:
        if incomplete:
            raise ValueError('Incomplete prediction')
        answer = json.loads(prediction['raw'])
        validate_schema(answer, SCHEMAS[case['task']])
        schema_valid = True
        validate_evidence(case['payload'], answer)
        offsets_valid = True
        grounding = assess_evidence(case, answer, reference_reviewed)
    except (ValueError, KeyError, TypeError, IndexError):
        pass
    evidence_valid = offsets_valid and grounding['available'] and grounding['relevant'] and grounding['sufficient']
    valid = schema_valid and evidence_valid
    reference = case['target']
    correct = bool(schema_valid and answer['result'] == reference['result'])
    review_correct = bool(schema_valid and answer['requires_review'] == reference['requires_review'])
    score = float(valid and correct and review_correct)
    missed = bool(schema_valid and reference['requires_review'] and not answer['requires_review'])
    unsupported = bool(schema_valid and not correct)
    result = {'task': case['task'], 'score': score, 'valid': valid, 'incomplete': incomplete,
              'unsupported_claims': unsupported, 'missed_required_review': missed,
              'schema_valid': schema_valid, 'evidence_offsets_valid': offsets_valid,
              'evidence_relevant': grounding['relevant'], 'evidence_sufficient': grounding['sufficient'],
              'semantic_grounding_available': grounding['available'], 'grounding_basis': grounding['basis'],
              'finding_correct': correct, 'review_flag_correct': review_correct,
              'evidence_valid': evidence_valid, 'uncertainty_correct': review_correct}
    if case['task'] == 'missing_documents':
        required = set(reference['result'])
        result['missing_document_recall'] = (len(required & set(answer['result'])) / len(required)
                                           if valid and required else 1.0 if valid else 0.0)
    return result


def evaluate(benchmark, generations, checkpoint, *, holdout_state=None, release_authorized=False):
    cases = benchmark['cases']
    identity = digest(benchmark)
    binding = digest(adapter_files(checkpoint))
    if benchmark['tier'] == 'release_holdout':
        if holdout_state is None:
            raise ValueError('Sealed evaluation requires a logged holdout access reservation')
        holdout_state.reserve_holdout(identity, binding, release_authorized)
    if (generations['benchmark_id'] != identity or generations['adapter_sha256'] != binding
            or generations['contract_version'] != CONTRACT_VERSION or generations['model_name'] != 'Qwen/Qwen3-8B'):
        raise ValueError('Predictions do not bind the benchmark, adapter and model contracts')
    by_id = generations['predictions']
    if set(by_id) != {r['id'] for r in cases}:
        raise ValueError('Predictions must cover identical complete evaluation cases')
    scores = {r['id']: score_case(r, by_id[r['id']], benchmark['independently_reviewed']) for r in cases}
    grouped = defaultdict(list)
    for row in scores.values():
        grouped[row['task']].append(row['score'])
    per_task = {t: mean(values) for t, values in grouped.items()}
    report = {'contract_version': CONTRACT_VERSION, 'model_name': 'Qwen/Qwen3-8B',
              'benchmark_hash': identity, 'adapter_sha256': binding, 'scoring': SCORING,
              'evaluation_protocol': digest([SCORING, generations.get('inference_config'), CONTRACT_VERSION]),
              'tier': benchmark['tier'], 'process': benchmark['process'], 'per_task': per_task,
              'macro_task_score': mean(per_task.values()), 'minimum_task_score': min(per_task.values()),
              'examples_per_task': dict(Counter(r['task'] for r in cases)), 'case_count': len(cases),
              'case_families': {r['id']: r['family_id'] for r in cases}, 'case_scores': scores,
              'semantic_grounding_available_rate': mean(r['semantic_grounding_available'] for r in scores.values()),
              'evidence_relevant_rate': mean(r['evidence_relevant'] for r in scores.values()),
              'evidence_sufficient_rate': mean(r['evidence_sufficient'] for r in scores.values()),
              'schema_valid_rate': mean(r['schema_valid'] for r in scores.values()),
              'evidence_valid_rate': mean(r['evidence_valid'] for r in scores.values()),
              'schema_evidence_valid_rate': mean(r['valid'] for r in scores.values()),
              'uncertainty_correct_rate': mean(r['uncertainty_correct'] for r in scores.values()),
              'critical_error_count': sum(r['unsupported_claims'] or r['missed_required_review'] for r in scores.values()),
              'missing_document_recall': mean(r['missing_document_recall'] for r in scores.values() if 'missing_document_recall' in r) if any('missing_document_recall' in r for r in scores.values()) else None,
              'inconsistency_detection': per_task.get('cross_document_consistency'),
              'in_generator_accuracy': mean(per_task.values()) if benchmark['process'] == 'in_generator' else None,
              'out_of_generator_transfer_accuracy': mean(per_task.values()) if benchmark['process'] == 'independent' else None,
              'release_ready': False}
    from .evidence import assess_evidence
    available = all(assess_evidence(case, case['target'], benchmark['independently_reviewed'])['available'] for case in cases)
    report['grounded_score_available'] = available
    if not available:
        report.update(macro_task_score=None, minimum_task_score=None,
                      per_task={task: None for task in per_task}, inconsistency_detection=None,
                      in_generator_accuracy=None, out_of_generator_transfer_accuracy=None,
                      grounding_status='unavailable: independent semantic references require review')
    report['report_hash'] = digest(report)
    return report


def evaluate_runtime(runtime, benchmark, checkpoint):
    """Reuses an already loaded StudentRuntime to avoid repeated model loading."""
    from distillation_loop.adapter_identity import checkpoint_binding
    binding = checkpoint_binding(checkpoint)
    if runtime.restored_adapter != binding:
        raise ValueError('Runtime must restore the verified evaluation adapter')
    predictions = runtime.infer_many([r['payload'] for r in benchmark['cases']])
    if len(predictions) != len(benchmark['cases']) or checkpoint_binding(checkpoint) != binding:
        raise ValueError('Incomplete generation or adapter changed')
    generations = {'benchmark_id': digest(benchmark), 'adapter_sha256': digest(adapter_files(checkpoint)),
                   'contract_version': CONTRACT_VERSION, 'model_name': 'Qwen/Qwen3-8B',
                   'inference_config': runtime.config['inference'],
                   'predictions': {r['id']: p for r, p in zip(benchmark['cases'], predictions)}}
    return generations, evaluate(benchmark, generations, checkpoint)


def compare(baseline, candidate):
    for report in (baseline, candidate):
        if report['report_hash'] != digest({k: v for k, v in report.items() if k != 'report_hash'}):
            raise ValueError('Evaluation report integrity mismatch')
    for key in ('benchmark_hash', 'contract_version', 'model_name', 'evaluation_protocol', 'case_families'):
        if baseline[key] != candidate[key]:
            raise ValueError('Incompatible paired evaluation: ' + key)
    if not all(r.get('grounded_score_available', True) for r in (baseline, candidate)):
        return {'accepted': False, 'reason': 'Semantic grounding evaluation unavailable', 'release_ready': False}
    if set(baseline['per_task']) != set(TASKS) or any(n < 2 for n in candidate['examples_per_task'].values()):
        return {'accepted': False, 'reason': 'Require all tasks and at least two cases per task', 'release_ready': False}
    keys = sorted(baseline['case_scores'])
    paired = paired_summary([candidate['case_scores'][k]['score'] - baseline['case_scores'][k]['score'] for k in keys],
                            [baseline['case_families'][k] for k in keys],
                            strata=[baseline['case_scores'][k]['task'] for k in keys])
    gate = promotion_gate(baseline, candidate, {'max_task_regression': 0, 'minimum_macro_improvement': .01},
                          {'macro_task_score': .9, 'minimum_task_score': .8, 'schema_evidence_valid_rate': .98}, TASKS)
    if candidate['critical_error_count'] or candidate['evidence_valid_rate'] < baseline['evidence_valid_rate']:
        gate['accepted'] = False
    return {**gate, 'paired': paired, 'status': 'provisional_best' if gate['accepted'] else 'rejected',
            'release_ready': False, 'limitation': 'Experimental promotion never establishes deployment readiness.'}


def promote(storage, candidate_dir, baseline, candidate, expected_best=None):
    """Caller holds writer lock; compare-and-swap best pointer, preserve adapters."""
    files = adapter_files(candidate_dir)
    if candidate['adapter_sha256'] != digest(files):
        raise ValueError('Candidate adapter differs from evaluated weights')
    best = storage.path('checkpoints/best/registry.json')
    current = read_json(best) if best.exists() else None
    if current != expected_best:
        raise ValueError('Best adapter changed during comparison')
    if current and current['adapter_sha256'] == candidate['adapter_sha256'] and current['report_hash'] == candidate['report_hash']:
        # Recover a crash after best-pointer commit but before stage completion.
        return compare(baseline, candidate)
    if current and baseline['adapter_sha256'] != current['adapter_sha256']:
        raise ValueError('Baseline is not the current best compatible adapter')
    result = compare(baseline, candidate)
    storage.write('checkpoints/candidates/' + candidate['adapter_sha256'] + '.json',
                  {'decision': result, 'report': candidate, 'directory': str(candidate_dir)})
    if result['accepted']:
        if current:
            storage.write('checkpoints/archived/' + digest(current) + '.json', current)
        storage.write('checkpoints/best/registry.json', {'directory': str(candidate_dir),
                      'adapter_sha256': candidate['adapter_sha256'], 'report_hash': candidate['report_hash'],
                      'status': 'provisional_best', 'release_ready': False})
    return result


def make_lighteval_metric():
    """Optional native LightEval metric for the new contract, using saved texts."""
    from lighteval.metrics.metrics_sample import SampleLevelComputation
    from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
    from lighteval.tasks.requests import SamplingMethod

    class DocumentVerificationComputation(SampleLevelComputation):
        def compute(self, model_response, doc, **kwargs):
            row = score_case(doc.specific['case'], {'raw': model_response.text[0],
                                                   'truncated': doc.specific.get('truncated', False)})
            return {'task_score': row['score'], 'schema_evidence_valid': float(row['valid'])}

    return SampleLevelMetricGrouping(
        metric_name=['task_score', 'schema_evidence_valid'],
        higher_is_better={'task_score': True, 'schema_evidence_valid': True},
        category=SamplingMethod.GENERATIVE, sample_level_fn=DocumentVerificationComputation(),
        corpus_level_fn={'task_score': mean, 'schema_evidence_valid': mean})
