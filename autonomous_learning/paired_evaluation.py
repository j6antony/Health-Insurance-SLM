"""Frozen, development-only paired evaluation. Never trains or promotes adapters."""
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
import json

from .storage import digest, checksum, atomic_json
from .dataset import content, validate_example
from .tasks import TASKS
from .evaluation import DEVELOPMENT_PROTOCOL, SCORING, evaluate, compare


def development_leakage(training, cases):
    """Keep the existing .98 content threshold; only template sharing is permitted.

    This exception is never used by independent probes or sealed holdouts.
    Synthetic IDs are removed using the existing duplicate-screening function.
    """
    training = list(training)
    families = {r['family_id'] for r in training}
    ids = {r['id'] for r in training}
    texts = sorted({content(r) for r in training})
    seen, seen_ids = set(), set()
    for case in cases:
        if case['family_id'] in families or case['id'] in ids:
            raise ValueError('Development source-family or ID leakage')
        if case['id'] in seen_ids:
            raise ValueError('Duplicate evaluation ID')
        seen_ids.add(case['id'])
        text = content(case)
        if text in seen:
            raise ValueError('Duplicate evaluation document content')
        seen.add(text)
        for old in texts:
            matcher = SequenceMatcher(None, text, old, autojunk=False)
            if matcher.quick_ratio() >= .98 and matcher.ratio() >= .98:
                raise ValueError('Development training/evaluation near-duplicate leakage')
    return {'training_rows': len(training), 'training_families': len(families),
            'unique_training_document_contents': len(texts), 'evaluation_families': len({r['family_id'] for r in cases}),
            'shared_template_families': sorted({r['template_family'] for r in training} & {r['template_family'] for r in cases}),
            'normalized_content_threshold': .98, 'family_id_overlap': 0, 'exact_or_near_content_overlap': 0,
            'limitation': 'Same-generator development only; no independent transfer or release claim.'}


def freeze(storage, training_paths):
    """Write once before inference; resume reuses the identical benchmark."""
    base = storage.path('experiments/test-c-baseline-comparison')
    base.mkdir(exist_ok=True)
    benchmark_path, manifest_path = base / 'benchmark.json', base / 'benchmark-manifest.json'
    if benchmark_path.exists() or manifest_path.exists():
        if not (benchmark_path.exists() and manifest_path.exists()):
            raise ValueError('Partial benchmark freeze exists; inspect instead of regenerating')
        manifest = json.loads(manifest_path.read_text())
        benchmark = json.loads(benchmark_path.read_text())
        if checksum(benchmark_path) != manifest['benchmark_file_sha256'] or digest(benchmark) != manifest['benchmark_id']:
            raise ValueError('Frozen benchmark changed')
        return benchmark, manifest
    training, sources = [], []
    for p in sorted(set(map(Path, training_paths))):
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        training.extend(rows)
        sources.append({'path': str(p), 'sha256': checksum(p), 'examples': len(rows)})
    if not training:
        raise ValueError('Training lineage exclusions are required')
    from .scenarios import make_scenario
    from .rendering import render
    from .tasks import project
    categories = ('hospitalization', 'outpatient', 'pharmacy', 'diagnostic')
    states = ('valid', 'missing', 'unreadable', 'incomplete', 'date_conflict', 'unknown_signature', 'unknown_applicability')
    cases, specifications = [], []
    for ti, task in enumerate(TASKS):
        for ci, category in enumerate(categories):
            state = states[(ti + ci) % len(states)]
            condition = None if state == 'unknown_applicability' else bool((ti + ci) % 2)
            for attempt in range(30):
                index = ti * 400 + ci * 60 + attempt
                scenario = make_scenario(2026111903, index, category, condition, state, (ti + ci + attempt) % 9)
                row = project(scenario, render(scenario))[ti]
                validate_example(row)
                try:
                    development_leakage(training, [*cases, row])
                except ValueError:
                    continue
                cases.append(row)
                specifications.append({'case_id': row['id'], 'family_id': row['family_id'], 'task': task,
                                       'category': category, 'state': state, 'seed': 2026111903, 'index': index})
                break
            else:
                raise ValueError('Could not construct a separated development case')
    benchmark = {'version': 2, 'tier': 'development', 'process': 'in_generator',
                 'independently_reviewed': False, 'development_protocol': DEVELOPMENT_PROTOCOL, 'cases': cases}
    from .evaluation import import_benchmark
    import_benchmark(storage, benchmark, training)
    atomic_json(benchmark_path, benchmark)
    manifest = {'version': 1, 'status': 'frozen_before_any_inference', 'benchmark_id': digest(benchmark),
                'benchmark_file_sha256': checksum(benchmark_path), 'scoring': SCORING,
                'development_protocol': DEVELOPMENT_PROTOCOL, 'cases': 28,
                'per_task': dict(Counter(r['task'] for r in cases)), 'specifications': specifications,
                'leakage': development_leakage(training, cases), 'training_sources': sources,
                'lineage_limit': 'All available seven-task cumulative exports and Test B replay are excluded. The recovered legacy adapter has no complete historical training manifest; disjointness from unknown historical corpora cannot be certified.',
                'prohibited_uses': ['training', 'independent_probe', 'release_holdout', 'automatic_promotion']}
    atomic_json(manifest_path, manifest)
    return benchmark, manifest


def _atoms(case, answer):
    values = answer['result']
    if case['task'] in ('conditional_requirements', 'cross_document_consistency', 'finding_explanation'):
        return {digest([x['rule_id'], x['status']]) for x in values}
    return {digest(x) for x in values}


def diagnose(case, prediction, score):
    labels = []
    if prediction.get('truncated'): labels.append('truncation')
    if prediction.get('error'): labels.append('generation_failure')
    try:
        answer = json.loads(prediction['raw'])
    except (ValueError, TypeError, KeyError):
        return {'categories': labels + ['invalid_output_json'], 'structured_unsupported': None,
                'rule_or_result_correct': False, 'free_text_assertion_verified': None}
    if not score['schema_valid']:
        return {'categories': labels + ['output_schema_mismatch'], 'structured_unsupported': None,
                'rule_or_result_correct': False, 'free_text_assertion_verified': None}
    expected, actual = _atoms(case, case['target']), _atoms(case, answer)
    if actual != expected:
        labels.append({'document_classification': 'wrong_document_classification',
                       'field_extraction': 'missing_or_hallucinated_fields',
                       'missing_documents': 'incorrect_missing_document_finding',
                       'evidence_quality': 'incorrect_evidence_quality',
                       'conditional_requirements': 'incorrect_rule_identification_or_status',
                       'cross_document_consistency': 'incorrect_rule_identification_or_status',
                       'finding_explanation': 'incorrect_rule_identification_or_status'}[case['task']])
    elif not score['finding_correct']:
        labels.append('noncanonical_reason_or_result_order')
    if case['target']['evidence'] and not answer['evidence']: labels.append('missing_evidence_citation')
    if not score['evidence_offsets_valid']: labels.append('invalid_evidence_offset_or_source_reference')
    elif score['semantic_grounding_available']:
        if not score['evidence_relevant']: labels.append('irrelevant_evidence')
        if not score['evidence_sufficient']: labels.append('insufficient_evidence')
    if not score['review_flag_correct']: labels.append('wrong_review_flag')
    if not score['finding_correct'] and case['task'] == 'finding_explanation':
        labels.append('unverified_explanation')
    return {'categories': labels, 'structured_unsupported': bool(actual - expected),
            'rule_or_result_correct': actual == expected,
            'free_text_assertion_verified': None,
            'free_text_limitation': 'Arbitrary reason wording is not semantically judged; canonical exact reasons remain part of strict v3 task scoring.'}


def analyze(benchmark, baseline, candidate, output):
    """Offline scoring of immutable raw responses; no repair or model calls."""
    output = Path(output); output.mkdir(exist_ok=True, parents=True)
    reports, details, metrics = {}, {}, {}
    for name, arm in (('baseline', baseline), ('candidate', candidate)):
        report = evaluate(benchmark, arm['generations'], arm['checkpoint'])
        per_case = {}
        for case in benchmark['cases']:
            prediction = arm['generations']['predictions'][case['id']]
            per_case[case['id']] = {**report['case_scores'][case['id']],
                                    **diagnose(case, prediction, report['case_scores'][case['id']])}
        values = list(per_case.values())
        assessed = [r for r in values if r['semantic_grounding_available']]
        structured = [r for r in values if r['structured_unsupported'] is not None]
        metrics[name] = {'macro_task_score': report['macro_task_score'],
            'schema_valid_rate': report['schema_valid_rate'],
            'evidence_offset_valid_rate': mean(r['evidence_offsets_valid'] for r in values),
            'evidence_valid_rate': report['evidence_valid_rate'],
            'evidence_relevance_assessable_rate': len(assessed) / len(values),
            'evidence_relevance_rate_assessable_only': mean(r['evidence_relevant'] for r in assessed) if assessed else None,
            'evidence_sufficiency_rate_assessable_only': mean(r['evidence_sufficient'] for r in assessed) if assessed else None,
            'canonical_finding_accuracy': mean(r['finding_correct'] for r in values),
            'rule_or_result_accuracy_without_reason_wording': mean(r['rule_or_result_correct'] for r in values),
            'review_flag_accuracy': report['uncertainty_correct_rate'],
            'truncation_rate': mean(bool(arm['generations']['predictions'][c['id']].get('truncated')) for c in benchmark['cases']),
            'generation_failure_rate': mean(bool(arm['generations']['predictions'][c['id']].get('error')) for c in benchmark['cases']),
            'unsupported_assertion_rate': None,
            'unsupported_assertion_limitation': 'Full free-text semantic assertion checking unavailable; the next metric covers deterministic structured result atoms only.',
            'structured_unsupported_result_rate_schema_valid_only': mean(r['structured_unsupported'] for r in structured) if structured else None,
            'critical_error_count': report['critical_error_count'],
            'critical_error_definition': 'Existing v3 proxy: schema-valid wrong canonical result or missed required review; malformed outputs are counted separately.',
            'per_task': report['per_task'], 'case_count': len(values)}
        reports[name], details[name] = report, per_case
        atomic_json(output / (name + '-report.json'), report)
        atomic_json(output / (name + '-case-scores.json'), per_case)
    comparison = compare(reports['baseline'], reports['candidate'])
    # Never use this comparison's provisional gate as a promotion command.
    comparison.update(promotion_performed=False, development_only=True)
    atomic_json(output / 'paired-comparison.json', comparison)
    atomic_json(output / 'metrics.json', metrics)
    failures, aggregate = {}, {}
    for name in ('baseline', 'candidate'):
        failures[name] = dict(Counter(label for r in details[name].values() for label in r['categories']))
        aggregate[name] = {t: {'cases': sum(r['task'] == t for r in details[name].values()),
                              'failures': sum(r['task'] == t and r['score'] == 0 for r in details[name].values()),
                              'categories': dict(Counter(label for r in details[name].values() if r['task'] == t for label in r['categories']))}
                           for t in TASKS}
    atomic_json(output / 'failure-analysis.json', {'counts': failures, 'per_task': aggregate,
                                                 'baseline_cases': details['baseline'], 'candidate_cases': details['candidate']})
    from .planner import planner_input, heuristic
    feedback = planner_input({**reports['candidate'], 'failure_counts': failures['candidate'], 'paired': comparison.get('paired')},
                             statistics={'evaluation_tier': 'development', 'process': 'in_generator', 'examples_per_task': reports['candidate']['examples_per_task']})
    feedback.update(per_task_failures=aggregate, severity_policy={'missed_required_review': 'critical', 'structured_unsupported_result': 'critical',
                    'invalid_evidence_offset_or_source_reference': 'high', 'invalid_output_json': 'high'},
                    frequency=failures, recommendation_basis='Measured saved outputs only; no sealed case contents.',
                    underrepresented_conditions='Only four cases per task; category/state/task interactions, independent prose and generalization remain unmeasured.')
    plan = heuristic(feedback=feedback)
    # A transparent severity policy breaks zero-macro ties using observed errors,
    # rather than silently selecting the first task in a dictionary.
    weights = {'invalid_output_json': 3, 'output_schema_mismatch': 3,
               'invalid_evidence_offset_or_source_reference': 3, 'missing_evidence_citation': 3,
               'insufficient_evidence': 3, 'irrelevant_evidence': 2, 'wrong_review_flag': 4,
               'truncation': 2, 'generation_failure': 4}
    priorities = {}
    for task in TASKS:
        mine = [r for r in details['candidate'].values() if r['task'] == task]
        priorities[task] = mean(sum(weights.get(label, 2) for label in row['categories'])
                                + 5 * bool(row['structured_unsupported'])
                                + 5 * row['missed_required_review'] for row in mine)
    total = sum(priorities.values())
    if total:
        exact = {t: 65 * priorities[t] / total for t in TASKS}
        focus = {t: 5 + int(exact[t]) for t in TASKS}
        for t in sorted(TASKS, key=lambda t: (-(exact[t] % 1), t))[:100 - sum(focus.values())]:
            focus[t] += 1
        plan['focus'] = focus
    focus_order = sorted(TASKS, key=lambda t: (-priorities[t], t))
    feedback['measured_priority'] = [{'task': t, 'severity_weighted_error_frequency': priorities[t],
                                    'baseline_failures': aggregate['baseline'][t]['failures'],
                                    'candidate_failures': aggregate['candidate'][t]['failures'],
                                    'candidate_categories': aggregate['candidate'][t]['categories']} for t in focus_order]
    plan['rationale'] = 'Prioritize measured severity-weighted development failures; retain at least 5% per task. No independent generalization claim.'
    from .planner import validate_plan
    validate_plan(plan)
    # Keep recommendation offline with zero budgets; never submit it to a cycle.
    atomic_json(output / 'curriculum-feedback.json', {'planner_input': feedback, 'proposed_plan_not_executed': plan})
    ordered = sorted(benchmark['cases'], key=lambda c: (-len(details['candidate'][c['id']]['categories']), c['task'], c['id']))
    examples, used = [], set()
    for case in ordered:
        if case['task'] in used: continue
        used.add(case['task'])
        examples.append({'case_id': case['id'], 'task': case['task'], 'prompt': case['payload'], 'correct_target': case['target'],
                         'baseline': baseline['generations']['predictions'][case['id']], 'candidate': candidate['generations']['predictions'][case['id']],
                         'baseline_scores': details['baseline'][case['id']], 'candidate_scores': details['candidate'][case['id']]})
    atomic_json(output / 'representative-cases.json', examples)
    return {'reports': reports, 'metrics': metrics, 'comparison': comparison, 'failure_counts': failures}


def prepare(storage):
    """Freeze 28 cases and package both immutable adapters for a manual GPU transfer."""
    import shutil, zipfile
    from distillation_loop.adapter_identity import adapter_files
    from .jobs import adapter_compatibility
    from .model import MODEL, MODEL_REVISION
    root = storage.root
    paths = list(storage.path('datasets/approved_sft').glob('*/examples.jsonl'))
    paths += list(storage.path('experiments/quality-pilot-20261119-release/artifacts/datasets/approved_sft').glob('*/examples.jsonl'))
    benchmark, manifest = freeze(storage, paths)
    base = storage.path('experiments/test-c-baseline-comparison')
    baseline = storage.path('checkpoints/candidates/e8e5edc7-drive-transfer/adapter')
    candidate = storage.path('gpu_jobs/imported_results/c03c14696b8ab4b394266f83d850641ae3e71f9eaa3cc749a34de12489834fc5/adapter')
    original = json.loads(storage.path('gpu_jobs/packages/c03c14696b8ab4b394266f83d850641ae3e71f9eaa3cc749a34de12489834fc5/manifest.json').read_text())['job']
    result = json.loads((candidate.parent / 'result.json').read_text())
    if adapter_files(baseline) != original['adapter_files'] or adapter_files(candidate) != result['adapter_files']:
        raise ValueError('Evaluation adapter identity changed')
    config = original['config']
    config['inference'] = {'max_new_tokens': 1536, 'batch_size': 1}
    for adapter in (baseline, candidate):
        if not adapter_compatibility(adapter, config['training'])['compatible']:
            raise ValueError('Evaluation adapter is incompatible')
    source = Path(__file__).resolve().parents[1]
    files = {'benchmark.json': base / 'benchmark.json', 'benchmark-manifest.json': base / 'benchmark-manifest.json',
             'requirements-training.txt': source / 'requirements-training.txt'}
    for package in ('autonomous_learning', 'distillation_loop', 'self_instruct_pipeline'):
        files.update({str(p.relative_to(source)): p for p in (source / package).glob('*.py') if not p.name.startswith('._')})
    for name, adapter in (('baseline', baseline), ('candidate', candidate)):
        files.update({name + '/' + f: adapter / f for f in adapter_files(adapter)})
    constraints = ('torch==2.8.0\ntorchvision==0.23.0\ntorchaudio==2.8.0\ntransformers==4.57.3\ntrl==0.25.1\n'
                   'peft==0.18.0\nlighteval==0.11.0\naccelerate==1.11.0\nbitsandbytes==0.48.2\ndatasets==4.4.1\n'
                   'tokenizers==0.22.2\nhuggingface-hub==0.36.0\nsafetensors==0.6.2\n')
    atomic_path = base / 'constraints.txt'; atomic_path.write_text(constraints)
    files['constraints.txt'] = atomic_path
    job = {'version': 1, 'operation': 'paired_inference_only', 'model_name': MODEL, 'model_revision': MODEL_REVISION,
           'config': config, 'benchmark_id': digest(benchmark), 'benchmark_file_sha256': manifest['benchmark_file_sha256'],
           'scoring': SCORING, 'development_protocol': DEVELOPMENT_PROTOCOL,
           'inference_protocol': {'context': 4096, 'max_new_tokens': 1536, 'batch_size': 1,
                                  'enable_thinking': False, 'do_sample': False, 'seed': 43},
           'adapter_files': {'baseline': adapter_files(baseline), 'candidate': adapter_files(candidate)},
           'files': {name: checksum(p) for name, p in sorted(files.items())},
           'max_gpu_usd': 1.0, 'max_runtime_seconds': 3600,
           'no_training': True, 'no_promotion': True}
    job['job_id'] = digest(job)
    package = base / ('package-' + job['job_id'][:12] + '.zip')
    if not package.exists():
        temporary = package.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            z.writestr('evaluation-job.json', json.dumps(job, sort_keys=True, indent=2))
            for name, path in sorted(files.items()): z.write(path, name)
        temporary.replace(package)
    receipt = {'job': job, 'package': str(package), 'sha256': checksum(package), 'benchmark_manifest': manifest}
    atomic_json(base / 'package-receipt.json', receipt)
    return receipt


def verify_package(root):
    root = Path(root)
    job = json.loads((root / 'evaluation-job.json').read_text())
    if job['job_id'] != digest({k: v for k, v in job.items() if k != 'job_id'}):
        raise ValueError('Evaluation job identity changed')
    for name, expected in job['files'].items():
        p = root / name
        if not p.resolve().is_relative_to(root.resolve()) or p.is_symlink() or checksum(p) != expected:
            raise ValueError('Evaluation package checksum mismatch: ' + name)
    benchmark = json.loads((root / 'benchmark.json').read_text())
    if (digest(benchmark) != job['benchmark_id'] or benchmark['tier'] != 'development'
            or benchmark.get('development_protocol') != DEVELOPMENT_PROTOCOL or len(benchmark['cases']) != 28
            or job['operation'] != 'paired_inference_only' or not job['no_training'] or not job['no_promotion']):
        raise ValueError('Paired evaluation protocol mismatch')
    for row in benchmark['cases']: validate_example(row)
    return job, benchmark


def main():
    import argparse
    from .storage import Storage
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('prepare',))
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    storage = Storage(args.root); storage.initialize()
    with storage.lock(): receipt = prepare(storage)
    print(json.dumps({'package': receipt['package'], 'sha256': receipt['sha256'], 'job_id': receipt['job']['job_id'],
                      'benchmark_id': receipt['job']['benchmark_id']}, indent=2))


if __name__ == '__main__': main()
