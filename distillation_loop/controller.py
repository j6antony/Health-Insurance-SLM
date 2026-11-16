"""One bounded loop: generate, infer, get teacher targets, train, evaluate."""
import json
import math
import os
from pathlib import Path
from time import monotonic

from self_instruct_pipeline.config import resolve_path
from self_instruct_pipeline.persistence import Store, atomic_json, digest, read_jsonl
from self_instruct_pipeline.prompt_builder import target_payload
from self_instruct_pipeline.validation import ValidationError, validate_answer
from .adapter_identity import adapter_files, checkpoint_binding
from .core import ResponseError, clean_response, feedback_weights, satisfactory, validate_report
from .correction import adaptive_pairs, balanced_selection
from .data import CurriculumData, frozen_development
from .evaluation import LightevalEvaluator
from .benchmarks import load_benchmarks, load_fixed_validation, comparison
from .review import SCORING_VERSION
from .trusted_templates import TEMPLATE_VERSION
from .quality import (operational_gate, promotion_gate, regression_gate, require_experimental_result,
                      training_coverage, validate_quality_settings)
from .leakage import ExclusionRegistry, freeze_loss_split, validate_frozen_training_records
from .semantic import SemanticJudge
from .student import infer_payloads
from .kaggle_backup import check_backup_setup, save_adapter_dataset


def save_predictions(runtime, records, schemas, directory, *, checkpoint=None):
    store = Store(directory)
    existing = {row['scenario_id']: row for row in
                store.unique_records('student_responses.jsonl', 'scenario_id')}
    pending = []
    for record in records:
        if record['id'] in existing:
            if (existing[record['id']]['scenario_hash'] != digest(record)
                    or existing[record['id']].get('checkpoint') != checkpoint):
                raise ValueError('Student prediction belongs to a changed scenario')
        else:
            pending.append(record)
    if not pending:
        return existing
    predictions = infer_payloads(runtime, [target_payload(record, schemas) for record in pending])
    if len(predictions) != len(pending):
        raise ValueError('Missing student diagnostic predictions')
    for record, prediction in zip(pending, predictions):
        cleaned = None
        error = prediction.get('error')
        try:
            cleaned = clean_response(prediction['raw'], prediction['truncated'])
            validate_answer(record, cleaned, schemas[record['task']])
        except (ResponseError, ValidationError, KeyError, TypeError) as exc:
            cleaned, error = None, str(exc)
        row = {
            'scenario_id': record['id'], 'scenario_hash': digest(record), 'task': record['task'],
            'raw': prediction['raw'], 'cleaned': cleaned,
            'schema_evidence_valid': cleaned is not None,
            'truncated': prediction['truncated'], 'error': error,
            'purpose': 'pre-training student prediction',
        }
        if checkpoint is not None:
            row['checkpoint'] = checkpoint
        store.append('student_responses.jsonl', row)
        existing[record['id']] = row
    return existing


def backup_destination():
    configured = [destination for destination, enabled in (
        ('drive', os.environ.get('DISTILLATION_DRIVE_BACKUP') == '1'),
        ('github', os.environ.get('GITHUB_ADAPTER_UPLOAD') == '1'),
        ('kaggle', bool(os.environ.get('KAGGLE_ADAPTER_DATASET', '').strip())),
    ) if enabled]
    if len(configured) > 1:
        raise ValueError('Configure only one adapter backup destination')
    return configured[0] if configured else None


def backup_adapter(destination, checkpoint, run_dir, round_id, *, experimental):
    if destination == 'drive':
        from .drive_backup import save_drive_adapter
        save_drive_adapter(checkpoint, run_dir, round_id, require_mounted=True)
    elif destination == 'kaggle':
        save_adapter_dataset(checkpoint, run_dir, round_id)
    elif destination == 'github':
        if experimental:
            directory = Path(run_dir) / f'round_{round_id:02d}'
            directory.mkdir(exist_ok=True)
            atomic_json(directory / 'backup_status.json', {
                'destination': 'github', 'status': 'blocked', 'checkpoint': str(checkpoint),
                'reason': 'Experimental checkpoints are ineligible for GitHub adapter upload',
            })
            print('GitHub adapter backup blocked for experimental checkpoint.', flush=True)
        else:
            from .github_backup import upload_latest_adapter
            upload_latest_adapter(run_dir)


def run_loop(runtime, parent_config, config, *, round_limit=None, initial_adapter=None,
             semantic_result=None, continuation_authorized=False, evaluate_release=False):
    decision = None
    if config.get('require_semantic_decision'):
        if not semantic_result or continuation_authorized is not True:
            raise ValueError('Stage 2 stop: supply the saved semantic result and authorize continuation')
        decision = json.loads(Path(semantic_result).read_text())
        decision = {**decision, 'diagnostic_decision': require_experimental_result(decision, config.get('quality', {}))}
        from self_instruct_pipeline.budget import ACTIVE_BUDGET
        if ACTIVE_BUDGET.get() is None:
            raise ValueError('The revised semantic loop requires an active cumulative API budget')
    # Per-call journals are already saved, even if training/evaluation fails.
    try:
        return _run_loop(runtime, parent_config, config, round_limit=round_limit, initial_adapter=initial_adapter,
                         decision=decision, evaluate_release=evaluate_release)
    finally:
        from self_instruct_pipeline.usage import export_usage
        directory = resolve_path(config['output_dir'])
        if directory.exists():
            print('API usage:', export_usage(directory), flush=True)


def _run_loop(runtime, parent_config, config, *, round_limit, initial_adapter, decision, evaluate_release):
    from .hybrid import options as hybrid_options
    hybrid = hybrid_options(config)
    if hybrid and (evaluate_release or not config['evaluation'].get('split_path')):
        raise ValueError('Hybrid experiments require frozen development and keep release scoring separate')
    store = Store(resolve_path(config['output_dir']))
    backup_target = backup_destination()
    if backup_target == 'kaggle':
        check_backup_setup()
    schemas = json.loads(resolve_path(parent_config['paths']['schemas']).read_text())['tasks']
    settings_hash = digest({'config': config, 'parent': parent_config,
                            'schemas': schemas, 'student': runtime.identity(), 'scoring': SCORING_VERSION, 'trusted_templates': TEMPLATE_VERSION})
    if config['evaluation'].get('semantic_judge'):
        settings_hash = digest([settings_hash, SemanticJudge.version])
    historical, history_hashes = [], []
    if hybrid:
        from .leakage import historical_records
        paths = [config['historical_training_plan'], *config.get('hybrid_prior_training_plans', [])]
        for path in paths:
            records, plan_hash = historical_records(resolve_path(path), schemas)
            historical.extend(records)
            history_hashes.append(plan_hash)
        settings_hash = digest([settings_hash, 'hybrid-v1', history_hashes])
    state_path = store.path('loop_state.json')
    saved_split = config['evaluation'].get('split_path')
    semantic_scoring = bool(config['evaluation'].get('semantic_judge'))
    release_path = store.path('release/intent.json')
    quality = validate_quality_settings(config, schemas)
    adaptive = config.get('student_conditioned_training', False)
    stop_on_thresholds = config.get('stop_when_thresholds_met', True)
    if config.get('preference_training') and not callable(getattr(runtime, 'train_preferences', None)):
        raise ValueError('Preference training requires a runtime with train_preferences')
    if type(adaptive) is not bool:
        raise ValueError('student_conditioned_training must be a boolean')

    def passed(report):
        return satisfactory(report, config['thresholds'], schemas)

    def unused_path(path):
        if not path.exists():
            return path
        number = 2
        while path.with_name(f'{path.name}-retry-{number:02d}').exists():
            number += 1
        return path.with_name(f'{path.name}-retry-{number:02d}')

    with store.lock():
        reviewed, final_test = load_benchmarks(config, schemas)
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state and state.get('settings_hash') != settings_hash:
            raise ValueError('Settings changed; use a new output directory')
        development = reviewed if reviewed is not None else frozen_development(parent_config, config, store.directory)
        if state and (state['development_hash'] != digest(development) or state.get('test_hash') != digest(final_test)):
            raise ValueError('Frozen evaluation data have changed')
        if decision is not None:
            identity = decision.get('identity', {})
            if (identity.get('development_sha256') != digest(development)
                    or identity.get('holdout_sha256') != digest(final_test)
                    or identity.get('schema_sha256') != digest(schemas)
                    or decision.get('scoring') != SemanticJudge.version):
                raise ValueError('Semantic decision does not match the benchmark, schema, and rubric')
            expected_cases = {item['record']['id']: item['record']['task'] for item in development}
            for stage in ('initial', 'sft'):
                report = decision['runs'][stage]
                if (report.get('benchmark_hash') != identity['development_sha256']
                        or report.get('scoring') != SemanticJudge.version
                        or report.get('evaluation_protocol') != identity.get('comparison_manifest_sha256')
                        or {key: row['task'] for key, row in report['case_scores'].items()} != expected_cases):
                    raise ValueError('Semantic decision does not contain the complete frozen development comparison')

        if final_test and config['evaluation']['mode'] == 'fixed':
            manifest_path = store.path('evaluation_split.json')
            if saved_split:
                split = json.loads(resolve_path(saved_split).read_text())
            else:
                split = {'dataset': config['evaluation']['validation_path'],
                         'seed': config['evaluation'].get('subset_seed', config['random_seed'])}
                for name, rows in [('development', development), ('holdout', final_test)]:
                    split[name] = {'sha256': digest(rows),
                                   'case_ids': [item['record']['id'] for item in rows],
                                   'source_groups': sorted({item['source_group'] for item in rows})}
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != split:
                raise ValueError('Frozen evaluation split manifest has changed')
            if not manifest_path.exists():
                atomic_json(manifest_path, split)
        loss_split = None
        registry = None
        if saved_split:
            if not config.get('historical_training_plan'):
                raise ValueError('The saved split requires the original Round 1 training plan')
            candidates = config.get('teacher_loss_candidates')
            loss_split = freeze_loss_split(
                store.path('heldout_teacher_loss_split.json'), resolve_path(config['historical_training_plan']),
                development + final_test, schemas,
                candidates_path=resolve_path(candidates) if candidates else None,
                exposure_roots=[resolve_path(root) for root in config.get('historical_exposure_roots', ())])
            registry = ExclusionRegistry(development + final_test + loss_split['rows'])
            validate_frozen_training_records(store, registry)
        if evaluate_release and saved_split and (state is None or state['status'] == 'running'):
            raise ValueError('A controlled release evaluation requires a completed development run')
        previous_limit = state.get('round_limit', config['max_rounds']) if state else config['max_rounds']
        limit = previous_limit if round_limit is None else round_limit
        if type(limit) is not int or limit < previous_limit:
            raise ValueError('Round limit must be an integer at least as large as the saved limit')
        if state and limit > previous_limit:
            if (store.path('final_test_baseline').exists()
                    or store.path('final_test_selected').exists() or release_path.exists()):
                raise ValueError('The final holdout has already been used for this run')
            state['round_limit'] = limit
            if state['status'] == 'max_rounds_reached':
                state['status'] = 'running'
            atomic_json(state_path, state)
        runtime.prepare()
        evaluator = LightevalEvaluator(runtime, schemas, config['development_examples_per_task'])

        if state is None:
            print('Development set complete. Saving initial student adapter (local; no teacher API calls).', flush=True)
            if initial_adapter is not None:
                print('Initializing from the GitHub release adapter.', flush=True)
                runtime.restore(initial_adapter)
            initial = runtime.save_checkpoint(unused_path(store.path('initial_adapter')))
            state = {
                'settings_hash': settings_hash, 'development_hash': digest(development),
                'test_hash': digest(final_test),
                'completed_rounds': 0, 'history': [], 'status': 'running', 'round_limit': limit,
                'initial_checkpoint': initial, 'last_checkpoint': initial, 'best_checkpoint': initial,
                'next_round_start_checkpoint': initial,
                'best_report': None, 'latest_report': None, 'latest_checkpoint': initial,
                'provisional_best_checkpoint': None,
                **({'release_checkpoint': None} if saved_split else {}),
                **({'diagnostic_decision': decision['diagnostic_decision'],
                    'semantic_decision_sha256': digest(decision)} if decision is not None else {}),
            }
            atomic_json(state_path, state)
        if decision is not None and state.get('semantic_decision_sha256') != digest(decision):
            raise ValueError('The authorized semantic decision changed on resume')
        if state['development_hash'] != digest(development):
            raise ValueError('Frozen development set has changed')
        if state.get('test_hash', digest([])) != digest(final_test):
            raise ValueError('Frozen final test set has changed')

        if config['evaluation'].get('semantic_judge'):
            from self_instruct_pipeline.parent_backend import create_parent
            from self_instruct_pipeline.usage import UsageLogger
            judge_dir = store.path('benchmark_judging')
            logger = UsageLogger(judge_dir)
            client = create_parent(parent_config['parent'], logger=logger)
            if hybrid:
                from self_instruct_pipeline.parallel import CachedParent
                client = CachedParent(client, judge_dir / 'request_cache', parent_config['parent'],
                                      requests_per_second=hybrid['requests_per_second'])
                evaluator.judge_workers = hybrid['workers']
            evaluator.semantic_judge = SemanticJudge(client, judge_dir, schemas)

        def controlled_release():
            from .paired import paired_summary

            selected = state.get('provisional_best_checkpoint')
            if not selected:
                raise ValueError('No acceptable provisional development checkpoint for release evaluation')
            release = Store(store.path('release'))
            checkpoints = {'initial': state['initial_checkpoint'], 'selected': selected}
            fingerprints = {name: adapter_files(checkpoint) for name, checkpoint in checkpoints.items()}
            identity = {'benchmark_hash': digest(final_test), **checkpoints, 'checkpoints': fingerprints,
                        'development_hash': digest(development), 'run': str(store.directory.resolve())}
            reservation_store = Store(resolve_path(saved_split).parent / 'release_reservations')
            reservation = reservation_store.path(identity['benchmark_hash'] + '.json')
            with reservation_store.lock():
                if reservation.exists() and json.loads(reservation.read_text()) != identity:
                    raise ValueError('This holdout has already been reserved for another release comparison')
                if not reservation.exists():
                    atomic_json(reservation, identity)
            if release_path.exists() and json.loads(release_path.read_text()) != identity:
                raise ValueError('The holdout was already reserved for a different checkpoint comparison')
            if not release_path.exists():
                atomic_json(release_path, identity)
            output = release.path('comparison.json')
            if output.exists():
                if json.loads(output.read_text())['identity'] != identity:
                    raise ValueError('Saved release comparison belongs to different inputs')
                return
            test_evaluator = LightevalEvaluator(runtime, schemas, 6)
            if hasattr(evaluator, 'semantic_judge'):
                # Keep holdout judgment data and its cache out of development state.
                test_evaluator.semantic_judge = SemanticJudge(
                    evaluator.semantic_judge.parent, release.path('judgments'), schemas)
            reports, scores = {}, {}
            for label, checkpoint in [('initial', identity['initial']), ('selected', selected)]:
                runtime.restore(checkpoint)
                test_evaluator.checkpoint_path = checkpoint
                folder = release.path(label)
                report = test_evaluator.evaluate(final_test, folder)
                validate_report(report, schemas, 6)
                reports[label] = report
                rows = read_jsonl(folder / 'predictions.jsonl')
                scores[label] = {row['scenario_id']: row['task_score'] for row in rows}
                if len(rows) != len(final_test) or scores[label].keys() != {item['record']['id'] for item in final_test}:
                    raise ValueError('Release predictions do not match the frozen holdout')
            changes = [(item, scores['selected'][item['record']['id']] - scores['initial'][item['record']['id']])
                       for item in final_test]
            paired = paired_summary([change for _, change in changes],
                                    [item['source_group'] for item, _ in changes],
                                    strata=[item['record']['task'] for item, _ in changes])
            per_task = {}
            for task in schemas:
                subset = [(item, change) for item, change in changes if item['record']['task'] == task]
                per_task[task] = paired_summary([change for _, change in subset],
                                               [item['source_group'] for item, _ in subset])
            atomic_json(output, {'identity': identity, **comparison(reports['initial'], reports['selected']),
                                 'paired': paired, 'per_task_paired': per_task, 'release_checkpoint': None,
                                 'release_eligible': False,
                                 'reference_status': 'synthetic silver; independent release evidence required',
                                 'limitations': ['Six cases per task; one case changes the task score by up to 16.7 points.',
                                                 '108 cases cannot reliably establish a one-point overall gain.',
                                                 'Use a larger, independently validated or human-reviewed benchmark for stronger release claims.']})

        def verify_release_binding(comparison_report, selected_checkpoint):
            benchmark_hash = digest(final_test)
            for label, checkpoint in (('baseline', state['initial_checkpoint']),
                                      ('selected', selected_checkpoint)):
                report = comparison_report[label]
                if (report.get('benchmark_hash') != benchmark_hash
                        or report.get('adapter_sha256') != checkpoint_binding(checkpoint)[1]):
                    raise ValueError('Saved release lacks verified benchmark or adapter identity')

        def finalize():
            if saved_split:
                if evaluate_release:
                    controlled_release()
                runtime.restore(state.get('release_checkpoint') or state['best_checkpoint'])
                return state
            reviewed_test = config['evaluation']['mode'] == 'reviewed'
            if final_test and 'final_comparison' not in state:
                print(f'Final holdout: evaluating the starting and selected adapters on '
                      f'{len(final_test)} cases each.', flush=True)
                minimum = (config['evaluation']['minimum_test_examples_per_task'] if reviewed_test else
                           min(sum(x['record']['task'] == task for x in final_test) for task in schemas))
                test_evaluator = LightevalEvaluator(runtime, schemas, minimum)
                if hasattr(evaluator, 'semantic_judge'):
                    test_evaluator.semantic_judge = evaluator.semantic_judge
                runtime.restore(state['initial_checkpoint'])
                test_evaluator.checkpoint_path = state['initial_checkpoint']
                baseline = test_evaluator.evaluate(final_test, store.path('final_test_baseline'))
                runtime.restore(state['best_checkpoint'])
                test_evaluator.checkpoint_path = state['best_checkpoint']
                selected = test_evaluator.evaluate(final_test, store.path('final_test_selected'))
                for report in (baseline, selected):
                    validate_report(report, schemas, minimum)
                final_comparison = comparison(baseline, selected)
                if reviewed_test:
                    promotion = promotion_gate(baseline, selected, quality, config['thresholds'], schemas)
                    if promotion['accepted']:
                        verify_release_binding(final_comparison, state['best_checkpoint'])
                    state['promotion'] = promotion
                    state['release_checkpoint'] = state['best_checkpoint'] if promotion['accepted'] else state['initial_checkpoint']
                state['final_comparison'] = final_comparison
                atomic_json(state_path, state)
            elif reviewed_test and final_test and state.get('promotion', {}).get('accepted'):
                saved = state['final_comparison']
                verify_release_binding(saved, state['release_checkpoint'])
                if not promotion_gate(saved['baseline'], saved['selected'], quality,
                                      config['thresholds'], schemas)['accepted']:
                    raise ValueError('Saved release lacks complete passing safety evidence')
            if not reviewed_test:
                state['promotion'] = {'accepted': False, 'reason': 'synthetic evaluation; no independent gold benchmark'}
                state['evaluation_status'] = 'automatically_evaluated'
                state['selected_checkpoint'] = state['best_checkpoint']
                atomic_json(state_path, state)
            runtime.restore(state.get('release_checkpoint', state['best_checkpoint']))
            return state

        print('Restoring student adapter (local).', flush=True)
        # Legacy runs may have saved the latest candidate as last_checkpoint.
        # The accepted best is the sole source of truth for a training resume.
        if state.get('next_round_start_checkpoint', state['best_checkpoint']) != state['best_checkpoint']:
            raise ValueError('Next-round checkpoint disagrees with the accepted best')
        state.update(next_round_start_checkpoint=state['best_checkpoint'], last_checkpoint=state['best_checkpoint'])
        runtime.restore(state['next_round_start_checkpoint'])
        best_report = state['best_report'] or {}
        if best_report.get('generation_key'):
            best_adapter = best_report.get('adapter_sha256')
            if not best_adapter:
                raise ValueError('Saved accepted evaluation lacks verified adapter-weight identity')
            if getattr(runtime, 'restored_adapter', None) != (
                    str(Path(state['best_checkpoint']).resolve()), best_adapter):
                raise ValueError('Accepted adapter differs from the evaluated checkpoint')
        if state['status'] != 'running':
            return finalize()

        # Evaluate once before training. A successful baseline needs no training.
        if state.get('baseline_report') is None:
            print(f'Starting baseline evaluation on {len(development)} examples.', flush=True)
            evaluator.checkpoint_path = state['initial_checkpoint']
            baseline_started = monotonic()
            report = evaluator.evaluate(development, store.path('baseline'))
            if hybrid:
                atomic_json(store.path('baseline/timing.json'),
                            {'evaluation_seconds': monotonic() - baseline_started})
            validate_report(report, schemas, config['development_examples_per_task'])
            state.update(latest_report=report, best_report=report, baseline_report=report)
            if passed(report):
                operational_gate(report, report, quality)
                state.update(provisional_best_checkpoint=state['initial_checkpoint'], provisional_best_report=report)
            if stop_on_thresholds and passed(report):
                state['status'] = 'thresholds_met'
            atomic_json(state_path, state)
            if stop_on_thresholds and passed(report):
                return finalize()

        exclusions = development + final_test
        if saved_split:
            exclusions = read_jsonl(resolve_path(config['evaluation']['validation_path'])) + loss_split['rows']
        elif config['evaluation']['mode'] == 'fixed':
            import copy
            full_config = copy.deepcopy(config)
            full_config['evaluation'].pop('total_examples', None)
            full_config['development_examples_per_task'] = 30
            exclusions = load_fixed_validation(full_config, schemas)
        data = CurriculumData(parent_config, exclusions, store.directory)
        if registry is not None:
            data.exclusion_registry = registry
        data.deterministic_fraction = quality.get('deterministic_template_fraction', 0.0)
        data.export_reference_pairs = not adaptive
        data.expanded_templates = config.get('balanced_curriculum', False)
        try:
            data.open()
            if hybrid:
                data.pipeline.request_options = hybrid
            for round_id in range(state['completed_rounds'] + 1, limit + 1):
                directory = store.path(f'round_{round_id:02d}')
                directory.mkdir(exist_ok=True)
                round_started = monotonic()
                print(f"\n=== Round {round_id}/{limit} ===", flush=True)

                # 1. Build toward a cumulative VERIFIED-pair target, replacing failures.
                candidate_feedback = state.get('curriculum_feedback', {})
                feedback_report = state['best_report']
                if adaptive:
                    scores = {task: 1 - stats['candidates_with_errors'] / stats['attempts']
                              for task, stats in candidate_feedback.items()
                              if max(stats['error_counts'].values(), default=0) >= 2}
                    feedback_report = {'per_task': {task: scores.get(task, state['best_report']['per_task'][task] if hybrid else 1.0) for task in schemas}}
                    data.pipeline.training_candidate_feedback = {task: candidate_feedback[task] for task in scores}
                weights = feedback_weights(data.base_weights, feedback_report, schemas,
                                           config['thresholds']['minimum_task_score'], config['feedback_strength'])
                if config.get('balanced_curriculum'):
                    weights = {task: min(3.0, max(1.0, weight)) for task, weight in weights.items()}
                target = (1 if hybrid else round_id) * config['scenarios_per_round']
                frozen_pairs = directory / 'training_plan.json'
                diagnostics = {}
                if frozen_pairs.exists():
                    plan = json.loads(frozen_pairs.read_text())
                    diagnostics = plan.get('diagnostics', {})
                    plan_inputs = [plan['pairs'], diagnostics] if adaptive else plan['pairs']
                    if plan['hash'] != digest(plan_inputs):
                        raise ValueError('Frozen round training pairs have changed')
                    pairs = [(x['record'], x['answer']) for x in plan['pairs']]
                elif adaptive:
                    previous_pairs = []
                    if round_id > 1 and not hybrid:
                        previous_plan = json.loads((store.path(f'round_{round_id - 1:02d}') /
                                                    'training_plan.json').read_text())
                        if previous_plan['hash'] != digest([previous_plan['pairs'], previous_plan['diagnostics']]):
                            raise ValueError('Previous round training plan has changed')
                        previous_pairs = [(x['record'], x['answer']) for x in previous_plan['pairs']]
                    def predict(student, records, schemas, folder):
                        if registry is not None:
                            registry.require_clean(records, 'student-conditioned screening')
                        return save_predictions(student, records, schemas, folder,
                                                checkpoint=state['next_round_start_checkpoint'])

                    if hybrid:
                        from .hybrid import hybrid_pairs
                        pairs, diagnostics = hybrid_pairs(
                            data, runtime, directory, state, config, weights, predict, historical)
                    else:
                        pairs, diagnostics = adaptive_pairs(
                            data, runtime, directory, state, config, weights, previous_pairs, predict)
                else:
                    pairs = data.verified_pairs(target, weights)
                if len({r['id'] for r, _ in pairs}) != len(pairs):
                    raise ValueError('Duplicate training scenario IDs')
                if registry is not None:
                    registry.require_clean((record for record, _ in pairs), 'training plan and replay')
                for record, answer in pairs:
                    validate_answer(record, answer, schemas[record['task']])
                previous = set(state.get('trained_scenario_ids',
                    [record['id'] for record, _ in pairs[:(round_id - 1) * config['scenarios_per_round']]]))
                new_pairs = [(record, answer) for record, answer in pairs if record['id'] not in previous]
                new_records = [record for record, _ in new_pairs]
                atomic_json(directory / 'generation_feedback.json', {'task_weights': weights,
                            **({'candidate_errors': candidate_feedback} if adaptive else {})})
                minimum = max(1, math.ceil(config['scenarios_per_round'] * quality.get('minimum_new_pair_fraction', .5)))
                coverage = training_coverage(pairs, target, len(new_pairs), minimum)
                atomic_json(directory / 'training_coverage.json', coverage)
                print(f"Training data: {len(pairs)}/{target} pairs; {len(new_pairs)} new; "
                      f"shortfall={coverage['shortfall']}; deterministic checks={coverage['deterministic_check_rate']:.1%}", flush=True)
                if coverage['deterministic_check_rate'] < quality.get('minimum_deterministic_check_rate', 0):
                    raise ValueError('Insufficient deterministic answer-check coverage')

                atomic_json(directory / 'training_filter.json', {
                    'target_verified_pairs': target, 'validated': len(pairs),
                    'target_reached': len(pairs) >= target,
                    'new_validated': len(new_pairs), 'minimum_new_required': minimum,
                    'training_scenario_ids': [record['id'] for record, _ in pairs]})
                if len(new_pairs) < minimum:
                    raise ValueError(f'Too few validated new training examples: {len(new_pairs)}; require at least {minimum}')
                if not frozen_pairs.exists():
                    rows = [{'record': r, 'answer': a} for r, a in pairs]
                    plan_inputs = [rows, diagnostics] if adaptive else rows
                    atomic_json(frozen_pairs, {'pairs': rows, 'hash': digest(plan_inputs),
                                **({'diagnostics': diagnostics} if adaptive else {})})
                if not adaptive:
                    save_predictions(runtime, new_records, schemas, directory)
                training_ids = [record['id'] for record, _ in pairs]
                replay_fraction = config['training'].get('replay_fraction', 0)
                if not config['training']['replay_previous_rounds']:
                    history_pairs = [pair for pair in pairs if pair[0]['id'] in previous]
                    replay = balanced_selection(list(reversed(history_pairs)),
                                                math.floor(len(new_pairs) * replay_fraction), initial=new_pairs)
                    pairs = new_pairs + replay
                if config.get('balanced_curriculum'):
                    from collections import Counter
                    atomic_json(directory / 'training_exposure.json', {
                        'new': dict(Counter(record['task'] for record, _ in new_pairs)),
                        'sft': dict(Counter(record['task'] for record, _ in pairs)),
                        'preference_pairs': len(diagnostics.get('preference_pairs', [])),
                    })

                # 4. Train the existing student adapter, then evaluate it.
                training_started = monotonic()
                trained_path = directory / 'training_complete.json'
                training_hash = digest({'pairs': pairs, 'initial': state['next_round_start_checkpoint'],
                                        'training': config['training'], 'student': runtime.identity()})
                sft_receipt = directory / 'sft_complete.json'
                loss_path = directory / 'heldout_teacher_loss.json'
                loss_inputs = None
                if loss_split and loss_split['rows']:
                    if not callable(getattr(runtime, 'completion_loss', None)):
                        raise ValueError('Selected runtime cannot measure held-out completion loss')
                    loss_inputs = [(row['record'], row['answer']) for row in loss_split['rows']]
                def record_loss(sft_checkpoint):
                    if loss_split is None:
                        return
                    identity = {'split_sha256': loss_split['split_sha256'], 'training_sha256': training_hash,
                                'before_checkpoint': state['next_round_start_checkpoint'], 'after_sft_checkpoint': sft_checkpoint}
                    if loss_path.exists():
                        if json.loads(loss_path.read_text())['identity'] != identity:
                            raise ValueError('Held-out loss belongs to different inputs')
                        return
                    baseline, after = None, None
                    if loss_inputs:
                        runtime.restore(state['next_round_start_checkpoint'])
                        baseline = runtime.completion_loss(loss_inputs, schemas)
                        runtime.restore(sft_checkpoint)
                        after = runtime.completion_loss(loss_inputs, schemas)
                    delta = None
                    if baseline and after and baseline['completion_loss'] is not None and after['completion_loss'] is not None:
                        delta = after['completion_loss'] - baseline['completion_loss']
                    atomic_json(loss_path, {'identity': identity, 'before': baseline, 'after_sft': after,
                                           'change': delta, 'per_task_examples': loss_split['per_task'],
                                           'excluded': loss_split['excluded'], 'limitations': loss_split['limitations']})

                if trained_path.exists():
                    trained = json.loads(trained_path.read_text())
                    if trained['training_hash'] != training_hash:
                        raise ValueError('Saved round training inputs changed; refusing to reuse checkpoint')
                    checkpoint = trained['checkpoint']
                    print('Reusing finished round training; retrying backup/evaluation.', flush=True)
                else:
                    print(f'Starting QLoRA training on {len(pairs)} pairs.', flush=True)
                    preferences = diagnostics.get('preference_pairs', []) if config.get('preference_training') else []
                    if sft_receipt.exists():
                        sft = json.loads(sft_receipt.read_text())
                        if sft['training_hash'] != training_hash:
                            raise ValueError('Saved SFT inputs changed')
                        checkpoint = sft['checkpoint']
                    else:
                        destination = directory / 'sft'
                        if (destination / 'adapter').exists():
                            destination = unused_path(destination)
                        checkpoint = runtime.train(pairs, schemas, destination)
                        atomic_json(sft_receipt, {'checkpoint': checkpoint, 'training_hash': training_hash})
                    record_loss(checkpoint)
                    if preferences:
                        runtime.restore(checkpoint)
                        destination = directory / 'preference'
                        if (destination / 'adapter').exists():
                            destination = unused_path(destination)
                        checkpoint = runtime.train_preferences(new_pairs, preferences, schemas, destination)
                    atomic_json(trained_path, {'checkpoint': checkpoint, 'training_hash': training_hash})
                    print(f'Checkpoint saved: {checkpoint}', flush=True)
                training_seconds = monotonic() - training_started
                if state.get('latest_checkpoint') != checkpoint:
                    state.update(latest_checkpoint=checkpoint, latest_report=None)
                    atomic_json(state_path, state)
                backup_adapter(backup_target, checkpoint, store.directory, round_id,
                               experimental=semantic_scoring or bool(saved_split))
                evaluator.checkpoint_path = checkpoint
                runtime.restore(checkpoint)
                evaluation_started = monotonic()
                report = evaluator.evaluate(development, directory / 'lighteval')
                validate_report(report, schemas, config['development_examples_per_task'])

                comparator = state['best_report']
                comparator_checkpoint = state['best_checkpoint']
                hard = operational_gate(comparator, report, quality)
                statistical = regression_gate(comparator, report, quality)
                improvement = report['macro_task_score'] - comparator['macro_task_score']
                minimum_gain = quality.get('minimum_macro_improvement', .01) if semantic_scoring else 0
                better = improvement > 1e-12 and improvement + 1e-12 >= minimum_gain
                promoted = hard['accepted'] and statistical['accepted'] and better
                reasons = list(hard['reasons'])
                if not statistical['accepted']:
                    losses = statistical.get('confirmed_regressions', statistical['regressions_exceeding_tolerance'])
                    reasons.append({'code': 'task_regression_screen', 'tasks': losses,
                                    'cases': sorted(key for key, row in report.get('case_scores', {}).items()
                                                    if row['task'] in losses and row['score'] <
                                                    comparator['case_scores'][key]['score'])})
                if not better:
                    reasons.append({'code': 'insufficient_macro_gain', 'observed': improvement,
                                    'minimum': minimum_gain,
                                    'cases': sorted(key for key, row in report.get('case_scores', {}).items()
                                                    if row['score'] < comparator['case_scores'][key]['score'])})
                selection = {'comparator_checkpoint': comparator_checkpoint, 'promoted': promoted,
                             'operational_safety': hard, 'statistical_regression': statistical,
                             'macro_improvement': improvement, 'minimum_macro_improvement': minimum_gain,
                             'quality_target_met': passed(report), 'release_eligible': False,
                             'reasons': reasons}
                if hybrid:
                    rank = state.get('experimental_macro_rank', [])
                    if not rank:
                        rank = [{'checkpoint': state['initial_checkpoint'],
                                 'macro': state['baseline_report']['macro_task_score']}]
                    rank.append({'checkpoint': checkpoint, 'macro': report['macro_task_score'],
                                 'operational_safety': hard, 'statistical_regression': statistical})
                    state['experimental_macro_rank'] = sorted(rank, key=lambda row: -row['macro'])
                    atomic_json(directory / 'experiment_timing.json', {
                        'evaluation_seconds': monotonic() - evaluation_started,
                        'training_or_restore_seconds': training_seconds,
                        'round_elapsed_seconds': monotonic() - round_started})
                if promoted:
                    state.update(best_report=report, best_checkpoint=checkpoint)
                    if passed(report):
                        state.update(provisional_best_checkpoint=checkpoint, provisional_best_report=report)
                next_checkpoint = state['best_checkpoint']
                state.update({key: value for key, value in diagnostics.items() if key != 'preference_pairs'})
                state.update(trained_scenario_ids=training_ids, completed_rounds=round_id,
                             last_checkpoint=next_checkpoint, next_round_start_checkpoint=next_checkpoint,
                             latest_checkpoint=checkpoint, latest_report=report)
                state['history'].append({'round': round_id, 'checkpoint': checkpoint, 'scores': report,
                                         'regression_gate': statistical, 'selection': selection, 'coverage': coverage})
                if 'case_scores' in report and 'case_scores' in state['baseline_report']:
                    baseline_cases = state['baseline_report']['case_scores']
                    atomic_json(directory / 'paired_case_changes.json', {
                        case_id: {**row, 'baseline': baseline_cases[case_id]['score']}
                        for case_id, row in report['case_scores'].items()
                        if row['score'] != baseline_cases[case_id]['score']
                    })
                if stop_on_thresholds and promoted and passed(report):
                    state['status'] = 'thresholds_met'
                elif round_id == limit:
                    state['status'] = 'max_rounds_reached'
                atomic_json(state_path, state)
                print(f"Round {round_id}/{limit} complete in {(monotonic() - round_started) / 60:.1f} min | "
                      f"development macro {report['macro_task_score']:.3f} | "
                      f"best {state['best_report']['macro_task_score']:.3f} | {state['status']}", flush=True)
                if state['status'] != 'running':
                    break
                runtime.restore(state['next_round_start_checkpoint'])
        finally:
            data.close()

        return finalize()
