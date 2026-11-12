"""One bounded loop: generate, infer, get teacher targets, train, evaluate."""
import json
import math
import os
from time import monotonic

from self_instruct_pipeline.config import resolve_path
from self_instruct_pipeline.persistence import Store, atomic_json, digest
from self_instruct_pipeline.prompt_builder import target_payload
from self_instruct_pipeline.validation import ValidationError, validate_answer
from .core import ResponseError, clean_response, feedback_weights, satisfactory, validate_report
from .correction import adaptive_pairs, balanced_selection
from .data import CurriculumData, frozen_development
from .evaluation import LightevalEvaluator
from .benchmarks import load_benchmarks, load_fixed_validation, comparison
from .review import SCORING_VERSION
from .trusted_templates import TEMPLATE_VERSION
from .quality import validate_quality_settings, regression_gate, promotion_gate, training_coverage
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


def run_loop(runtime, parent_config, config, *, round_limit=None, initial_adapter=None):
    # Per-call journals are already saved, even if training/evaluation fails.
    try:
        return _run_loop(runtime, parent_config, config, round_limit=round_limit, initial_adapter=initial_adapter)
    finally:
        from self_instruct_pipeline.usage import export_usage
        directory = resolve_path(config['output_dir'])
        if directory.exists():
            print('API usage:', export_usage(directory), flush=True)


def _run_loop(runtime, parent_config, config, *, round_limit=None, initial_adapter=None):
    store = Store(resolve_path(config['output_dir']))
    check_backup_setup()
    schemas = json.loads(resolve_path(parent_config['paths']['schemas']).read_text())['tasks']
    settings_hash = digest({'config': config, 'parent': parent_config,
                            'schemas': schemas, 'student': runtime.identity(), 'scoring': SCORING_VERSION, 'trusted_templates': TEMPLATE_VERSION})
    if config['evaluation'].get('semantic_judge'):
        from .semantic import SemanticJudge
        settings_hash = digest([settings_hash, SemanticJudge.version])
    state_path = store.path('loop_state.json')
    quality = validate_quality_settings(config, schemas)
    adaptive = config.get('student_conditioned_training', False)
    stop_on_thresholds = config.get('stop_when_thresholds_met', True)
    if type(adaptive) is not bool:
        raise ValueError('student_conditioned_training must be a boolean')

    def passed(report):
        return satisfactory(report, config['thresholds'], schemas)

    def rank(report):
        return passed(report), report['macro_task_score'], report['minimum_task_score']

    with store.lock():
        reviewed, final_test = load_benchmarks(config, schemas)
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state and state.get('settings_hash') != settings_hash:
            raise ValueError('Settings changed; use a new output directory')
        development = reviewed if reviewed is not None else frozen_development(parent_config, config, store.directory)
        if state and (state['development_hash'] != digest(development) or state.get('test_hash') != digest(final_test)):
            raise ValueError('Frozen evaluation data have changed')
        if final_test and config['evaluation']['mode'] == 'fixed':
            split = {'dataset': config['evaluation']['validation_path'],
                     'seed': config['evaluation'].get('subset_seed', config['random_seed'])}
            for name, rows in [('development', development), ('holdout', final_test)]:
                split[name] = {'sha256': digest(rows),
                               'case_ids': [item['record']['id'] for item in rows],
                               'source_groups': sorted({item['source_group'] for item in rows})}
            manifest_path = store.path('evaluation_split.json')
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != split:
                raise ValueError('Frozen evaluation split manifest has changed')
            if not manifest_path.exists():
                atomic_json(manifest_path, split)
        previous_limit = state.get('round_limit', config['max_rounds']) if state else config['max_rounds']
        limit = previous_limit if round_limit is None else round_limit
        if type(limit) is not int or limit < previous_limit:
            raise ValueError('Round limit must be an integer at least as large as the saved limit')
        if state and limit > previous_limit:
            if (store.path('final_test_baseline').exists()
                    or store.path('final_test_selected').exists()):
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
            initial = runtime.save_checkpoint(store.path('initial_adapter'))
            state = {
                'settings_hash': settings_hash, 'development_hash': digest(development),
                'test_hash': digest(final_test),
                'completed_rounds': 0, 'history': [], 'status': 'running', 'round_limit': limit,
                'initial_checkpoint': initial, 'last_checkpoint': initial, 'best_checkpoint': initial,
                'best_report': None, 'latest_report': None,
            }
            atomic_json(state_path, state)
        if state['development_hash'] != digest(development):
            raise ValueError('Frozen development set has changed')
        if state.get('test_hash', digest([])) != digest(final_test):
            raise ValueError('Frozen final test set has changed')

        if config['evaluation'].get('semantic_judge'):
            from .semantic import SemanticJudge
            from self_instruct_pipeline.parent_backend import create_parent
            from self_instruct_pipeline.usage import UsageLogger
            judge_dir = store.path('benchmark_judging')
            logger = UsageLogger(judge_dir)
            client = create_parent(parent_config['parent'], logger=logger)
            evaluator.semantic_judge = SemanticJudge(client, judge_dir, schemas)

        def finalize():
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
                test_evaluator.checkpoint_identity = state['initial_checkpoint']
                baseline = test_evaluator.evaluate(final_test, store.path('final_test_baseline'))
                runtime.restore(state['best_checkpoint'])
                test_evaluator.checkpoint_identity = state['best_checkpoint']
                selected = test_evaluator.evaluate(final_test, store.path('final_test_selected'))
                for report in (baseline, selected):
                    validate_report(report, schemas, minimum)
                state['final_comparison'] = comparison(baseline, selected)
                if reviewed_test:
                    state['promotion'] = promotion_gate(baseline, selected, quality, config['thresholds'], schemas)
                    state['release_checkpoint'] = state['best_checkpoint'] if state['promotion']['accepted'] else state['initial_checkpoint']
                atomic_json(state_path, state)
            if not reviewed_test:
                state['promotion'] = {'accepted': False, 'reason': 'synthetic evaluation; no independent gold benchmark'}
                state['evaluation_status'] = 'automatically_evaluated'
                state['selected_checkpoint'] = state['best_checkpoint']
                atomic_json(state_path, state)
            runtime.restore(state.get('release_checkpoint', state['best_checkpoint']))
            return state

        print('Restoring student adapter (local).', flush=True)
        runtime.restore(state['last_checkpoint'])
        if state['status'] != 'running':
            return finalize()

        # Evaluate once before training. A successful baseline needs no training.
        if state['latest_report'] is None:
            print(f'Starting baseline evaluation on {len(development)} examples.', flush=True)
            evaluator.checkpoint_identity = state['initial_checkpoint']
            report = evaluator.evaluate(development, store.path('baseline'))
            validate_report(report, schemas, config['development_examples_per_task'])
            state.update(latest_report=report, best_report=report, baseline_report=report)
            if stop_on_thresholds and passed(report):
                state['status'] = 'thresholds_met'
            atomic_json(state_path, state)
            if stop_on_thresholds and passed(report):
                return finalize()

        exclusions = development + final_test
        if config['evaluation']['mode'] == 'fixed':
            import copy
            full_config = copy.deepcopy(config)
            full_config['evaluation'].pop('total_examples', None)
            full_config['development_examples_per_task'] = 30
            exclusions = load_fixed_validation(full_config, schemas)
        data = CurriculumData(parent_config, exclusions, store.directory)
        data.deterministic_fraction = quality.get('deterministic_template_fraction', 0.0)
        data.export_reference_pairs = not adaptive
        data.expanded_templates = config.get('balanced_curriculum', False)
        try:
            data.open()
            for round_id in range(state['completed_rounds'] + 1, limit + 1):
                directory = store.path(f'round_{round_id:02d}')
                directory.mkdir(exist_ok=True)
                round_started = monotonic()
                print(f"\n=== Round {round_id}/{limit} ===", flush=True)

                # 1. Build toward a cumulative VERIFIED-pair target, replacing failures.
                candidate_feedback = state.get('curriculum_feedback', {})
                feedback_report = state['latest_report']
                if adaptive:
                    scores = {task: 1 - stats['candidates_with_errors'] / stats['attempts']
                              for task, stats in candidate_feedback.items()
                              if max(stats['error_counts'].values(), default=0) >= 2}
                    feedback_report = {'per_task': {task: scores.get(task, 1.0) for task in schemas}}
                    data.pipeline.training_candidate_feedback = {task: candidate_feedback[task] for task in scores}
                weights = feedback_weights(data.base_weights, feedback_report, schemas,
                                           config['thresholds']['minimum_task_score'], config['feedback_strength'])
                if config.get('balanced_curriculum'):
                    weights = {task: min(3.0, max(1.0, weight)) for task, weight in weights.items()}
                target = round_id * config['scenarios_per_round']
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
                    if round_id > 1:
                        previous_plan = json.loads((store.path(f'round_{round_id - 1:02d}') /
                                                    'training_plan.json').read_text())
                        if previous_plan['hash'] != digest([previous_plan['pairs'], previous_plan['diagnostics']]):
                            raise ValueError('Previous round training plan has changed')
                        previous_pairs = [(x['record'], x['answer']) for x in previous_plan['pairs']]
                    pairs, diagnostics = adaptive_pairs(
                        data, runtime, directory, state, config, weights, previous_pairs,
                        lambda student, records, schemas, folder: save_predictions(
                            student, records, schemas, folder, checkpoint=state['last_checkpoint']))
                else:
                    pairs = data.verified_pairs(target, weights)
                if len({r['id'] for r, _ in pairs}) != len(pairs):
                    raise ValueError('Duplicate training scenario IDs')
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
                    rejected_rounds = {entry['round'] for entry in state['history']
                                       if not entry['regression_gate']['accepted']} if replay_fraction else set()
                    if rejected_rounds:
                        rejected_ids = set()
                        for rejected_round in rejected_rounds:
                            rows = json.loads((store.path(f'round_{rejected_round:02d}') /
                                               'training_plan.json').read_text())['pairs']
                            start = (rejected_round - 1) * config['scenarios_per_round']
                            rejected_ids.update(row['record']['id'] for row in rows[start:start + config['scenarios_per_round']])
                        history_pairs = [pair for pair in history_pairs if pair[0]['id'] not in rejected_ids]
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
                trained_path = directory / 'training_complete.json'
                training_hash = digest({'pairs': pairs, 'initial': state['last_checkpoint'],
                                        'training': config['training'], 'student': runtime.identity()})
                if trained_path.exists():
                    trained = json.loads(trained_path.read_text())
                    if trained['training_hash'] != training_hash:
                        raise ValueError('Saved round training inputs changed; refusing to reuse checkpoint')
                    checkpoint = trained['checkpoint']
                    runtime.restore(checkpoint)
                    print('Reusing finished round training; retrying backup/evaluation.', flush=True)
                else:
                    print(f'Starting QLoRA training on {len(pairs)} pairs.', flush=True)
                    preferences = diagnostics.get('preference_pairs', [])
                    sft_receipt = directory / 'sft_complete.json'
                    if preferences and sft_receipt.exists():
                        sft = json.loads(sft_receipt.read_text())
                        if sft['training_hash'] != training_hash:
                            raise ValueError('Saved SFT inputs changed')
                        checkpoint = sft['checkpoint']
                    else:
                        checkpoint = runtime.train(pairs, schemas, directory / 'sft')
                    if preferences:
                        atomic_json(sft_receipt, {'checkpoint': checkpoint, 'training_hash': training_hash})
                        runtime.restore(checkpoint)
                        checkpoint = runtime.train_preferences(
                            new_pairs, diagnostics['preference_pairs'], schemas, directory / 'preference')
                    atomic_json(trained_path, {'checkpoint': checkpoint, 'training_hash': training_hash})
                    print(f'Checkpoint saved: {checkpoint}', flush=True)
                if os.environ.get('DISTILLATION_DRIVE_BACKUP') == '1':
                    from .drive_backup import save_drive_adapter
                    save_drive_adapter(checkpoint, store.directory, round_id)
                elif os.environ.get('GITHUB_ADAPTER_UPLOAD') == '1':
                    from .github_backup import upload_latest_adapter
                    upload_latest_adapter(store.directory)
                else:
                    save_adapter_dataset(checkpoint, store.directory, round_id)
                evaluator.checkpoint_identity = checkpoint
                report = evaluator.evaluate(development, directory / 'lighteval')
                validate_report(report, schemas, config['development_examples_per_task'])

                # Save only completed rounds. An interrupted round restarts from
                # the previous completed adapter when this function is rerun.
                state.update(diagnostics)
                state.update(trained_scenario_ids=training_ids, completed_rounds=round_id, last_checkpoint=checkpoint, latest_report=report)
                validation_safety = regression_gate(state['baseline_report'], report, quality)
                state['history'].append({'round': round_id, 'checkpoint': checkpoint, 'scores': report, 'regression_gate': validation_safety, 'coverage': coverage})
                if validation_safety['accepted'] and rank(report) > rank(state['best_report']):
                    state.update(best_report=report, best_checkpoint=checkpoint)
                if config.get('rollback_on_regression') and not validation_safety['accepted']:
                    state['last_checkpoint'] = state['best_checkpoint']
                    runtime.restore(state['last_checkpoint'])
                if 'case_scores' in report:
                    baseline_cases = state['baseline_report']['case_scores']
                    atomic_json(directory / 'paired_case_changes.json', {
                        case_id: {**row, 'baseline': baseline_cases[case_id]['score']}
                        for case_id, row in report['case_scores'].items()
                        if row['score'] != baseline_cases[case_id]['score']
                    })
                if stop_on_thresholds and passed(report) and validation_safety['accepted']:
                    state['status'] = 'thresholds_met'
                elif round_id == limit:
                    state['status'] = 'max_rounds_reached'
                atomic_json(state_path, state)
                print(f"Round {round_id}/{limit} complete in {(monotonic() - round_started) / 60:.1f} min | "
                      f"development macro {report['macro_task_score']:.3f} | "
                      f"best {state['best_report']['macro_task_score']:.3f} | {state['status']}", flush=True)
                if state['status'] != 'running':
                    break
        finally:
            data.close()

        return finalize()
