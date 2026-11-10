"""One bounded loop: generate, infer, get teacher targets, train, evaluate."""
import json

from self_instruct_pipeline.config import resolve_path
from self_instruct_pipeline.persistence import Store, atomic_json, digest
from self_instruct_pipeline.prompt_builder import target_payload
from self_instruct_pipeline.validation import ValidationError, validate_answer
from .core import ResponseError, clean_response, feedback_weights, satisfactory, validate_report
from .data import CurriculumData, frozen_development
from .evaluation import LightevalEvaluator
from .benchmarks import load_benchmarks, comparison
from .student import infer_payloads
from .kaggle_backup import check_backup_setup, save_adapter_dataset


def save_predictions(runtime, records, schemas, directory):
    store = Store(directory)
    existing = {row['scenario_id']: row for row in
                store.unique_records('student_responses.jsonl', 'scenario_id')}
    pending = []
    for record in records:
        if record['id'] in existing:
            if existing[record['id']]['scenario_hash'] != digest(record):
                raise ValueError('Student prediction belongs to a changed scenario')
        else:
            pending.append(record)
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
        store.append('student_responses.jsonl', {
            'scenario_id': record['id'], 'scenario_hash': digest(record), 'task': record['task'],
            'raw': prediction['raw'], 'cleaned': cleaned,
            'schema_evidence_valid': cleaned is not None,
            'truncated': prediction['truncated'], 'error': error,
            'purpose': 'diagnostic only; never an SFT target',
        })


def run_loop(runtime, parent_config, config):
    # Per-call journals are already saved, even if training/evaluation fails.
    try:
        return _run_loop(runtime, parent_config, config)
    finally:
        from self_instruct_pipeline.usage import export_usage
        directory = resolve_path(config['output_dir'])
        if directory.exists():
            print('API usage:', export_usage(directory), flush=True)


def _run_loop(runtime, parent_config, config):
    store = Store(resolve_path(config['output_dir']))
    check_backup_setup()
    schemas = json.loads(resolve_path(parent_config['paths']['schemas']).read_text())['tasks']
    settings_hash = digest({'config': config, 'parent': parent_config,
                            'schemas': schemas, 'student': runtime.identity()})
    state_path = store.path('loop_state.json')

    def passed(report):
        return satisfactory(report, config['thresholds'], schemas)

    def rank(report):
        return passed(report), report['macro_task_score'], report['minimum_task_score']

    with store.lock():
        reviewed, final_test = load_benchmarks(config, schemas)
        runtime.prepare()
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state and state.get('settings_hash') != settings_hash:
            raise ValueError('Settings changed; use a new output directory')
        development = reviewed if reviewed is not None else frozen_development(parent_config, config, store.directory)
        evaluator = LightevalEvaluator(runtime, schemas, config['development_examples_per_task'])

        if state is None:
            print('Development set complete. Saving initial student adapter (local; no teacher API calls).', flush=True)
            initial = runtime.save_checkpoint(store.path('initial_adapter'))
            state = {
                'settings_hash': settings_hash, 'development_hash': digest(development),
                'test_hash': digest(final_test),
                'completed_rounds': 0, 'history': [], 'status': 'running',
                'last_checkpoint': initial, 'best_checkpoint': initial,
                'best_report': None, 'latest_report': None,
            }
            atomic_json(state_path, state)
        if state['development_hash'] != digest(development):
            raise ValueError('Frozen development set has changed')
        if state.get('test_hash', digest([])) != digest(final_test):
            raise ValueError('Frozen final test set has changed')

        def finalize():
            if final_test and 'final_comparison' not in state:
                test_evaluator = LightevalEvaluator(runtime, schemas, config['evaluation']['minimum_test_examples_per_task'])
                runtime.restore(str(store.path('initial_adapter')))
                baseline = test_evaluator.evaluate(final_test, store.path('final_test_baseline'))
                runtime.restore(state['best_checkpoint'])
                selected = test_evaluator.evaluate(final_test, store.path('final_test_selected'))
                for report in (baseline, selected):
                    validate_report(report, schemas, config['evaluation']['minimum_test_examples_per_task'])
                state['final_comparison'] = comparison(baseline, selected)
                atomic_json(state_path, state)
            runtime.restore(state['best_checkpoint'])
            return state

        print('Restoring student adapter (local).', flush=True)
        runtime.restore(state['last_checkpoint'])
        if state['status'] != 'running':
            return finalize()

        # Evaluate once before training. A successful baseline needs no training.
        if state['latest_report'] is None:
            print(f'Starting baseline evaluation on {len(development)} examples (local GPU; no teacher API calls).', flush=True)
            report = evaluator.evaluate(development, store.path('baseline'))
            validate_report(report, schemas, config['development_examples_per_task'])
            state.update(latest_report=report, best_report=report)
            if passed(report):
                state['status'] = 'thresholds_met'
            atomic_json(state_path, state)
            if passed(report):
                return finalize()

        data = CurriculumData(parent_config, development + final_test, store.directory)
        try:
            data.open()
            for round_id in range(state['completed_rounds'] + 1, config['max_rounds'] + 1):
                directory = store.path(f'round_{round_id:02d}')
                directory.mkdir(exist_ok=True)
                print(f"Round {round_id}/{config['max_rounds']}", flush=True)

                # 1. Build toward a cumulative VERIFIED-pair target, replacing failures.
                weights = feedback_weights(data.base_weights, state['latest_report'], schemas,
                                           config['thresholds']['minimum_task_score'], config['feedback_strength'])
                target = round_id * config['scenarios_per_round']
                pairs = data.verified_pairs(target, weights)
                previous = set(state.get('trained_scenario_ids',
                    [record['id'] for record, _ in pairs[:(round_id - 1) * config['scenarios_per_round']]]))
                new_pairs = [(record, answer) for record, answer in pairs if record['id'] not in previous]
                new_records = [record for record, _ in new_pairs]
                atomic_json(directory / 'generation_feedback.json', {'task_weights': weights})
                from collections import Counter
                atomic_json(directory / 'training_coverage.json', {
                    'tasks': dict(Counter(r['task'] for r, _ in pairs)),
                    'case_types': dict(Counter(r.get('case_type', 'unlabeled') for r, _ in pairs)),
                    'difficulty': dict(Counter(r.get('difficulty', 'unlabeled') for r, _ in pairs)),
                    'distinct_scenarios': len({r['id'] for r, _ in pairs}),
                    'reference_status': 'automatically checked silver; teacher approval is not independent truth'})

                # 2. Student predictions remain diagnostics, never training labels.
                save_predictions(runtime, new_records, schemas, directory)
                minimum = max(1, (config['scenarios_per_round'] + 1) // 2)
                atomic_json(directory / 'training_filter.json', {
                    'target_verified_pairs': target, 'validated': len(pairs),
                    'target_reached': len(pairs) >= target,
                    'new_validated': len(new_pairs), 'minimum_new_required': minimum,
                    'training_scenario_ids': [record['id'] for record, _ in pairs]})
                if len(new_pairs) < minimum:
                    raise ValueError(f'Too few validated new training examples: {len(new_pairs)}; require at least {minimum}')
                training_ids = [record['id'] for record, _ in pairs]
                if not config['training']['replay_previous_rounds']:
                    pairs = new_pairs

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
                    checkpoint = runtime.train(pairs, schemas, directory / 'sft')
                    atomic_json(trained_path, {'checkpoint': checkpoint, 'training_hash': training_hash})
                save_adapter_dataset(checkpoint, store.directory, round_id)
                report = evaluator.evaluate(development, directory / 'lighteval')
                validate_report(report, schemas, config['development_examples_per_task'])

                # Save only completed rounds. An interrupted round restarts from
                # the previous completed adapter when this function is rerun.
                state.update(trained_scenario_ids=training_ids, completed_rounds=round_id, last_checkpoint=checkpoint, latest_report=report)
                state['history'].append({'round': round_id, 'checkpoint': checkpoint, 'scores': report})
                if rank(report) > rank(state['best_report']):
                    state.update(best_report=report, best_checkpoint=checkpoint)
                if passed(report):
                    state['status'] = 'thresholds_met'
                elif round_id == config['max_rounds']:
                    state['status'] = 'max_rounds_reached'
                atomic_json(state_path, state)
                print(f"Lighteval macro={report['macro_task_score']:.3f}; {state['status']}", flush=True)
                if state['status'] != 'running':
                    break
        finally:
            data.close()

        return finalize()
