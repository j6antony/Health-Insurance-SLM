"""One bounded loop: generate, infer, get teacher targets, train, evaluate."""
import json

from self_instruct_pipeline.config import resolve_path
from self_instruct_pipeline.persistence import Store, atomic_json, digest
from self_instruct_pipeline.prompt_builder import target_payload
from self_instruct_pipeline.validation import ValidationError, validate_answer
from .core import ResponseError, clean_response, feedback_weights, satisfactory, validate_report
from .data import CurriculumData, frozen_development
from .evaluation import LightevalEvaluator


def save_predictions(runtime, records, schemas, directory):
    store = Store(directory)
    existing = {row['scenario_id']: row for row in
                store.unique_records('student_responses.jsonl', 'scenario_id')}
    for record in records:
        if record['id'] in existing:
            if existing[record['id']]['scenario_hash'] != digest(record):
                raise ValueError('Student prediction belongs to a changed scenario')
            continue
        prediction = runtime.infer(target_payload(record, schemas))
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
    schemas = json.loads(resolve_path(parent_config['paths']['schemas']).read_text())['tasks']
    settings_hash = digest({'config': config, 'parent': parent_config,
                            'schemas': schemas, 'student': runtime.identity()})
    state_path = store.path('loop_state.json')

    def passed(report):
        return satisfactory(report, config['thresholds'], schemas)

    def rank(report):
        return passed(report), report['macro_task_score'], report['minimum_task_score']

    with store.lock():
        runtime.prepare()
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state and state.get('settings_hash') != settings_hash:
            raise ValueError('Settings changed; use a new output directory')
        development = frozen_development(parent_config, config, store.directory)
        evaluator = LightevalEvaluator(runtime, schemas, config['development_examples_per_task'])

        if state is None:
            print('Development set complete. Saving initial student adapter (local; no teacher API calls).', flush=True)
            initial = runtime.save_checkpoint(store.path('initial_adapter'))
            state = {
                'settings_hash': settings_hash, 'development_hash': digest(development),
                'completed_rounds': 0, 'history': [], 'status': 'running',
                'last_checkpoint': initial, 'best_checkpoint': initial,
                'best_report': None, 'latest_report': None,
            }
            atomic_json(state_path, state)
        if state['development_hash'] != digest(development):
            raise ValueError('Frozen development set has changed')
        print('Restoring student adapter (local).', flush=True)
        runtime.restore(state['last_checkpoint'])
        if state['status'] != 'running':
            runtime.restore(state['best_checkpoint'])
            return state

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
                return state

        data = CurriculumData(parent_config, development, store.directory)
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
                checkpoint = runtime.train(pairs, schemas, directory / 'sft')
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

        runtime.restore(state['best_checkpoint'])
        return state
