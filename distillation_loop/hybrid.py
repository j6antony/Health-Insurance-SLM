"""Fresh, versioned curriculum with a bounded student screening window."""
from collections import Counter
from time import monotonic
import json

from self_instruct_pipeline.parallel import bounded_map
from self_instruct_pipeline.persistence import atomic_json, digest, read_jsonl

from .correction import assess_candidate, balanced_selection, curriculum_feedback
from .data import fresh_teacher_pool
from .leakage import ExclusionRegistry
from .trusted_templates import fresh_template_pairs


def options(config):
    settings = config.get('hybrid_curriculum')
    if settings is None:
        return None
    keys = ('deterministic', 'general', 'adaptive', 'buffer', 'workers', 'batch_size', 'requests_per_second')
    if (set(settings) != set(keys) or any(type(settings[k]) is not int for k in keys)
            or min(settings[k] for k in ('deterministic', 'general', 'adaptive')) < 1
            or sum(settings[k] for k in keys[:3]) != config['scenarios_per_round']
            or not 0 <= settings['buffer'] <= 20
            or not 1 <= settings['workers'] <= 4 or not 1 <= settings['batch_size'] <= 4
            or not 1 <= settings['requests_per_second'] <= 4
            or not config.get('student_conditioned_training') or config.get('preference_training')
            or config['training']['replay_previous_rounds']
            or not config.get('rollback_on_regression')):
        raise ValueError('Invalid hybrid curriculum counts, request limits, or training safeguards')
    return settings


def hybrid_pairs(data, runtime, directory, state, config, weights, predict, historical):
    started = monotonic()
    settings = options(config)
    pipe = data.pipeline
    round_id = state['completed_rounds'] + 1
    seed = f"{config['random_seed']}-{round_id}"
    history = []
    for path in sorted(directory.parent.glob('round_*/training_plan.json')):
        if path.parent == directory:
            continue
        plan = json.loads(path.read_text())
        if digest([plan['pairs'], plan['diagnostics']]) != plan['hash']:
            raise ValueError('Historical experiment training plan changed')
        history.extend(row['record'] for row in plan['pairs'])
    previous = ExclusionRegistry(historical + history, near_duplicates=False)
    previous_teacher = ExclusionRegistry([r for r in historical + history if not r.get('verification_basis')])
    benchmark = data.exclusion_registry
    def excluded(record):
        return previous.match(record) or benchmark.match(record) or pipe.exclusion_dedup.check(record)

    def fits(pair):
        return not hasattr(runtime, 'training_pair_fits') or runtime.training_pair_fits(*pair, pipe.schemas)

    deterministic = fresh_template_pairs(settings['deterministic'], seed, pipe.schemas, excluded)
    if not all(map(fits, deterministic)):
        raise ValueError('A deterministic training target exceeds the training token budget')
    fixed = ExclusionRegistry([r for r, _ in deterministic], near_duplicates=False)
    path = directory / 'hybrid_pool.json'
    if path.exists():
        progress = json.loads(path.read_text())
    else:
        progress = {'older_ids': [r['id'] for r in pipe.accepted], 'general': [], 'windows': []}
        atomic_json(path, progress)

    def eligible(pair):
        return (not excluded(pair[0]) and not previous_teacher.match(pair[0])
                and not fixed.match(pair[0]) and fits(pair))

    old = set(progress['older_ids'])
    def fetch(target, sampling_weights):
        return fresh_teacher_pool(pipe, target, sampling_weights, old, eligible=eligible)

    if not progress['general']:
        pool = fetch(settings['general'], dict.fromkeys(pipe.schemas, 1))
        general = balanced_selection(pool, settings['general'], initial=deterministic)
        if len(general) != settings['general']:
            raise ValueError('Insufficient fresh verified general targets')
        progress['general'] = [r['id'] for r, _ in general]
        atomic_json(path, progress)
    pool = {r['id']: (r, a) for r, a in fetch(0, weights)}
    general = [pool[key] for key in progress['general']]
    general_elapsed = monotonic() - started
    ready = deterministic + general
    results = {}
    used = set(progress['general'])
    floor = 2

    for attempt in range(5):
        if attempt < len(progress['windows']):
            identifiers = progress['windows'][attempt]
        else:
            selected = select_adaptive(results, settings['adaptive'], ready)
            screened = Counter(r['scenario']['task'] for r in results.values())
            deficits = {t: max(0, floor - screened[t]) for t in pipe.schemas}
            missing = max(settings['adaptive'] - len(selected), sum(deficits.values()))
            if not missing:
                break
            window = missing + settings['buffer']
            refill_weights = {t: float(weights[t]) * (3 if deficits[t] else 1) for t in pipe.schemas}
            pool = {r['id']: (r, a) for r, a in fetch(len(used) + window, refill_weights)}
            pending = [pair for key, pair in pool.items() if key not in used]
            reserved = []
            for pair in pending:
                task = pair[0]['task']
                if deficits[task]:
                    reserved.append(pair)
                    deficits[task] -= 1
            reserved = reserved[:window]
            reserved_ids = {r['id'] for r, _ in reserved}
            remainder = [pair for pair in pending if pair[0]['id'] not in reserved_ids]
            remainder.sort(key=lambda pair: (
                -refill_weights[pair[0]['task']], {'hard': 0, 'medium': 1, 'easy': 2}[pair[0]['difficulty']]))
            chosen = reserved + remainder[:window - len(reserved)]
            identifiers = [r['id'] for r, _ in chosen]
            if not identifiers:
                break
            progress['windows'].append(identifiers)
            atomic_json(path, progress)
        pairs = [pool[key] for key in identifiers]
        prompts = [r for r, _ in pairs]
        predictions = predict(runtime, prompts, pipe.schemas, directory)
        pipe.client()  # Initialize the shared client on the caller thread.
        def assess(pair):
            record, reference = pair
            return assess_candidate(pipe, record, reference, predictions[record['id']],
                                    directory / 'corrections', state['last_checkpoint'])
        for row in bounded_map(assess, pairs, settings['workers']):
            results[row['scenario']['id']] = row
        used.update(identifiers)

    adaptive = select_adaptive(results, settings['adaptive'], ready)
    screened = Counter(r['scenario']['task'] for r in results.values())
    selected = ready + adaptive
    coverage = Counter(r['task'] for r, _ in selected)
    if (len(adaptive) != settings['adaptive'] or any(coverage[t] < 2 for t in pipe.schemas)
            or any(screened[t] < floor for t in pipe.schemas)):
        raise ValueError(f'Hybrid curriculum coverage incomplete: selected={dict(coverage)}, screened={dict(screened)}')
    unique = ExclusionRegistry([], near_duplicates=False)
    for record, _ in selected:
        if unique.match(record):
            raise ValueError('Duplicate or shared source family in the hybrid round')
        unique.add(record)
    atomic_json(directory / 'curriculum_timing.json', {
        'general_and_deterministic_seconds': general_elapsed,
        'adaptive_seconds': monotonic() - started - general_elapsed})
    return selected, {'curriculum_version': 'hybrid-v1', 'source_seed': seed,
                      'training_mix': {k: settings[k] for k in ('deterministic', 'general', 'adaptive')},
                      'screened_per_task': dict(screened),
                      'curriculum_feedback': curriculum_feedback(results.values())}


def select_adaptive(results, count, initial):
    priority = {'needs_correction': 0, 'assessment_unavailable': 1, 'correct': 2}
    rows = [row for row in results.values() if row['status'] == 'accepted']
    rows.sort(key=lambda row: (priority[row['assessment']['verdict']],
                               {'hard': 0, 'medium': 1, 'easy': 2}[row['scenario']['difficulty']]))
    return balanced_selection([(r['scenario'], r['reference']) for r in rows], count, initial=initial)
