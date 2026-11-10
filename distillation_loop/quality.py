"""Model promotion gates and explicit data shortfall accounting."""
from collections import Counter
import math


def regression_gate(baseline, candidate, settings):
    tolerances = settings.get('per_task_max_regression', {})
    default = settings.get('max_task_regression', 0.0)
    failures = {task: baseline['per_task'][task] - candidate['per_task'][task]
                for task in baseline['per_task']
                if baseline['per_task'][task] - candidate['per_task'][task] > tolerances.get(task, default) + 1e-12}
    return {'accepted': not failures, 'regressions_exceeding_tolerance': failures}


def promotion_gate(baseline, candidate, settings, thresholds, tasks):
    from .core import satisfactory
    result = regression_gate(baseline, candidate, settings)
    improvement = candidate['macro_task_score'] - baseline['macro_task_score']
    result.update(macro_improvement=improvement,
                  thresholds_met=satisfactory(candidate, thresholds, tasks))
    result['accepted'] = (result['accepted'] and result['thresholds_met']
                          and improvement + 1e-12 >= settings.get('minimum_macro_improvement', 0.01))
    return result


def training_coverage(pairs, target, new_count, minimum_new):
    tasks = Counter(r['task'] for r, _ in pairs)
    checked = Counter(r['task'] for r, _ in pairs if r.get('answer_checks'))
    return {'target_verified_pairs': target, 'achieved_pairs': len(pairs),
            'shortfall': max(0, target - len(pairs)), 'target_reached': len(pairs) >= target,
            'new_validated': new_count, 'minimum_new_required': minimum_new,
            'tasks': dict(tasks), 'deterministic_checks_by_task': dict(checked),
            'deterministic_check_count': sum(checked.values()),
            'deterministic_check_rate': sum(checked.values()) / len(pairs) if pairs else 0.0,
            'template_reasoning_patterns': dict(Counter(f"{r['template_family']}:{r['reasoning_variant']}" for r, _ in pairs if r.get('template_family'))),
            'trusted_template_count': sum(bool(r.get('verification_basis')) for r, _ in pairs),
            'case_types': dict(Counter(r.get('case_type', 'unlabeled') for r, _ in pairs)),
            'difficulty': dict(Counter(r.get('difficulty', 'unlabeled') for r, _ in pairs)),
            'distinct_scenarios': len({r['id'] for r, _ in pairs}),
            'reference_status': 'synthetic silver; absent checks mean no deterministic verification'}


def validate_quality_settings(config, tasks=None):
    settings = config.get('quality', {})
    allowed = {'max_task_regression', 'per_task_max_regression', 'minimum_macro_improvement',
               'minimum_new_pair_fraction', 'minimum_deterministic_check_rate', 'deterministic_template_fraction'}
    if not isinstance(settings, dict) or set(settings) - allowed:
        raise ValueError('Unknown quality setting')
    for name in ('max_task_regression', 'minimum_macro_improvement', 'minimum_new_pair_fraction', 'minimum_deterministic_check_rate', 'deterministic_template_fraction'):
        default = {'minimum_new_pair_fraction': 0.5, 'minimum_macro_improvement': .01}.get(name, 0.0)
        value = settings.get(name, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'quality.{name} must be finite in [0, 1]')
    if settings.get('minimum_new_pair_fraction', .5) == 0:
        raise ValueError('A round must require new data')
    overrides = settings.get('per_task_max_regression', {})
    if not isinstance(overrides, dict) or (tasks is not None and set(overrides) - set(tasks)):
        raise ValueError('Unknown regression task')
    for value in overrides.values():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid task regression tolerance')
    return settings
