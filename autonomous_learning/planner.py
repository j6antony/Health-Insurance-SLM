"""Strict offline curriculum interface; aggregate feedback excludes holdout contents."""
from copy import deepcopy
import math
from .tasks import TASKS

LIMITS = {'max_scenarios': 1000, 'max_variants': 10, 'max_family_cap': 64,
          'max_api_usd': 0, 'max_gpu_usd': 0, 'replay_fraction': .25}


def validate_plan(plan, limits=None, previous=None):
    limits = {**LIMITS, **(limits or {})}
    keys = {'cycle_id', 'master_scenarios', 'focus', 'variants_per_family', 'max_training_examples_per_family',
            'training_method', 'adapter_strategy', 'replay_fraction', 'max_api_usd', 'max_gpu_usd',
            'rationale', 'expected_effect'}
    if set(plan) != keys:
        raise ValueError('Curriculum requires exactly the version-1 contract fields')
    bounds = {'cycle_id': (1, 1000000), 'master_scenarios': (4, limits['max_scenarios']),
              'variants_per_family': (0, limits['max_variants']),
              'max_training_examples_per_family': (7, limits['max_family_cap'])}
    for key, (low, high) in bounds.items():
        if type(plan[key]) is not int or not low <= plan[key] <= high:
            raise ValueError('Curriculum bound violated: ' + key)
    if plan['training_method'] != 'qlora_sft' or plan['adapter_strategy'] not in ('fresh', 'continued'):
        raise ValueError('Unsupported training method/adapter strategy')
    if plan['replay_fraction'] != limits['replay_fraction']:
        raise ValueError('Replay allocation is fixed by operator policy')
    focus = plan['focus']
    if set(focus) != set(TASKS) or any(type(v) is not int or v < 5 for v in focus.values()) or sum(focus.values()) != 100:
        raise ValueError('Task focus must total 100 with at least 5 per allowed task')
    for key in ('max_api_usd', 'max_gpu_usd'):
        if type(plan[key]) not in (int, float) or not math.isfinite(plan[key]) or not 0 <= plan[key] <= limits[key]:
            raise ValueError('Curriculum budget exceeds operator limits')
    for key in ('rationale', 'expected_effect'):
        if not isinstance(plan[key], str) or not 1 <= len(plan[key]) <= 2000:
            raise ValueError('Invalid curriculum explanation')
    if previous:
        if plan['cycle_id'] != previous['cycle_id'] + 1:
            raise ValueError('Curriculum cycle sequence changed')
        if not previous['master_scenarios'] / 2 <= plan['master_scenarios'] <= previous['master_scenarios'] * 2:
            raise ValueError('Unbounded curriculum size change')
        if any(abs(focus[t] - previous['focus'][t]) > 20 for t in TASKS):
            raise ValueError('Unbounded curriculum focus change')
    return deepcopy(plan)


def planner_input(report=None, statistics=None, quarantine=None, budget=None, previous=None):
    # Accept aggregates only. No arbitrary case contents or sealed holdout report
    # are forwarded to a programmatic or interactive planner.
    report = report or {}
    if report.get('grounded_score_available') is False:
        report = {'case_count': report.get('case_count', 0)}
    return {'performance': {k: report[k] for k in ('per_task', 'macro_task_score', 'paired', 'failure_counts', 'case_count') if k in report},
            'dataset': statistics or {}, 'quarantine_counts': quarantine or {},
            'available_budget': budget or {'api_usd': 0, 'gpu_usd': 0}, 'prior_plan': previous}


def heuristic(cycle=1, count=24, feedback=None, previous=None):
    focus = {t: 14 for t in TASKS}
    focus[TASKS[-1]] += 2
    feedback = feedback or {}
    performance = feedback.get('performance', {})
    if performance.get('case_count', 0) >= 8 and performance.get('per_task'):
        weakest = min(performance['per_task'], key=performance['per_task'].get)
        if weakest in focus:
            for t in TASKS:
                if t != weakest:
                    focus[t] -= 1
            focus[weakest] += 6
    if previous:
        count = max(previous['master_scenarios'] // 2, min(count, previous['master_scenarios'] * 2))
    return {'cycle_id': cycle, 'master_scenarios': count, 'focus': focus,
            'variants_per_family': 3, 'max_training_examples_per_family': 8,
            'training_method': 'qlora_sft', 'adapter_strategy': 'fresh', 'replay_fraction': .25,
            'max_api_usd': 0, 'max_gpu_usd': 0,
            'rationale': 'Maintain task coverage; prioritize measured weaknesses when sufficient cases exist.',
            'expected_effect': 'Test document verification skills; improvement is unmeasured.'}
