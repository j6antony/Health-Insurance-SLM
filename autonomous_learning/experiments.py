"""Controlled-comparison contracts. No results are invented for unrun arms."""
import math
from .storage import digest


def comparison_plan(mode, arms):
    if mode not in ('equal_training_budget', 'equal_api_cost') or set(arms) != {'A', 'B'}:
        raise ValueError('Require A (direct teacher) and B (scenario reuse) arms')
    required = {'approach', 'accepted_examples', 'training_tokens', 'initial_adapter_sha256',
                'training_config', 'benchmark_id', 'api_ceiling_usd', 'dataset_id', 'generation_seconds',
                'api_cost_usd', 'unique_families'}
    for label, arm in arms.items():
        if set(arm) != required or arm['approach'] != ('direct_teacher' if label == 'A' else 'scenario_reuse'):
            raise ValueError('Invalid controlled arm contract')
        for key in ('accepted_examples', 'training_tokens', 'unique_families'):
            if type(arm[key]) is not int or arm[key] < 1:
                raise ValueError('Measured positive integer required: ' + key)
        for key in ('api_ceiling_usd', 'api_cost_usd', 'generation_seconds'):
            if type(arm[key]) not in (int, float) or not math.isfinite(arm[key]) or arm[key] < 0:
                raise ValueError('Finite measured value required: ' + key)
        if arm['api_cost_usd'] > arm['api_ceiling_usd']:
            raise ValueError('Arm exceeded API ceiling')
    a, b = arms['A'], arms['B']
    for key in ('initial_adapter_sha256', 'training_config', 'benchmark_id'):
        if a[key] != b[key] or not a[key]:
            raise ValueError('Unmatched experiment condition: ' + key)
    if mode == 'equal_training_budget' and any(a[key] != b[key] for key in ('accepted_examples', 'training_tokens')):
        raise ValueError('Equal-training comparison requires matched count and measured token exposure')
    if mode == 'equal_api_cost' and a['api_ceiling_usd'] != b['api_ceiling_usd']:
        raise ValueError('API ceilings must match')
    result = {'version': 1, 'mode': mode, 'arms': arms, 'performance': 'unavailable until checkpoints evaluated',
              'limitations': ['Synthetic silver datasets; requires independent transfer evaluation.',
                              'Exact token matching is conservative; no hypothetical savings are measured.']}
    return {**result, 'experiment_id': digest(result)}
