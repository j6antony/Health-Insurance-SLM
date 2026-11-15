"""Paired confidence bounds for scores in [0, 1]."""
import math
from statistics import mean


def paired_summary(changes):
    delta = mean(changes)
    # Hoeffding's two-sided bound handles all ties and sparse changes without
    # returning a misleading zero-width interval. Each paired delta is in [-1, 1].
    radius = math.sqrt(2 * math.log(40) / len(changes))
    return {
        'cases': len(changes), 'delta': delta,
        'gains': sum(x > 0 for x in changes), 'losses': sum(x < 0 for x in changes),
        'ties': sum(x == 0 for x in changes),
        'conservative_95pct_ci': [max(-1, delta - radius), min(1, delta + radius)],
    }
