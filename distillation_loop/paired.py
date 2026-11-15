"""Paired confidence bounds for scores in [0, 1]."""
import math
from collections import Counter
from statistics import mean


def paired_summary(changes, groups=None):
    delta = mean(changes)
    # Hoeffding's two-sided bound handles all ties and sparse changes without
    # returning a misleading zero-width interval. Each paired delta is in [-1, 1].
    sizes = Counter(groups).values() if groups is not None else [1] * len(changes)
    radius = math.sqrt(2 * math.log(40) * sum((size / len(changes)) ** 2 for size in sizes))
    return {
        'cases': len(changes), 'independent_units': len(sizes), 'delta': delta,
        'gains': sum(x > 0 for x in changes), 'losses': sum(x < 0 for x in changes),
        'ties': sum(x == 0 for x in changes),
        'conservative_95pct_ci': [max(-1, delta - radius), min(1, delta + radius)],
    }
