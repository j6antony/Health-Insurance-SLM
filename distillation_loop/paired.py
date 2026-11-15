"""Paired confidence intervals with task stratification and optional family clusters."""
from collections import defaultdict
import math
import random
from statistics import mean

RESAMPLES = 2000
SEED = 42


def paired_summary(changes, groups=None, *, strata=None):
    strata = list(strata) if strata is not None else [None] * len(changes)
    families = list(groups) if groups is not None else list(range(len(changes)))
    partitions = {}
    for task, family, change in zip(strata, families, changes, strict=True):
        if groups is not None and (family is None or family == ''):
            raise ValueError('Paired bootstrap requires a source family for every case')
        partitions.setdefault(task, defaultdict(list))[family].append(change)
    task_values = {task: [value for values in clusters.values() for value in values]
                   for task, clusters in partitions.items()}
    task_deltas = {task: mean(values) for task, values in task_values.items()}
    delta = mean(task_deltas.values())
    weights = [len(values) / (len(task_values[task]) * len(partitions))
               for task, clusters in partitions.items() for values in clusters.values()]
    radius = math.sqrt(2 * math.log(40) * sum(weight ** 2 for weight in weights))
    report = {
        'cases': len(changes), 'independent_units': len(weights), 'delta': delta,
        'gains': sum(x > 0 for x in changes), 'losses': sum(x < 0 for x in changes),
        'ties': sum(x == 0 for x in changes),
        'conservative_ci_method': 'Hoeffding bound on paired deltas',
        'conservative_95pct_ci': [max(-1, delta - radius), min(1, delta + radius)],
        'per_task_deltas': {task: value for task, value in task_deltas.items() if task is not None},
        'bootstrap_seed': SEED, 'bootstrap_resamples': RESAMPLES,
    }

    rng = random.Random(SEED)
    values = list(task_values.values())
    case_samples = sorted(mean(sum(rng.choices(task, k=len(task))) / len(task) for task in values)
                          for _ in range(RESAMPLES))
    report.update(bootstrap_95pct_ci=[case_samples[49], case_samples[1949]],
                  bootstrap_ci_method='paired cases resampled within task; equally weighted task means')
    if groups is not None:
        rng = random.Random(SEED)
        pooled = [[(sum(values), len(values)) for values in clusters.values()]
                  for clusters in partitions.values()]

        def resample_task(task):
            sums, counts = zip(*rng.choices(task, k=len(task)))
            return sum(sums) / sum(counts)

        samples = sorted(mean(resample_task(task) for task in pooled) for _ in range(RESAMPLES))
        report.update(source_families=len(set(families)),
                      source_families_per_task={task: len(clusters) for task, clusters in partitions.items()
                                                if task is not None},
                      clustered_bootstrap_95pct_ci=[samples[49], samples[1949]],
                      clustered_bootstrap_ci_method='paired source families resampled within task; '
                                                    'actual sampled case counts and equally weighted task means')
    return report
