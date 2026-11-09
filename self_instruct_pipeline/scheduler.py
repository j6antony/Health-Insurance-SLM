from collections import Counter


class WeightedDeficitScheduler:
    """Plan weighted deficits using accepted records, with per-batch reservations."""

    def __init__(self, tasks, weights=None):
        self.tasks = list(tasks)
        overrides = weights or {}
        if not self.tasks or set(overrides) - set(self.tasks):
            raise ValueError('Unknown task weight or no tasks')
        self.weights = {task: overrides.get(task, 1) for task in self.tasks}
        if any(weight <= 0 for weight in self.weights.values()):
            raise ValueError('Weights must be positive')
        self.counts = Counter({task: 0 for task in self.tasks})

    def update(self, records):
        for record in records:
            if record['task'] not in self.weights:
                raise ValueError('Unknown accepted task')
            self.counts[record['task']] += 1

    def next_quotas(self, count, rng):
        if type(count) is not int or count < 1:
            raise ValueError('count must be positive')
        reserved = Counter()
        total_weight = sum(self.weights.values())
        for i in range(count):
            total = sum(self.counts.values()) + i + 1
            candidates = self.tasks[:]
            rng.shuffle(candidates)
            task = max(candidates, key=lambda t: total * self.weights[t] / total_weight - self.counts[t] - reserved[t])
            reserved[task] += 1
        return dict(reserved)


def apportioned_counts(count, weights, rng):
    scheduler = WeightedDeficitScheduler(weights, weights)
    planned = scheduler.next_quotas(count, rng)
    return {key: planned.get(key, 0) for key in weights}
