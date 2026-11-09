import json
from collections import Counter


def group_seed_tasks(seeds):
    groups = {}
    for seed in seeds:
        task = seed.get('task')
        if not isinstance(task, str) or not task:
            raise ValueError('Every seed requires an explicit task')
        groups.setdefault(task, []).append(seed)
    if not groups:
        raise ValueError('No seed examples')
    return groups


class BalancedSeedSampler:
    def __init__(self, seeds, rng):
        self.groups = group_seed_tasks(seeds)
        self.rng = rng
        self.exposure = Counter()

    def sample(self, quotas, scheduler, accepted, count=8, machine_fraction=0.25):
        """Distinct original categories first; at most 25% validated machine examples."""
        if not 6 <= count <= 8 or not 0 <= machine_fraction <= 0.25:
            raise ValueError('Invalid demonstration settings')
        machines = [r for r in accepted if r.get('validation_status') == 'scenario_checks_passed' and r['task'] in quotas]
        self.rng.shuffle(machines)
        chosen_machines = []
        for record in machines:
            if len(chosen_machines) >= int(count * machine_fraction):
                break
            if record['task'] not in {r['task'] for r in chosen_machines}:
                chosen_machines.append(record)
        selected = [{'id': r['id'], 'task': r['task'], 'instruction': r['instruction'], 'input': r['scenario']} for r in chosen_machines]
        represented = {r['task'] for r in selected}
        candidates = list(self.groups)
        self.rng.shuffle(candidates)
        candidates.sort(key=lambda task: (
            not (task in quotas and task not in represented),
            scheduler.counts[task] / scheduler.weights[task], self.exposure[task]))
        for task in candidates:
            if len(selected) == count:
                break
            if task in represented:
                continue
            seed = self.rng.choice(self.groups[task])
            selected.append(self._demonstration(seed))
        # If all requested categories are machine-represented, other originals
        # still preserve distinct categories. Full 18-task pack always fills this.
        if len(selected) != count:
            raise ValueError('Insufficient distinct categories for demonstrations')
        self.exposure.update(r['task'] for r in selected)
        self.rng.shuffle(selected)
        return selected

    @staticmethod
    def _demonstration(seed):
        instance = seed['instances'][0]
        input_text = instance['input'].split('\nResponse schema:\n', 1)[0]
        return {'id': seed['id'], 'task': seed['task'], 'instruction': seed['instruction'],
                'input': json.loads(input_text), 'output': json.loads(instance['output'])}
