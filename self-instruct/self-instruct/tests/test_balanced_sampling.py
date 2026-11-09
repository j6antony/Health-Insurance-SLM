import importlib.util
import json
import pathlib
import random
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bootstrap', ROOT / 'self_instruct/bootstrap_instructions.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class BalancedSamplingTests(unittest.TestCase):
    def setUp(self):
        self.seeds = [json.loads(line) for line in (ROOT / 'data/seed_tasks.jsonl').read_text().splitlines()]
        self.groups = bootstrap.group_seed_tasks(self.seeds)

    def test_explicit_task_not_id_or_order(self):
        seeds = [dict(seed, id='arbitrary') for seed in reversed(self.seeds)]
        groups = bootstrap.group_seed_tasks(seeds)
        self.assertEqual(len(groups), 18)
        self.assertEqual(len(groups['claim_scrubbing']), 12)
        with self.assertRaises(ValueError):
            bootstrap.group_seed_tasks([{'id': 'seed_001'}])

    def test_distinct_tasks_and_full_records(self):
        sampler = bootstrap.BalancedSeedSampler(self.groups)
        seeds = sampler.sample_seeds('field_extraction', 8)
        self.assertEqual(len(seeds), 8)
        self.assertEqual(len({seed['task'] for seed in seeds}), 8)
        self.assertIn('field_extraction', {seed['task'] for seed in seeds})
        self.assertTrue(all('instances' in seed for seed in seeds))

    def test_underrepresented_task_wins_on_resume(self):
        records = [{'task': t} for t in self.groups for _ in range(4) if t != 'field_extraction']
        sampler = bootstrap.BalancedSeedSampler(self.groups, records)
        self.assertEqual(sampler.next_task(), 'field_extraction')
        # A seeded but unlabeled legacy output must not silently corrupt counts.
        with self.assertRaises(ValueError):
            bootstrap.BalancedSeedSampler(self.groups, [{'instruction': 'legacy'}])

    def test_batch_reservations_do_not_count_as_accepted(self):
        sampler = bootstrap.BalancedSeedSampler(self.groups)
        tasks = [sampler.next_task() for _ in range(18)]
        self.assertEqual(len(set(tasks)), 18)
        self.assertEqual(sum(sampler.generated_counts.values()), 0)
        sampler.start_batch()
        self.assertEqual(sum(sampler.pending_counts.values()), 0)

    def test_feedback_balances_accepted_counts(self):
        sampler = bootstrap.BalancedSeedSampler(self.groups)
        for _ in range(73):
            sampler.start_batch()
            task = sampler.next_task()
            sampler.record_accepted({'task': task})
        counts = list(sampler.generated_counts.values())
        self.assertEqual(sum(counts), 73)
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_seed_exposure_not_group_size(self):
        sampler = bootstrap.BalancedSeedSampler(self.groups)
        # Nine requests with eight distinct tasks each = four exposures per task.
        for _ in range(9):
            least = min(sampler.seed_groups, key=lambda t: sampler.seed_counts[t])
            sampler.sample_seeds(least, 8)
        counts = list(sampler.seed_counts.values())
        self.assertEqual(sum(counts), 72)
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_batch_prompt_target_and_machine_scope(self):
        records = [{'task': 'field_extraction', 'instruction': 'UNIQUE_MACHINE_EXAMPLE'}]
        groups = {'field_extraction': self.groups['field_extraction']}
        sampler = bootstrap.BalancedSeedSampler(groups, records)
        batch = bootstrap.build_batch(sampler, records, 2, 8)
        for request in batch:
            self.assertEqual(request['task'], 'field_extraction')
            self.assertIn('only for task: field_extraction', request['prompt'])
            self.assertIn('UNIQUE_MACHINE_EXAMPLE', request['prompt'])
            self.assertEqual(len(request['seed_ids']), 1)

    def test_invalid_sizes_and_deterministic_ties(self):
        a = bootstrap.BalancedSeedSampler(self.groups, rng=random.Random(8))
        b = bootstrap.BalancedSeedSampler(self.groups, rng=random.Random(8))
        self.assertEqual(bootstrap.build_batch(a, [], 5, 8), bootstrap.build_batch(b, [], 5, 8))
        for n in (0, 19):
            with self.assertRaises(ValueError):
                a.sample_seeds('field_extraction', n)
        with self.assertRaises(ValueError):
            bootstrap.build_batch(a, [], 0, 8)


if __name__ == '__main__':
    unittest.main()
