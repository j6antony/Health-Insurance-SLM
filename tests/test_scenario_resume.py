import tempfile
import unittest

from distillation_loop.leakage import ExclusionRegistry
from self_instruct_pipeline.config import load_config
from self_instruct_pipeline.parent_backend import ParentError
from self_instruct_pipeline.pipeline import Pipeline
from test_self_instruct import MockParent, scenario


class ScenarioResumeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        config = load_config()
        config['paths']['output_dir'] = directory.name
        config['generation']['target_accepted_scenarios'] = 1
        self.parent = MockParent()
        self.pipeline = Pipeline(config, self.parent)
        self.pipeline.initialize_manifest()
        self.plan = {'task_quotas': {'claim_scrubbing': 1}, 'difficulty': {'medium': 1},
                     'seed_ids': []}
        self.pipeline.plan = lambda **kwargs: {**self.plan, 'messages': [
            {'role': 'system', 'content': 'test'},
            {'role': 'user', 'content': '{"request":{"task_quotas":{"claim_scrubbing":1},'
                                        '"difficulty":{"medium":1}}}'}]}

    def test_raw_scenario_accepted_or_excluded_before_permanent_id(self):
        protected = {**scenario(i=1), 'id': 'frozen-development'}
        raw_holdout = scenario(i=99)
        raw_holdout['instruction'] = 'Identify the forwarding address on the returned envelope.'
        raw_holdout['scenario']['documents'][0]['text'] = 'Undeliverable: addressee relocated overseas.'
        holdout = {**raw_holdout, 'id': 'frozen-holdout'}
        registry = ExclusionRegistry([protected, holdout])
        self.pipeline.exclusion_guard = registry.match
        self.assertEqual(registry.match(raw_holdout), 'canonical_content')
        self.assertEqual(self.pipeline.generate(one_batch=True), 0)
        self.assertEqual(self.pipeline.store.read('rejected_records.jsonl')[-1]['reason'],
                         'held_out_overlap')
        self.pipeline.exclusion_guard = ExclusionRegistry([holdout]).match
        self.assertEqual(self.pipeline.generate(one_batch=True), 1)
        self.assertTrue(self.pipeline.accepted[0]['id'])

    def test_unexpected_key_error_stops_before_another_request(self):
        def broken(row):
            raise KeyError('programmer bug')

        self.pipeline.exclusion_guard = broken
        with self.assertRaisesRegex(KeyError, 'programmer bug'):
            self.pipeline.generate()
        self.assertEqual([operation for operation, _ in self.parent.calls],
                         ['scenario_generation', 'scenario_audit'])

    def test_provider_rejection_has_distinct_metadata_and_no_retry(self):
        def reject(*args, **kwargs):
            raise ParentError('OpenAI HTTP 400 during scenario_generation',
                              metadata={'http_status': 400, 'operation': 'scenario_generation',
                                        'error_code': 'invalid_prompt'})

        self.parent.generate_json = reject
        with self.assertRaises(ParentError):
            self.pipeline.generate()
        row, = self.pipeline.store.read('rejected_records.jsonl')
        self.assertEqual(row['failure_type'], 'provider_rejection')
        self.assertEqual(row['record']['error_code'], 'invalid_prompt')
        self.assertEqual(len(self.pipeline.store.read('generation_batches.jsonl')), 1)

    def test_recovery_reaudits_and_is_idempotent(self):
        self.pipeline.exclusion_guard = ExclusionRegistry([]).match
        self.pipeline.store.append('generation_batches.jsonl', dict(self.plan, batch_id='batch'))
        self.pipeline.store.append('rejected_records.jsonl', {
            'id': 'legacy', 'batch_id': 'batch', 'phase': 'scenario_validation',
            'reason': "'id'", 'record': [scenario(i=713)]})
        accepted, = self.pipeline.recover_legacy_batch('legacy')
        self.assertEqual(accepted['recovered_from_rejection_id'], 'legacy')
        self.assertEqual(self.pipeline.recover_legacy_batch('legacy'), [])
        self.assertEqual([operation for operation, _ in self.parent.calls], ['scenario_audit'])
