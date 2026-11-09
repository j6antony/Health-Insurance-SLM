import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from distillation_loop.core import load_loop_config
from distillation_loop.data import frozen_development
from self_instruct_pipeline.config import load_config
from self_instruct_pipeline.persistence import Store
from test_self_instruct import MockParent, Pipeline, scenario


class FilterTests(unittest.TestCase):
    def test_partial_answers_keep_passed_and_record_failures(self):
        class BadFirst(MockParent):
            def generate_json(self, messages, schema, operation='generation', temperature=None):
                result = super().generate_json(messages, schema, operation, temperature)
                if operation == 'teacher_answer' and not hasattr(self, 'failed'):
                    self.failed = True
                    result['EXTRA'] = 'invalid'
                return result
        with tempfile.TemporaryDirectory() as folder:
            config = load_config()
            config['paths']['output_dir'] = folder
            config['generation'].update(target_accepted_scenarios=2, batch_size=2,
                                         max_answer_attempts_per_run=1)
            pipeline = Pipeline(config, BadFirst())
            pipeline.generate()
            self.assertEqual(pipeline.answer(allow_partial=True, max_sample_attempts=1), 1)
            self.assertEqual(pipeline.export(), 1)
            self.assertEqual(len(pipeline.store.read('rejected_records.jsonl')), 1)

    def test_repairs_are_verified_and_total_attempts_survive_resume(self):
        class BadAnswers(MockParent):
            def generate_json(self, messages, schema, operation='generation', temperature=None):
                result = super().generate_json(messages, schema, operation, temperature)
                if operation == 'teacher_answer': result['EXTRA'] = 'invalid'
                return result
        with tempfile.TemporaryDirectory() as folder:
            config = load_config(); config['paths']['output_dir'] = folder
            config['generation'].update(target_accepted_scenarios=2, batch_size=2)
            parent = BadAnswers(); pipeline = Pipeline(config, parent)
            pipeline.generate()
            self.assertEqual(pipeline.answer(allow_partial=True), 0)
            self.assertEqual(sum(op == 'teacher_answer' for op, _ in parent.calls), 6)
            later = BadAnswers(); resumed = Pipeline(config, later)
            self.assertEqual(resumed.answer(allow_partial=True), 0)
            self.assertEqual(later.calls, [])
            summary = json.loads((Path(folder)/'failure_summary.json').read_text())
            self.assertEqual(summary['failure_types']['format_error'], 6)

    def test_one_bad_scenario_does_not_discard_valid_batch_member(self):
        class BadOne(MockParent):
            def generate_json(self, messages, schema, operation='generation', temperature=None):
                value = super().generate_json(messages, schema, operation, temperature)
                if operation == 'scenario_generation' and not hasattr(self, 'bad'):
                    self.bad = True
                    value[0]['scenario']['documents'][0]['text'] = ''
                return value
        with tempfile.TemporaryDirectory() as folder:
            config = load_config(); config['paths']['output_dir'] = folder
            config['generation'].update(target_accepted_scenarios=2, batch_size=2)
            parent = BadOne(); pipeline = Pipeline(config, parent)
            self.assertEqual(pipeline.generate(), 2)
            batches = pipeline.store.read('generation_batches.jsonl')
            self.assertEqual([sum(r['task_quotas'].values()) for r in batches], [2, 1])
            rejected = pipeline.store.read('rejected_records.jsonl')
            self.assertEqual(len(rejected), 1)
            self.assertIsInstance(rejected[0]['record'], dict)
            self.assertFalse(rejected[0]['repairable'])

    def test_generation_cap_is_persistent(self):
        with tempfile.TemporaryDirectory() as folder:
            config = load_config(); config['paths']['output_dir'] = folder
            config['generation'].update(target_accepted_scenarios=2, batch_size=2,
                max_retries_per_batch=0, max_consecutive_failed_batches=100)
            pipeline = Pipeline(config, MockParent(unsafe=True))
            with self.assertRaisesRegex(ValueError, 'generation attempt limit'):
                pipeline.generate()
            self.assertEqual(len(pipeline.store.read('generation_batches.jsonl')), 6)
            later = MockParent()
            with self.assertRaisesRegex(ValueError, 'generation attempt limit'):
                Pipeline(config, later).generate()
            self.assertEqual(later.calls, [])

    def test_multi_page_evidence_keeps_document_and_page_scope(self):
        from self_instruct_pipeline.validation import validate_scenario, validate_answer, ValidationError
        from test_self_instruct import SCHEMAS
        row = scenario('field_extraction')
        row['scenario']['documents'] = [
            {'document_id': 'D1', 'page_number': 1, 'text': 'Synthetic policy record.'},
            {'document_id': 'D1', 'page_number': 2, 'text': 'Policy: SYN-42.'}]
        row['scenario']['context']['target_document_id'] = 'D1'
        validate_scenario(row, SCHEMAS, [])
        answer = {'fields': [{'field_name': 'policy_number', 'value': 'SYN-42',
                   'evidence': 'Policy: SYN-42.', 'page_number': 2, 'requires_review': False}]}
        validate_answer(row, answer, SCHEMAS['field_extraction'])
        answer['fields'][0]['page_number'] = 1
        with self.assertRaises(ValidationError): validate_answer(row, answer, SCHEMAS['field_extraction'])
        row['scenario']['documents'][1]['page_number'] = 1
        with self.assertRaises(ValidationError): validate_scenario(row, SCHEMAS, [])

    def test_targeted_quota_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            config = load_config();config['paths']['output_dir'] = folder
            pipeline = Pipeline(config, MockParent())
            plan = pipeline.plan(task_quotas={'support_chat': 2})
            self.assertEqual(plan['task_quotas'], {'support_chat': 2})
            self.assertEqual(sum(plan['difficulty'].values()), 2)

    def development_case(self, always_fail=False):
        config, loop = load_config(), load_loop_config()
        loop.update(development_examples_per_task=1, development_fill_attempts=1)
        tasks = list(json.loads(Path(config['paths']['schemas']).read_text())['tasks'])
        failed_task = tasks[0]
        instances = []
        class FakePipeline:
            def __init__(self, config, parent=None):
                self.config = config
                self.accepted = []
                self.pairs = []
                self.targets = []
                self.store = Store(config['paths']['output_dir'])
                instances.append(self)
            def initialize_manifest(self): pass
            def checkpoint(self): pass
            def generate(self, task_targets=None):
                self.targets.append(task_targets.copy())
                counts = Counter(r['task'] for r in self.accepted)
                for task, wanted in task_targets.items():
                    for _ in range(wanted - counts[task]):
                        self.accepted.append({'id': str(len(self.accepted)), 'task': task})
            def answer(self, allow_partial=False, scenario_ids=None):
                assert allow_partial
                done = {r['id'] for r, a in self.pairs}
                for row in self.accepted:
                    if row['id'] in done or (scenario_ids is not None and row['id'] not in scenario_ids): continue
                    if row['task'] == failed_task and (always_fail or len(self.targets) == 1): continue
                    self.pairs.append((row, {'ok': True}))
        return config, loop, tasks, FakePipeline, instances

    def test_evaluation_replaces_only_missing_task_then_freezes(self):
        config, loop, tasks, fake, instances = self.development_case()
        with tempfile.TemporaryDirectory() as folder, patch('distillation_loop.data.Pipeline', fake), patch('distillation_loop.data.checked_pairs', lambda p: p.pairs):
            root = Path(folder)
            data = frozen_development(config, loop, root)
            self.assertEqual(Counter(x['record']['task'] for x in data), Counter(tasks))
            first, replacement = instances[0].targets
            self.assertEqual(sum(replacement.values()) - sum(first.values()), 1)
            self.assertEqual(replacement[tasks[0]], 2)
            self.assertTrue(all(replacement[t] == 1 for t in tasks[1:]))
            self.assertEqual(frozen_development(config, loop, root), data)
            self.assertEqual(len(instances), 1)

    def test_interrupted_replacement_pass_resumes_same_targets(self):
        config, loop, tasks, fake, instances = self.development_case()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = json.loads(json.dumps(config))
            cfg['paths']['output_dir'] = str(root/'development_generation')
            pipeline = fake(cfg)
            original = pipeline.generate
            def interrupted(task_targets=None):
                original(task_targets)
                if not hasattr(pipeline, 'interrupted'):
                    pipeline.interrupted = True
                    raise RuntimeError('interrupted request')
            pipeline.generate = interrupted
            with patch('distillation_loop.data.Pipeline', lambda *a, **k: pipeline), patch('distillation_loop.data.checked_pairs', lambda p: p.pairs):
                with self.assertRaisesRegex(RuntimeError, 'interrupted request'):
                    frozen_development(config, loop, root)
                progress = json.loads((root/'development_replacements.json').read_text())
                self.assertEqual(progress['passes'], 1)
                self.assertIn('active', progress)
                data = frozen_development(config, loop, root)
            self.assertEqual(len(data), 18)
            self.assertEqual(pipeline.targets[0], pipeline.targets[1])
            progress = json.loads((root/'development_replacements.json').read_text())
            self.assertEqual(progress['passes'], 1)
            self.assertNotIn('active', progress)

    def test_replacement_limit_survives_rerun(self):
        config, loop, tasks, fake, instances = self.development_case(always_fail=True)
        with tempfile.TemporaryDirectory() as folder, patch('distillation_loop.data.Pipeline', fake), patch('distillation_loop.data.checked_pairs', lambda p: p.pairs):
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, 'attempt limit'):
                frozen_development(config, loop, root)
            self.assertEqual(len(instances[0].targets), 2)
            with self.assertRaises(ValueError): frozen_development(config, loop, root)
            self.assertEqual(instances[1].targets, [])
