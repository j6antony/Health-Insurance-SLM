import copy
import unittest
from test_self_instruct import SCHEMAS, scenario
from self_instruct_pipeline.validation import validate_answer, ValidationError


class ExtractionConflictTests(unittest.TestCase):
    def setUp(self):
        self.row = scenario('field_extraction')
        self.row['scenario']['documents'] = [
            {'document_id': 'D1', 'page_number': 1, 'text': 'Patient: Mira Chen. Total: USD 1,860.00.'},
            {'document_id': 'D2', 'page_number': 1, 'text': 'Patient: Mina Chen. Total: USD 1,680.00.'}]
        self.answer = {'fields': [
            {'field_name': 'patient_name', 'value': 'Mira Chen', 'evidence': 'Patient: Mira Chen.', 'page_number': 1, 'requires_review': True},
            {'field_name': 'patient_name', 'value': 'Mina Chen', 'evidence': 'Patient: Mina Chen.', 'page_number': 1, 'requires_review': True}]}

    def test_both_conflicting_values_are_valid_with_evidence(self):
        validate_answer(self.row, self.answer, SCHEMAS['field_extraction'])

    def test_conflicting_entries_require_review(self):
        self.answer['fields'][1]['requires_review'] = False
        with self.assertRaisesRegex(ValidationError, 'must all require review'):
            validate_answer(self.row, self.answer, SCHEMAS['field_extraction'])

    def test_same_value_twice_is_still_rejected(self):
        self.answer['fields'][1] = copy.deepcopy(self.answer['fields'][0])
        with self.assertRaisesRegex(ValidationError, 'Duplicate extracted value'):
            validate_answer(self.row, self.answer, SCHEMAS['field_extraction'])

    def test_unsupported_value_or_wrong_scope_is_still_rejected(self):
        self.row['scenario']['context']['target_document_id'] = 'D1'
        with self.assertRaisesRegex(ValidationError, 'scope|mismatch'):
            validate_answer(self.row, self.answer, SCHEMAS['field_extraction'])
        self.row['scenario']['context'].clear()
        self.answer['fields'][1]['value'] = 'Invented Name'
        with self.assertRaisesRegex(ValidationError, 'unsupported'):
            validate_answer(self.row, self.answer, SCHEMAS['field_extraction'])

    def test_explicit_recovery_grants_only_attempt_four(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from test_self_instruct import MockParent, Pipeline
        from self_instruct_pipeline.config import load_config
        from distillation_loop.core import load_loop_config
        from distillation_loop.data import repair_development_task
        class BadParent(MockParent):
            def generate_json(self, messages, schema, operation='generation', temperature=None):
                value = super().generate_json(messages, schema, operation, temperature)
                if operation == 'teacher_answer': value['EXTRA'] = 'bad'
                return value
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = load_config()
            config['generation']['task_weights'] = {}
            config['generation']['target_accepted_scenarios'] = 1
            config['paths']['output_dir'] = str(root/'development_generation')
            parent = BadParent()
            pipeline = Pipeline(config, parent)
            pipeline.initialize_manifest()
            pipeline.generate(task_targets={'field_extraction': 1})
            pipeline.answer(allow_partial=True)
            before = sum(op == 'teacher_answer' for op, _ in parent.calls)
            loop = load_loop_config(); loop['development_examples_per_task'] = 1
            with patch('distillation_loop.data.Pipeline', side_effect=lambda cfg: Pipeline(cfg, parent)):
                self.assertFalse(repair_development_task(config, loop, root)['validated'])
                self.assertFalse(repair_development_task(config, loop, root)['validated'])
            after = sum(op == 'teacher_answer' for op, _ in parent.calls)
            self.assertEqual(after - before, 1)
