import unittest

from distillation_loop.leakage import ExclusionRegistry
from self_instruct_pipeline.prompt_builder import teacher_messages, verification_messages
from self_instruct_pipeline.validation import ValidationError, validate_audit


class SemanticAuditRemediationTests(unittest.TestCase):
    def test_both_answer_paths_receive_grounding_contract(self):
        record = {'instruction': 'Report the disposition', 'task': 'example',
                  'scenario': {'documents': []}}
        schemas = {'example': {}}
        teacher = teacher_messages(record, schemas)[0]['content']
        audit = verification_messages(record, {}, schemas)[0]['content']
        for prompt in (teacher, audit):
            self.assertIn('explicitly documented negative', prompt)
            self.assertIn('precedence rule needed to answer', prompt)
            self.assertIn('name the specific gap and require review', prompt)
            self.assertIn('do not cite them as authority', prompt)
        self.assertIn('even if missing_information is empty', audit)
        self.assertIn('resolves a live conflict', audit)

    def test_failed_grounding_verdict_cannot_pass(self):
        with self.assertRaises(ValidationError):
            validate_audit({'grounded': False, 'calculations_checked': True,
                            'uncertainty_handled': False, 'unsupported_actions': False,
                            'safe_for_synthetic_sft': False,
                            'reasons': ['A missing receipt does not establish nonpayment.']})

    def test_both_partitions_excluded_by_id_and_family(self):
        development = {'id': 'dev', 'source_group': 'dev-family', 'task': 'example',
                       'instruction': 'Summarize a letter', 'scenario': {'documents': []}}
        release = {**development, 'id': 'release', 'source_group': 'release-family'}
        registry = ExclusionRegistry([development, release])
        for record in (development, release):
            with self.subTest(record=record['id']):
                self.assertEqual(registry.match(record), 'scenario_id')
                self.assertEqual(registry.match({**record, 'id': 'new-id'}), 'source_family')
