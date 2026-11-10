import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from distillation_loop.benchmarks import load_benchmarks
from distillation_loop.core import score_response
from distillation_loop.controller import run_loop
from self_instruct_pipeline.prompt_builder import target_payload
from self_instruct_pipeline.validation import ValidationError, validate_answer
import test_distillation_loop as fixtures
from test_distillation_loop import SCHEMAS, record, FakeRuntime, FakeEvaluator


class ReviewedEvaluationTests(unittest.TestCase):
    def test_rule_checks_reject_wrong_teacher_and_stay_hidden(self):
        source = record()
        source['answer_checks'] = [{'path': ['requires_review'], 'operation': 'any', 'operands': [False, True]}]
        answer = {'document_type': 'hospital_bill', 'evidence': ['Synthetic hospital invoice'], 'requires_review': False}
        with self.assertRaises(ValidationError):
            validate_answer(source, answer, SCHEMAS[source['task']])
        answer['requires_review'] = True
        validate_answer(source, answer, SCHEMAS[source['task']])
        self.assertNotIn('answer_checks', json.dumps(target_payload(source, SCHEMAS)))

    def test_reversed_text_cannot_pass_on_word_overlap(self):
        schema = {'type': 'object'}
        source = record('reasoning')
        gold = {'answer': 'Do not submit because authorization is required.', 'requires_review': False}
        wrong = dict(gold, answer='Do submit because authorization is not required.')
        self.assertEqual(score_response(source, gold, json.dumps(wrong), {'reasoning': schema})['task_score'], 0)

    def test_review_and_split_source_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {split: str(Path(directory) / f'{split}.jsonl') for split in ('validation', 'test')}
            cfg = {'development_examples_per_task': 1, 'evaluation': {'mode': 'reviewed', 'validation_path': paths['validation'], 'test_path': paths['test'], 'minimum_test_examples_per_task': 1}}
            for i, split in enumerate(paths):
                r = record(i=i)
                r['instruction'] = f'Independent {split} task'
                r['scenario']['documents'][0]['text'] = 'Synthetic hospital invoice' if i == 0 else 'Completely different synthetic admission record'
                row = {'record': r, 'reference': {'document_type': 'hospital_bill', 'evidence': [r['scenario']['documents'][0]['text']], 'requires_review': False}, 'review': {'status': 'approved', 'reviewer': 'expert', 'source_group': 'shared'}}
                Path(paths[split]).write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'source groups'):
                load_benchmarks(cfg, {'document_classification': SCHEMAS['document_classification']})
            row['review']['source_group'] = 'separate'
            Path(paths['test']).write_text(json.dumps(row) + '\n')
            validation, test = load_benchmarks(cfg, {'document_classification': SCHEMAS['document_classification']})
            self.assertEqual((len(validation), len(test)), (1, 1))
            row['review']['status'] = 'pending'
            Path(paths['test']).write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'approved'):
                load_benchmarks(cfg, {'document_classification': SCHEMAS['document_classification']})

    def test_final_test_runs_after_selection_and_is_cached_on_resume(self):
        fixture = fixtures.ControllerTests()
        fixture.setUp()
        self.addCleanup(fixture.tmp.cleanup)
        fixture.config['max_rounds'] = 1
        fixture.config['evaluation']['minimum_test_examples_per_task'] = 3
        evaluator = FakeEvaluator([.1, .95, .2, .7])
        runtime = FakeRuntime()
        final = [{'record': record(i=1000), 'reference': {}}]
        with patch('distillation_loop.controller.load_benchmarks', return_value=(fixture.development, final)):
            result = fixture.run_loop(runtime, evaluator)
            self.assertEqual(evaluator.calls, 4)
            self.assertAlmostEqual(result['final_comparison']['macro_delta'], .5)
            fixture.run_loop(runtime, evaluator)
            self.assertEqual(evaluator.calls, 4)
