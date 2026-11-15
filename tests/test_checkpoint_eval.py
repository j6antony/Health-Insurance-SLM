import unittest
from unittest.mock import patch

from distillation_loop.checkpoint_eval import RepetitionStopping, evaluate_predictions
from distillation_loop.student import StudentRuntime


class CheckpointEvaluationTests(unittest.TestCase):
    def test_incomplete_json_is_counted_even_when_repetition_stopped_before_token_limit(self):
        cases = [{'record': {'id': 'case', 'task': 'task'}, 'reference': {}}]
        prediction = {'raw': '{"unfinished":', 'truncated': True, 'generated_tokens': 33,
                      'stop_reason': 'repetition'}
        rows, report = evaluate_predictions(cases, [prediction], {'task': {}})
        self.assertEqual(rows[0]['task_score'], 0)
        self.assertEqual(report['schema_evidence_valid_rate'], 0)
        self.assertEqual(report['truncated_rate'], 1)
        self.assertEqual(report['repetition_stop_rate'], 1)
        self.assertEqual(report['token_limit_rate'], 0)
        self.assertEqual(report['generated_tokens']['total'], 33)

    def test_stopper_detects_odd_period_per_sequence_without_counting_padding(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch is needed for the batched stopping criterion')

        class Tokenizer:
            eos_token_id = 1
            pad_token_id = 0
            padding_side = 'right'

            def __call__(self, *args, **kwargs):
                return Batch(input_ids=torch.tensor([[25], [25]]))

            def decode(self, ids, **kwargs):
                return ' '.join(map(str, ids))

        class Batch(dict):
            def to(self, device):
                return self

        class Model:
            config = type('Config', (), {'use_cache': False})()

            def eval(self):
                pass

            def get_input_embeddings(self):
                return type('Embedding', (), {'weight': torch.zeros(1)})()

            def generate(self, input_ids, *, stopping_criteria, **kwargs):
                repeated = list(range(100, 111)) * 3
                ordinary = list(range(200, 233))
                for length in (24, 32, 33):
                    ids = torch.tensor([[25, *repeated[:length]], [25, *ordinary[:length]]])
                    self.flags = stopping_criteria[0](ids, None).tolist()
                    if length < 33:
                        self_test.assertEqual(self.flags, [False, False])
                self_test.assertEqual(self.flags, [True, False])
                stopped = torch.tensor([[25, *repeated, 0, 0], [25, *ordinary, 1, 0]])
                self_test.assertEqual(stopping_criteria[0](stopped, None).tolist(), [True, False])
                return stopped

        self_test = self
        runtime = StudentRuntime(Model(), Tokenizer(), {'inference': {'max_new_tokens': 40}})
        runtime.repetition_stopping = RepetitionStopping
        with patch.dict('sys.modules', {'transformers': type('Transformers', (), {'StoppingCriteriaList': list})}):
            repeated, ordinary = runtime._generate_batch(['a', 'b'])
        self.assertEqual(repeated['generated_tokens'], 33)
        self.assertEqual(repeated['stop_reason'], 'repetition')
        self.assertTrue(repeated['truncated'])
        self.assertEqual(ordinary['generated_tokens'], 34)
        self.assertFalse(ordinary['truncated'])
        self.assertEqual(runtime.tokenizer.padding_side, 'right')
        self.assertFalse(runtime.model.config.use_cache)
