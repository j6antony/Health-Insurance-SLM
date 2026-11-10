"""Real CPU batch-generation check without model downloads or API calls."""
import importlib.util
import unittest

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('torch', 'transformers', 'tokenizers'))


@unittest.skipUnless(AVAILABLE, 'Optional torch/transformers CPU check')
class BatchNativeTests(unittest.TestCase):
    def test_actual_left_padded_generation_restores_padding_and_cache(self):
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from distillation_loop.student import StudentRuntime
        raw = Tokenizer(models.WordLevel({'[UNK]':0, '[PAD]':1, '[EOS]':2, 'a':3, 'b':4}, unk_token='[UNK]'))
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
        tokenizer.padding_side = 'right'
        torch.manual_seed(1)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=5, n_embd=16, n_layer=1, n_head=2,
            n_positions=32, eos_token_id=2, pad_token_id=1, use_cache=False)).eval()
        runtime = object.__new__(StudentRuntime)
        runtime.model, runtime.tokenizer = model, tokenizer
        runtime.config = {'inference': {'max_new_tokens': 3}}
        rows = runtime._generate_batch(['a', 'a b'])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(0 < row['generated_tokens'] <= 3 for row in rows))
        self.assertEqual(tokenizer.padding_side, 'right')
        self.assertFalse(model.config.use_cache)
