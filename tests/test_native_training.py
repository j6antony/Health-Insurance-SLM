"""Optional real-package CPU integration checks (no model downloads/API calls)."""
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

NATIVE_AVAILABLE = importlib.util.find_spec('trl') is not None and importlib.util.find_spec('lighteval') is not None


@unittest.skipUnless(NATIVE_AVAILABLE, 'Install requirements-training.txt for native CPU integration checks')
class NativeTrainingTests(unittest.TestCase):
    def test_native_lighteval_metric_and_corpus(self):
        from distillation_loop.evaluation import LightevalEvaluator
        from self_instruct_pipeline.config import ROOT
        schemas=json.loads((ROOT/'data/self_instruct/task_schemas.json').read_text())['tasks']
        task='document_classification'
        record={'id':'cpu-eval-1','task':task,'instruction':'Classify this synthetic document.',
                'scenario':{'synthetic':True,'documents':[{'document_id':'D1','page_number':1,
                'text':'Synthetic hospital invoice.'}],'context':{}}}
        reference={'document_type':'hospital_bill','evidence':['Synthetic hospital invoice.'],'requires_review':False}
        class Runtime:
            def infer(self,payload):
                if set(payload)!={'instruction','scenario','response_schema'}:raise AssertionError('Input leakage')
                return {'raw':json.dumps(reference),'truncated':False}
        with tempfile.TemporaryDirectory() as directory:
            result=LightevalEvaluator(Runtime(),{task:schemas[task]},1).evaluate(
                [{'record':record,'reference':reference}],Path(directory))
            self.assertEqual(result['macro_task_score'],1.0)
            self.assertEqual(result['lighteval_corpus']['task_score'],1.0)
            self.assertEqual(result['engine'],'lighteval-0.11.0-custom-metrics')

    def test_native_sft_completion_mask_and_weight_update(self):
        import torch
        from datasets import Dataset
        from tokenizers import Tokenizer, models
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        from trl import SFTConfig, SFTTrainer
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
        vocab={f't{i}':i for i in range(10)}
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocab,unk_token='t0')),
                                          unk_token='t0',pad_token='t0',eos_token='t9')
        model=GPT2LMHeadModel(GPT2Config(vocab_size=10,n_embd=16,n_layer=1,n_head=2,n_positions=16,
                                       bos_token_id=1,eos_token_id=9,pad_token_id=0))
        collator=DataCollatorForLanguageModeling(pad_token_id=0,completion_only_loss=True)
        example={'input_ids':[1,2,3,9],'completion_mask':[0,0,1,1]}
        self.assertEqual(collator([example])['labels'].tolist(),[[-100,-100,3,9]])
        before=model.transformer.wte.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            args=SFTConfig(output_dir=directory,use_cpu=True,fp16=False,bf16=False,
                           per_device_train_batch_size=1,gradient_accumulation_steps=1,max_steps=1,
                           learning_rate=0.001,gradient_checkpointing=False,max_length=16,packing=False,
                           completion_only_loss=True,dataset_kwargs={'skip_prepare_dataset':True},
                           report_to='none',save_strategy='no',eval_strategy='no',disable_tqdm=True)
            trainer=SFTTrainer(model=model,args=args,train_dataset=Dataset.from_list([example]),
                               processing_class=tokenizer,data_collator=collator)
            result=trainer.train()
            self.assertTrue(math.isfinite(result.training_loss))
            self.assertFalse(torch.equal(before,model.transformer.wte.weight.detach()))


if __name__=='__main__':unittest.main()
