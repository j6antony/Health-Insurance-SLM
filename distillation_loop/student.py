import gc
import json
import math
from pathlib import Path

from self_instruct_pipeline.persistence import atomic_json
from self_instruct_pipeline.prompt_builder import target_payload


def render_prompt(tokenizer, payload):
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def tokenize_teacher_pair(tokenizer, payload, answer, max_length):
    # Keep the exact same Qwen non-thinking assistant prefix as inference.
    prompt = render_prompt(tokenizer, payload)
    prefix = tokenizer(prompt, add_special_tokens=False)['input_ids']
    completion = tokenizer(json.dumps(answer, ensure_ascii=False), add_special_tokens=False)['input_ids']
    if tokenizer.eos_token_id is None:
        raise ValueError('Student tokenizer needs an EOS token')
    completion += [tokenizer.eos_token_id]
    if not prefix or not completion or len(prefix) + len(completion) > max_length:
        raise ValueError('Training sequence exceeds budget; no silent truncation is allowed')
    return {'input_ids': prefix + completion, 'completion_mask': [0] * len(prefix) + [1] * len(completion)}


class StudentRuntime:
    """Use the notebook's existing quantized model; train LoRA, never base weights."""

    def __init__(self, model, tokenizer, loop_config):
        self.model, self.tokenizer, self.config = model, tokenizer, loop_config
        self.prepared = False
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def prepare(self):
        import torch
        from transformers import set_seed
        from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if self.prepared:
            return
        if not torch.cuda.is_available():
            raise ValueError('The configured 4-bit QLoRA loop requires a CUDA GPU')
        if not isinstance(self.model, PeftModel):
            set_seed(self.config['random_seed'])
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
            cfg = self.config['training']
            self.model = get_peft_model(self.model, LoraConfig(
                r=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'],
                task_type='CAUSAL_LM', bias='none',
                target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
        if not any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('No trainable adapter parameters')
        self.prepared = True
        self.model.eval()

    def identity(self):
        config=self.model.config
        return {'model_name':getattr(config,'_name_or_path',None),
                'model_revision':getattr(config,'_commit_hash',None),
                'tokenizer_name':getattr(self.tokenizer,'name_or_path',None),
                'tokenizer_revision':getattr(self.tokenizer,'init_kwargs',{}).get('_commit_hash'),
                'tokenizer_class':type(self.tokenizer).__name__}

    def restore(self, checkpoint):
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights
        self.prepare()
        if not Path(checkpoint, 'adapter_config.json').is_file():
            raise ValueError('Missing LoRA adapter checkpoint')
        state = load_peft_weights(str(checkpoint), device='cpu')
        set_peft_model_state_dict(self.model, state)
        self.model.eval()

    def save_checkpoint(self, checkpoint):
        self.prepare()
        checkpoint = Path(checkpoint)
        checkpoint.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint, safe_serialization=True)
        self.tokenizer.save_pretrained(checkpoint)
        return str(checkpoint)

    def infer(self, payload):
        import torch
        self.prepare()
        self.model.eval()
        cfg = self.config
        text = render_prompt(self.tokenizer, payload)
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')
        length = encoded['input_ids'].shape[-1]
        budget = cfg['inference']['max_new_tokens']
        if length + budget > cfg['training']['max_sequence_length']:
            return {'raw': '', 'truncated': True, 'error': 'Prompt exceeds inference context budget'}
        inputs = encoded.to(self.model.get_input_embeddings().weight.device)
        previous_cache = self.model.config.use_cache
        try:
            self.model.config.use_cache = True
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                                             pad_token_id=self.tokenizer.pad_token_id,
                                             eos_token_id=self.tokenizer.eos_token_id)
            generated = output[0][length:]
            ids = generated.tolist()
            eos = self.tokenizer.eos_token_id
            ended = bool(ids) and ids[-1] in (eos if isinstance(eos, list) else [eos])
            return {'raw': self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
                    'truncated': len(ids) >= budget and not ended, 'generated_tokens': len(ids)}
        finally:
            self.model.config.use_cache = previous_cache

    def train(self, pairs, schemas, output_dir):
        import torch
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
        self.prepare()
        cfg = self.config['training']
        examples = [tokenize_teacher_pair(self.tokenizer, target_payload(r, schemas), answer,
                                         cfg['max_sequence_length']) for r, answer in pairs]
        if not examples:
            raise ValueError('No accepted teacher targets for training')
        self.model.config.use_cache = False
        # Lighteval and inference must not leave the global autograd state disabled.
        torch.set_grad_enabled(True)
        self.model.train()
        trainer = None
        try:
            args = SFTConfig(
                output_dir=str(output_dir), per_device_train_batch_size=1,
                gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
                num_train_epochs=cfg['epochs_per_round'],
                max_steps=min(cfg['max_steps_per_round'],
                              math.ceil(len(examples) / cfg['gradient_accumulation_steps']) * cfg['epochs_per_round']),
                learning_rate=cfg['learning_rate'], max_length=cfg['max_sequence_length'],
                completion_only_loss=True, packing=False,
                dataset_kwargs={'skip_prepare_dataset': True},
                gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                fp16=True, bf16=False, optim='adamw_torch', report_to='none',
                save_strategy='no', eval_strategy='no', logging_steps=1,
                seed=self.config['random_seed'], remove_unused_columns=False)
            trainer = SFTTrainer(model=self.model, args=args, train_dataset=Dataset.from_list(examples),
                                 processing_class=self.tokenizer,
                                 data_collator=DataCollatorForLanguageModeling(
                                     pad_token_id=self.tokenizer.pad_token_id, completion_only_loss=True))
            # Verify the actual trainer collator masks every prompt token.
            sample = examples[0]
            batch = trainer.data_collator([sample])
            prompt_length = sample['completion_mask'].index(1)
            if not bool((batch['labels'][0, :prompt_length] == -100).all()):
                raise ValueError('SFT prompt masking failed')
            if not bool((batch['labels'][0, prompt_length:] != -100).all()):
                raise ValueError('SFT completion labels were unexpectedly masked')
            result = trainer.train()
            if not math.isfinite(result.training_loss):
                raise ValueError('SFT produced a nonfinite training loss; checkpoint not committed')
            self.model = trainer.model
            checkpoint = Path(output_dir) / 'adapter'
            checkpoint.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(checkpoint, safe_serialization=True)
            self.tokenizer.save_pretrained(checkpoint)
            atomic_json(Path(output_dir) / 'training_metrics.json', result.metrics)
            return str(checkpoint)
        finally:
            self.model.eval()
            if trainer is not None:
                del trainer
            gc.collect()
            torch.cuda.empty_cache()
