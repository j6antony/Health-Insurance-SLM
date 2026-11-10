import gc
import json
import math
import os
import sys
from pathlib import Path

from self_instruct_pipeline.persistence import atomic_json
from self_instruct_pipeline.prompt_builder import target_payload


def require_single_gpu(torch):
    if not torch.cuda.is_available():
        raise ValueError('The configured 4-bit QLoRA loop requires a CUDA GPU')
    if torch.cuda.device_count() != 1:
        raise ValueError('This QLoRA loop requires exactly one visible GPU. Restart the kernel and set CUDA_VISIBLE_DEVICES=0 before importing torch or loading the model.')


def clear_cuda_cache(torch):
    # Preserve the original training traceback if the CUDA context is poisoned.
    training_error = sys.exc_info()[0] is not None
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        if not training_error:
            raise


def decode_prediction(tokenizer, ids, budget):
    eos = tokenizer.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos])
    ended = False
    for index, token in enumerate(ids):
        if token in stops:
            ids, ended = ids[:index + 1], True
            break
    return {'raw': tokenizer.decode(ids, skip_special_tokens=True).strip(),
            'truncated': len(ids) >= budget and not ended, 'generated_tokens': len(ids)}


def infer_payloads(runtime, payloads):
    if hasattr(runtime, 'infer_many'):
        return runtime.infer_many(payloads)
    return [runtime.infer(p) for p in payloads]


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

    def __init__(self, model, tokenizer, loop_config, distributed=False):
        self.model, self.tokenizer, self.config = model, tokenizer, loop_config
        self.distributed = distributed
        self.prepared = False
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def prepare(self):
        import torch
        from transformers import set_seed
        from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if self.prepared:
            return
        if self.distributed:
            if not torch.cuda.is_available():
                raise ValueError('Distributed QLoRA requires CUDA')
        else:
            require_single_gpu(torch)
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
        return self.infer_many([payload])[0]

    def infer_many(self, payloads, batch_size=2):
        """Length-bucketed left-padded decoding, with per-example EOS accounting."""
        import torch
        self.prepare()
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        results = [None] * len(payloads)
        texts = [render_prompt(self.tokenizer, p) for p in payloads]
        lengths = [len(self.tokenizer(t, add_special_tokens=False)['input_ids']) for t in texts]
        valid = []
        for index, length in enumerate(lengths):
            if length + self.config['inference']['max_new_tokens'] > self.config['training']['max_sequence_length']:
                results[index] = {'raw': '', 'truncated': True, 'error': 'Prompt exceeds inference context budget'}
            else:
                valid.append(index)
        valid.sort(key=lambda i: lengths[i])
        for start in range(0, len(valid), batch_size):
            indices = valid[start:start + batch_size]
            oom = False
            try:
                predictions = self._generate_batch([texts[i] for i in indices])
            except torch.cuda.OutOfMemoryError:
                if len(indices) == 1:
                    raise
                oom = True
            if oom:
                gc.collect(); clear_cuda_cache(torch)
                predictions = [self._generate_batch([texts[i]])[0] for i in indices]
            for index, prediction in zip(indices, predictions):
                results[index] = prediction
            print(f'GPU {os.environ.get("LOCAL_RANK", "0")} inference: {min(start + len(indices), len(valid))}/{len(valid)}', flush=True)
        return results

    def _generate_batch(self, texts):
        import torch
        self.model.eval()
        previous_cache, previous_padding = self.model.config.use_cache, self.tokenizer.padding_side
        try:
            self.model.config.use_cache = True
            self.tokenizer.padding_side = 'left'
            encoded = self.tokenizer(texts, padding=True, add_special_tokens=False, return_tensors='pt')
            width = encoded['input_ids'].shape[-1]
            inputs = encoded.to(self.model.get_input_embeddings().weight.device)
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=self.config['inference']['max_new_tokens'],
                    do_sample=False, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            rows = []
            for generated in output[:, width:].tolist():
                rows.append(decode_prediction(self.tokenizer, generated, self.config['inference']['max_new_tokens']))
            return rows
        finally:
            self.model.config.use_cache = previous_cache
            self.tokenizer.padding_side = previous_padding

    def train(self, pairs, schemas, output_dir):
        self.prepare()
        cfg = self.config['training']
        examples = [tokenize_teacher_pair(self.tokenizer, target_payload(r, schemas), answer,
                                         cfg['max_sequence_length']) for r, answer in pairs]
        if not examples:
            raise ValueError('No accepted teacher targets for training')
        return train_tokenized(self, examples, output_dir)


def train_tokenized(runtime, examples, output_dir):
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    if not examples:
        raise ValueError('No accepted teacher targets for training')
    cfg = runtime.config['training']
    world_size = int(os.environ.get('WORLD_SIZE', '1')) if runtime.distributed else 1
    if cfg['gradient_accumulation_steps'] % world_size:
        raise ValueError('gradient_accumulation_steps must be divisible by GPU process count')
    accumulation = cfg['gradient_accumulation_steps'] // world_size
    runtime.model.config.use_cache = False
    # Lighteval and inference must not leave the global autograd state disabled.
    torch.set_grad_enabled(True)
    runtime.model.train()
    trainer = None
    try:
        args = SFTConfig(
            output_dir=str(output_dir), per_device_train_batch_size=1,
            gradient_accumulation_steps=accumulation,
            num_train_epochs=cfg['epochs_per_round'],
            max_steps=min(cfg['max_steps_per_round'],
                          math.ceil(len(examples) / cfg['gradient_accumulation_steps']) * cfg['epochs_per_round']),
            learning_rate=cfg['learning_rate'], max_length=cfg['max_sequence_length'],
            completion_only_loss=True, packing=False,
            dataset_kwargs={'skip_prepare_dataset': True},
            gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
            ddp_find_unused_parameters=False, ddp_timeout=300,
            fp16=True, bf16=False, optim='adamw_torch', report_to='none',
            save_strategy='no', eval_strategy='no', logging_steps=1,
            seed=runtime.config['random_seed'], remove_unused_columns=False)
        if args.n_gpu != 1 or args.world_size != world_size:
            raise ValueError('Trainer topology does not match worker layout; use one torchrun process per GPU')
        trainer = SFTTrainer(model=runtime.model, args=args, train_dataset=Dataset.from_list(examples),
                             processing_class=runtime.tokenizer,
                             data_collator=DataCollatorForLanguageModeling(
                                 pad_token_id=runtime.tokenizer.pad_token_id, completion_only_loss=True))
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
        runtime.model = trainer.accelerator.unwrap_model(trainer.model)
        trainer.accelerator.wait_for_everyone()
        checkpoint = Path(output_dir) / 'adapter'
        checkpoint.mkdir(parents=True, exist_ok=True)
        if trainer.is_world_process_zero():
            runtime.model.save_pretrained(checkpoint, safe_serialization=True)
            runtime.tokenizer.save_pretrained(checkpoint)
            atomic_json(Path(output_dir) / 'training_metrics.json', result.metrics)
        trainer.accelerator.wait_for_everyone()
        return str(checkpoint)
    finally:
        runtime.model.eval()
        if trainer is not None:
            del trainer
        gc.collect()
        clear_cuda_cache(torch)
