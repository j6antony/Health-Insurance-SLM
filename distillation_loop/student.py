import gc
import json
import math
import os
import sys
from pathlib import Path

from .adapter_identity import checkpoint_binding

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


EVIDENCE_FIRST_KEYS = ('evidence', 'missing_information', 'issues', 'criteria_assessment',
                       'requires_review', 'suggested_codes', 'document_type', 'fields', 'draft', 'answer')


def ordered_answer(answer):
    return {key: answer[key] for key in dict.fromkeys((*EVIDENCE_FIRST_KEYS, *answer)) if key in answer}


def evidence_first_payload(payload):
    schema = payload['response_schema']
    properties = ordered_answer(schema['properties'])
    return {**payload, 'instruction': payload['instruction'] +
            '\nReturn JSON with the top-level fields in this order: ' + ', '.join(properties) + '.',
            'response_schema': {**schema, 'properties': properties,
                                'required': [key for key in properties if key in schema['required']]}}


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
        self.restored_adapter = None
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
        self.restored_adapter = None
        self.prepare()
        binding = checkpoint_binding(checkpoint)
        state = load_peft_weights(binding[0], device='cpu')
        loaded = set_peft_model_state_dict(self.model, state)
        if loaded.unexpected_keys or any('lora_' in key for key in loaded.missing_keys):
            raise ValueError('Adapter restore did not match the runtime LoRA parameters')
        self.model.eval()
        if checkpoint_binding(checkpoint) != binding:
            raise ValueError('Adapter changed during restore')
        self.restored_adapter = binding

    def save_checkpoint(self, checkpoint):
        self.prepare()
        checkpoint = Path(checkpoint)
        checkpoint.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint, safe_serialization=True)
        self.tokenizer.save_pretrained(checkpoint)
        return str(checkpoint)

    def infer(self, payload):
        return self.infer_many([payload])[0]

    def infer_many(self, payloads, batch_size=None, *, completed=None, on_batch=None):
        """Length-bucketed left-padded decoding, with per-example EOS accounting."""
        import torch
        self.prepare()
        if batch_size is None:
            batch_size = self.config['inference'].get('batch_size', 2)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        results = list(completed) if completed is not None else [None] * len(payloads)
        if len(results) != len(payloads):
            raise ValueError('Saved generation does not match the input length')
        if self.config['training'].get('evidence_first'):
            payloads = [evidence_first_payload(payload) for payload in payloads]
        texts = [render_prompt(self.tokenizer, p) for p in payloads]
        lengths = [len(self.tokenizer(t, add_special_tokens=False)['input_ids']) for t in texts]
        valid = []
        for index, length in enumerate(lengths):
            if results[index] is not None:
                continue
            if length + self.config['inference']['max_new_tokens'] > self.config['training']['max_sequence_length']:
                results[index] = {'raw': '', 'truncated': True, 'error': 'Prompt exceeds inference context budget'}
            else:
                valid.append(index)
        valid.sort(key=lambda i: lengths[i])
        already_done = len(payloads) - len(valid)
        start = 0
        while start < len(valid):
            indices = valid[start:start + batch_size]
            try:
                predictions = self._generate_batch([texts[i] for i in indices])
            except torch.cuda.OutOfMemoryError:
                if len(indices) == 1:
                    raise
            else:
                for index, prediction in zip(indices, predictions):
                    results[index] = prediction
                start += len(indices)
                if on_batch is not None:
                    on_batch(results, batch_size)
                print(f'GPU {os.environ.get("LOCAL_RANK", "0")} inference: '
                      f'{already_done + start}/{len(payloads)}', flush=True)
                continue

            # Free the failed batch after its exception traceback has been released.
            gc.collect()
            clear_cuda_cache(torch)
            batch_size = max(1, len(indices) // 2)
            print(f'GPU inference ran out of memory; retrying with batch size {batch_size}.', flush=True)
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
            budget = self.config['inference']['max_new_tokens']
            stopping = getattr(self, 'repetition_stopping', None)
            generation_options = {}
            if stopping is not None:
                from transformers import StoppingCriteriaList

                criterion = stopping(width, budget, self.tokenizer)
                generation_options['stopping_criteria'] = StoppingCriteriaList([criterion])
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                    **generation_options)
            stopped_at = criterion.stopped_at() if stopping is not None else [0] * len(texts)
            rows = []
            for generated, stop_length in zip(output[:, width:].tolist(), stopped_at):
                row = decode_prediction(self.tokenizer, generated[:stop_length] if stop_length else generated, budget)
                if stop_length:
                    row.update(truncated=True, stop_reason='repetition')
                rows.append(row)
            return rows
        finally:
            self.model.config.use_cache = previous_cache
            self.tokenizer.padding_side = previous_padding

    def _set_training_mode(self):
        self.model.train()

    def training_pair_fits(self, record, answer, schemas):
        cfg = self.config['training']
        payload = target_payload(record, schemas)
        if cfg.get('evidence_first'):
            payload, answer = evidence_first_payload(payload), ordered_answer(answer)
        prefix = self.tokenizer(render_prompt(self.tokenizer, payload), add_special_tokens=False)['input_ids']
        completion = self.tokenizer(json.dumps(answer, ensure_ascii=False), add_special_tokens=False)['input_ids']
        return (len(prefix) + len(completion) + 1 <= cfg['max_sequence_length']
                and len(prefix) + self.config['inference']['max_new_tokens'] <= cfg['max_sequence_length'])

    def tokenize_pair(self, record, answer, schemas):
        cfg = self.config['training']
        payload = target_payload(record, schemas)
        if cfg.get('evidence_first'):
            payload, answer = evidence_first_payload(payload), ordered_answer(answer)
        return tokenize_teacher_pair(self.tokenizer, payload, answer, cfg['max_sequence_length'])

    def completion_loss(self, pairs, schemas):
        import torch
        from collections import Counter

        self.prepare()
        self.model.eval()
        device = self.model.get_input_embeddings().weight.device
        total, tokens, tasks, excluded = 0.0, 0, Counter(), Counter()
        with torch.inference_mode():
            for record, answer in pairs:
                try:
                    sample = self.tokenize_pair(record, answer, schemas)
                except ValueError as exc:
                    if str(exc) != 'Training sequence exceeds budget; no silent truncation is allowed':
                        raise
                    excluded['sequence_budget'] += 1
                    continue
                ids = torch.tensor([sample['input_ids']], device=device)
                mask = torch.tensor([sample['completion_mask']], device=device, dtype=torch.bool)
                labels = ids.masked_fill(~mask, -100)
                labels[:, 0] = -100
                count = sum(sample['completion_mask'][1:])
                loss = float(self.model(input_ids=ids, labels=labels, use_cache=False).loss)
                if not math.isfinite(loss):
                    raise ValueError(f'Nonfinite held-out completion loss for {record["id"]}')
                total += loss * count
                tokens += count
                tasks[record['task']] += 1
        return {'completion_loss': total / tokens if tokens else None, 'completion_tokens': tokens,
                'valid_examples': sum(tasks.values()), 'per_task_examples': dict(tasks),
                'excluded': dict(excluded)}

    def train(self, pairs, schemas, output_dir):
        self.prepare()
        examples = [self.tokenize_pair(record, answer, schemas) for record, answer in pairs]
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
    microbatch_size = cfg.get('per_device_train_batch_size', 1)
    effective_batch_size = microbatch_size * accumulation * world_size
    runtime.model.config.use_cache = False
    # Lighteval and inference must not leave the global autograd state disabled.
    torch.set_grad_enabled(True)
    runtime.restored_adapter = None
    runtime._set_training_mode()
    trainer = None
    try:
        args = SFTConfig(
            output_dir=str(output_dir), per_device_train_batch_size=microbatch_size,
            gradient_accumulation_steps=accumulation,
            num_train_epochs=cfg['epochs_per_round'],
            max_steps=min(cfg['max_steps_per_round'],
                          math.ceil(len(examples) / effective_batch_size) * cfg['epochs_per_round']),
            learning_rate=cfg['learning_rate'], lr_scheduler_type=cfg.get('lr_scheduler_type', 'linear'), max_length=cfg['max_sequence_length'],
            completion_only_loss=True, packing=False,
            dataset_kwargs={'skip_prepare_dataset': True},
            gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
            ddp_find_unused_parameters=False, ddp_timeout=300,
            fp16=True, bf16=False, optim='adamw_torch', weight_decay=cfg.get('weight_decay', 0.0), report_to='none',
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
            atomic_json(Path(output_dir) / 'training_metrics.json', {
                **result.metrics, 'train_microbatch_size': microbatch_size,
                'gradient_accumulation_steps': accumulation, 'effective_train_batch_size': effective_batch_size,
            })
        trainer.accelerator.wait_for_everyone()
        return str(checkpoint)
    finally:
        runtime.model.eval()
        if trainer is not None:
            del trainer
        gc.collect()
        clear_cuda_cache(torch)
