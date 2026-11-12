"""Independent small single-GPU profile with Google Drive persistence."""
import copy
from collections import Counter
from pathlib import Path
from .benchmarks import load_benchmarks
from .student import StudentRuntime, require_single_gpu
from self_instruct_pipeline.persistence import atomic_json, atomic_jsonl, digest


def colab_configs(parent_config, loop_config, run_dir):
    parent, config = copy.deepcopy(parent_config), copy.deepcopy(loop_config)
    config.update(max_rounds=5, scenarios_per_round=24,
                  development_examples_per_task=5, output_dir=str(Path(run_dir)))
    config['evaluation'] = {
        'mode': 'fixed', 'validation_path': 'data/evaluation/fixed_validation_v1.jsonl',
        'manifest_path': 'data/evaluation/fixed_validation_v1.manifest.json',
        'total_examples': 100, 'subset_seed': 42,
        **({'holdout_families_per_task': 2} if config.get('student_conditioned_training') else {}),
    }
    config['training'].update(max_sequence_length=3072, epochs_per_round=1,
                              max_steps_per_round=20, gradient_accumulation_steps=4,
                              lora_rank=8, lora_alpha=16, replay_previous_rounds=True)
    config['inference'].update(max_new_tokens=1024, batch_size=1)
    config['quality'].update(minimum_new_pair_fraction=1.0,
                             minimum_deterministic_check_rate=.25,
                             deterministic_template_fraction=.25)
    return parent, config



def distillation_1000_configs(parent_config, loop_config, run_dir):
    parent, config = colab_configs(parent_config, loop_config, run_dir)
    config.update(max_rounds=5, scenarios_per_round=200, adaptive_pairs_per_round=80,
                  student_conditioned_training=True, stop_when_thresholds_met=False,
                  balanced_curriculum=True, stratified_screening=True, rollback_on_regression=True, preference_training=True,
                  development_examples_per_task=20, feedback_strength=2.0)
    config['evaluation'].update(holdout_families_per_task=2, total_examples=360,
                                semantic_judge=True)
    config['training'].update(max_steps_per_round=100, learning_rate=3e-5,
                              lora_rank=32, lora_alpha=32, lora_dropout=0.0, use_rslora=True,
                              evidence_first=True, weight_decay=.01, replay_previous_rounds=False,
                              replay_fraction=.15, lr_scheduler_type='cosine')
    config['quality'].update(max_task_regression=.05, minimum_deterministic_check_rate=.15,
                             deterministic_template_fraction=.20)
    parent['generation']['scenario_blueprints'] = True
    return parent, config


def l4_runtime_config(config, *, train_batch_size=4, inference_batch_size=8):
    """Apply execution batch sizes while preserving the run's optimizer batch."""
    training = config['training']
    effective_batch = training.get('per_device_train_batch_size', 1) * training['gradient_accumulation_steps']
    if type(train_batch_size) is not int or train_batch_size < 1 or effective_batch % train_batch_size:
        raise ValueError(f'Training batch must be a positive divisor of {effective_batch}')
    if type(inference_batch_size) is not int or inference_batch_size < 1:
        raise ValueError('Inference batch must be a positive integer')
    runtime_config = copy.deepcopy(config)
    runtime_config['training'].update(
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=effective_batch // train_batch_size,
    )
    runtime_config['inference']['batch_size'] = inference_batch_size
    return runtime_config


def save_colab_subset(config, schemas, run_dir):
    """Export the exact evaluation inputs/selection manifest beside resume state."""
    rows, _ = load_benchmarks(config, schemas)
    directory = Path(run_dir); directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'evaluation_{len(rows)}_manifest.json'
    manifest = {'sha256': digest(rows), 'total': len(rows),
                'per_task': dict(Counter(x['record']['task'] for x in rows)),
                'case_ids': [x['record']['id'] for x in rows],
                'subset_seed': config['evaluation']['subset_seed'],
                'source_dataset': config['evaluation']['validation_path']}
    case_path = directory/f'evaluation_{len(rows)}.jsonl'
    if path.exists():
        import json
        from self_instruct_pipeline.persistence import read_jsonl
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Colab evaluation subset changed; use a new run folder')
        if case_path.exists() and digest(read_jsonl(case_path)) != manifest['sha256']:
            raise ValueError('Saved Colab evaluation cases have changed')
        if not case_path.exists():
            atomic_jsonl(case_path,rows)
    else:
        atomic_jsonl(case_path,rows)
        atomic_json(path,manifest)
    return manifest



class UnslothStudentRuntime(StudentRuntime):
    """Single-GPU Colab runtime using Unsloth for QLoRA and student generation."""

    @classmethod
    def load(cls, model_name, config):
        from unsloth import FastLanguageModel
        import torch
        from transformers import BitsAndBytesConfig, set_seed

        require_single_gpu(torch)
        cfg = config['training']
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name, max_seq_length=cfg['max_sequence_length'],
            dtype=torch.float16, load_in_4bit=True, use_exact_model_name=True,
            device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
            ),
        )
        set_seed(config['random_seed'])
        model = FastLanguageModel.get_peft_model(
            model, r=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'],
            lora_dropout=cfg['lora_dropout'], use_rslora=cfg.get('use_rslora', False), bias='none',
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                            'gate_proj', 'up_proj', 'down_proj'],
            use_gradient_checkpointing='unsloth', random_state=config['random_seed'],
        )
        runtime = cls(model, tokenizer, config)
        runtime.prepare()
        return runtime

    def infer_many(self, payloads, batch_size=None):
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(self.model)
        return super().infer_many(payloads, batch_size)

    def _set_training_mode(self):
        from unsloth import FastLanguageModel

        FastLanguageModel.for_training(self.model)

    def train_preferences(self, pairs, preferences, schemas, output_dir):
        """Apply one low-rate ORPO pass to verified preference pairs."""
        import gc
        import json
        import math
        import torch
        from datasets import Dataset
        from trl import ORPOConfig, ORPOTrainer
        from self_instruct_pipeline.prompt_builder import target_payload
        from .student import clear_cuda_cache, evidence_first_payload, ordered_answer, render_prompt, tokenize_teacher_pair

        cfg = self.config['training']
        rejected = {row['scenario_id']: row['rejected'] for row in preferences}
        rows = []
        for record, answer in pairs:
            if record['id'] not in rejected:
                continue
            payload = target_payload(record, schemas)
            bad = rejected[record['id']]
            if cfg.get('evidence_first'):
                payload = evidence_first_payload(payload)
                answer = ordered_answer(answer)
            # Leave room for the boundary and special tokens added by TRL.
            limit = cfg['max_sequence_length'] - 4
            try:
                tokenize_teacher_pair(self.tokenizer, payload, answer, limit)
                tokenize_teacher_pair(self.tokenizer, payload, bad, limit)
            except ValueError:
                continue
            rows.append({'prompt': render_prompt(self.tokenizer, payload),
                         'chosen': json.dumps(answer, ensure_ascii=False),
                         'rejected': json.dumps(bad, ensure_ascii=False)})
        if not rows:
            return self.save_checkpoint(Path(output_dir) / 'adapter')

        self._set_training_mode()
        self.model.config.use_cache = False
        trainer = None
        try:
            args = ORPOConfig(
                output_dir=str(output_dir), learning_rate=1e-5, beta=.1,
                max_length=cfg['max_sequence_length'], max_prompt_length=cfg['max_sequence_length'],
                per_device_train_batch_size=1, gradient_accumulation_steps=4,
                num_train_epochs=1, lr_scheduler_type='cosine', fp16=True, bf16=False,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                optim='adamw_torch', weight_decay=cfg.get('weight_decay', 0.0), report_to='none', save_strategy='no',
                seed=self.config['random_seed'])
            trainer = ORPOTrainer(model=self.model, args=args, train_dataset=Dataset.from_list(rows),
                                  processing_class=self.tokenizer)
            result = trainer.train()
            if not math.isfinite(result.training_loss):
                raise ValueError('Preference training produced a nonfinite loss')
            self.model = trainer.accelerator.unwrap_model(trainer.model)
            atomic_json(Path(output_dir) / 'training_metrics.json',
                        {**result.metrics, 'preference_pairs': len(rows)})
            return self.save_checkpoint(Path(output_dir) / 'adapter')
        finally:
            self.model.eval()
            del trainer
            gc.collect()
            clear_cuda_cache(torch)
