"""Invoked by torchrun, never by executing notebook cells on multiple ranks."""
import json
import os
import sys
from pathlib import Path


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from peft import PeftModel
    from self_instruct_pipeline.persistence import atomic_jsonl
    from .student import StudentRuntime
    from .evaluation import require_training_versions
    require_training_versions()
    job_path = Path(sys.argv[1])
    job = json.loads(job_path.read_text())
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    set_seed(job['config']['random_seed'])
    tokenizer = AutoTokenizer.from_pretrained(job['model_name'], token=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(job['model_name'], token=False,
        device_map={'': rank}, dtype=torch.float16, attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16), low_cpu_mem_usage=True)
    # Prepare the quantized base BEFORE loading trainable saved adapters.
    runtime = StudentRuntime(model, tokenizer, job['config'], distributed=True)
    runtime.prepare()
    if job['adapter']:
        runtime.restore(job['adapter'])
    if job['operation'] == 'initialize':
        if rank == 0:
            runtime.save_checkpoint(job['checkpoint'])
    elif job['operation'] == 'infer':
        indices = list(range(rank, len(job['payloads']), world))
        predictions = runtime.infer_many([job['payloads'][i] for i in indices], job['batch_size'])
        atomic_jsonl(job_path.parent/f'predictions_{rank}.jsonl',
                     [{'index': index, 'prediction': prediction} for index, prediction in zip(indices, predictions)])
    elif job['operation'] == 'train':
        from .student import train_tokenized
        train_tokenized(runtime, job['examples'], Path(job['output_dir']))
    else:
        raise ValueError('Unknown GPU worker operation')
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
