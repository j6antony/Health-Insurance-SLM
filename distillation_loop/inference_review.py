"""Run standalone Qwen inference and save one reviewable JSON object per case."""

import argparse
import json
import os
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from self_instruct_pipeline.config import ROOT
from self_instruct_pipeline.persistence import append_jsonl, read_jsonl
from self_instruct_pipeline.prompt_builder import target_payload

from .core import load_loop_config
from .github_backup import (
    REPOSITORY, adapter_training_settings, restore_latest_release, upload_inference_predictions,
)
from .student import StudentRuntime, require_single_gpu


def load_runtime(model_name, adapter, config):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    require_single_gpu(torch)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={'': 0},
        dtype=torch.float16,
        attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    runtime = StudentRuntime(model, tokenizer, config)
    # The model is ready for inference. Skip prepare(), which initializes trainable LoRA.
    runtime.prepared = True
    return runtime


def load_unsloth_runtime(model_name, adapter, config, release):
    if adapter is not None:
        settings = release['training'] if release else adapter_training_settings(adapter, model_name)
        if settings is None:
            raise ValueError('This adapter is incompatible with the notebook Unsloth loader')
        config['training'].update(settings)

    from .colab import UnslothStudentRuntime

    runtime = UnslothStudentRuntime.load(model_name, config)
    if adapter is not None:
        runtime.restore(adapter)
    return runtime


def save_predictions(runtime, cases, schemas, output, adapter, release=None, *, backend='transformers'):
    model = {**runtime.identity(), 'adapter': str(adapter) if adapter else None}
    model['backend'] = backend
    if release is not None:
        model['adapter'] = {key: release[key] for key in ('repository', 'tag', 'sha256')}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a fresh file so simultaneous runs cannot write the same output.
    with output.open('x'):
        pass
    for index, case in enumerate(cases, 1):
        payload = target_payload(case, schemas)
        print(f'Case {index}/{len(cases)} ({case["id"]})', flush=True)
        prediction = runtime.infer(payload)
        append_jsonl(output, {
            'case_id': case['id'],
            'task': case['task'],
            'input': payload,
            'model': model,
            'prediction': prediction,
        })


def publish_predictions(output, cases, schemas, token):
    rows = read_jsonl(output)
    release = rows[0]['model']['adapter'] if rows else None
    if (not isinstance(release, dict) or len(rows) != len(cases)
            or any(row['case_id'] != case['id'] or row['task'] != case['task']
                   or row['input'] != target_payload(case, schemas)
                   or row['model']['adapter'] != release or 'raw' not in row['prediction']
                   for row, case in zip(rows, cases))):
        raise ValueError('GitHub upload requires a complete prediction file for these cases '
                         'from a single GitHub adapter release. Use a new output path to rerun inference.')
    url = upload_inference_predictions(output, release, token)
    print(f'GitHub predictions: {url}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=ROOT / 'data/evaluation/inference_review_v1.jsonl')
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument('--adapter', type=Path, help='Local directory of the extracted LoRA adapter')
    checkpoint.add_argument('--base', action='store_true', help='Use base model weights')
    parser.add_argument('--output', type=Path, required=True, help='Prediction JSONL file')
    parser.add_argument('--upload-to-github', action='store_true',
                        help='Attach predictions to the adapter release, or upload an existing complete file')
    parser.add_argument('--unsloth', action='store_true',
                        help='Use the Colab notebook Unsloth generation backend')
    parser.epilog = f'By default, fetch the newest compatible adapter from {REPOSITORY} Releases.'
    args = parser.parse_args()
    if args.output.exists() and not args.upload_to_github:
        parser.error(f'Output already exists: {args.output}')
    token = os.environ.get('GITHUB_TOKEN', '').strip()
    if args.upload_to_github and (args.base or args.adapter is not None or not token):
        parser.error('--upload-to-github requires a GitHub adapter and GITHUB_TOKEN with Contents write access')
    if args.adapter is not None and not (args.adapter / 'adapter_config.json').is_file():
        parser.error('--adapter must point to the directory containing adapter_config.json')

    cases = read_jsonl(args.cases)
    if not cases or len({case['id'] for case in cases}) != len(cases):
        parser.error('The cases file must contain at least one case and unique case IDs')
    schemas = json.loads((ROOT / 'data/self_instruct/task_schemas.json').read_text())['tasks']
    for case in cases:
        target_payload(case, schemas)
    if args.output.exists():
        publish_predictions(args.output, cases, schemas, token)
        return
    config = load_loop_config()
    with ExitStack() as cleanup:
        adapter = args.adapter
        release = None
        if adapter is None and not args.base:
            staging = cleanup.enter_context(TemporaryDirectory(prefix='inference-adapter-'))
            release = restore_latest_release(staging, args.model)
            if release is None:
                parser.exit(1, f'No compatible adapter release found in {REPOSITORY} for {args.model}.\n')
            adapter = Path(release['checkpoint'])
            print(f'Using {release["repository"]} release {release["tag"]} '
                  f'(SHA-256 {release["sha256"]})', flush=True)
        if args.unsloth:
            runtime = load_unsloth_runtime(args.model, adapter, config, release)
        else:
            runtime = load_runtime(args.model, adapter, config)
        save_predictions(runtime, cases, schemas, args.output, adapter, release,
                         backend='unsloth' if args.unsloth else 'transformers')
    print(f'Saved {len(cases)} predictions to {args.output}', flush=True)
    if args.upload_to_github:
        publish_predictions(args.output, cases, schemas, token)


if __name__ == '__main__':
    main()
