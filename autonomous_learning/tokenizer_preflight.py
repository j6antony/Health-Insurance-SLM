"""CPU-only real-tokenizer and SFT collator preflight; never loads model weights."""
import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path

from distillation_loop.student import render_prompt, tokenize_teacher_pair
from .model import MODEL, MODEL_REVISION
from .storage import atomic_json, checksum, digest


def distribution(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    return {'count': len(values), 'sum': sum(values), 'min': values[0], 'max': values[-1],
            **{name: values[round((len(values) - 1) * p)] for name, p in (('p50', .5), ('p90', .9), ('p95', .95), ('p99', .99))}}


def analyze(rows, tokenizer, max_length=4096, check_collator=True, check_cpu_config=True):
    lengths, samples, tasks = [], [], defaultdict(list)
    for row in rows:
        prompt = render_prompt(tokenizer, row['payload'])
        prefix = tokenizer(prompt, add_special_tokens=False)['input_ids']
        completion = tokenizer(json.dumps(row['target'], ensure_ascii=False), add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
        entry = {'id': row['id'], 'task': row['task'], 'prompt': len(prefix), 'completion': len(completion),
                 'total': len(prefix) + len(completion), 'overlength': len(prefix) + len(completion) > max_length}
        lengths.append(entry)
        tasks[row['task']].append(entry['completion'])
        if not entry['overlength']:
            sample = tokenize_teacher_pair(tokenizer, row['payload'], row['target'], max_length)
            if sample['input_ids'] != prefix + completion or sample['completion_mask'] != [0] * len(prefix) + [1] * len(completion):
                raise ValueError('Completion mask does not align with the non-thinking prompt')
            samples.append(sample)
        if '<think>\n\n</think>' not in prompt and '<think></think>' not in prompt:
            raise ValueError('Pinned Qwen non-thinking assistant prefix was not emitted')
    collator_result = 'not_requested'
    if check_collator:
        from datasets import Dataset
        from trl import SFTConfig
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
        if check_cpu_config:
            config = SFTConfig(output_dir='/tmp/health-slm-cpu-preflight', use_cpu=True, fp16=False, bf16=False,
                           completion_only_loss=True, packing=False, max_length=max_length,
                           dataset_kwargs={'skip_prepare_dataset': True}, report_to='none')
        dataset = Dataset.from_list(samples)
        collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                                 completion_only_loss=True)
        for start in range(0, len(samples), 4):
            subset = [dataset[i] for i in range(start, min(start + 4, len(dataset)))]
            batch = collator(subset)
            for index, sample in enumerate(subset):
                n = len(sample['input_ids'])
                boundary = sample['completion_mask'].index(1)
                labels = batch['labels'][index]
                if not bool((labels[:boundary] == -100).all()) or labels[boundary:n].tolist() != sample['input_ids'][boundary:] or not bool((labels[n:] == -100).all()):
                    raise ValueError('Actual TRL collator prompt/completion/padding mask mismatch')
        collator_result = ('passed_actual_trl_collator_and_dataset_and_cpu_sft_config' if check_cpu_config
                           else 'passed_actual_trl_collator_and_dataset_without_accelerate_device_setup')
    return {'model': MODEL, 'revision': MODEL_REVISION, 'max_context': max_length,
            'examples': len(rows), 'overlength_examples': sum(e['overlength'] for e in lengths),
            'prompt_tokens': distribution([e['prompt'] for e in lengths]),
            'completion_tokens': distribution([e['completion'] for e in lengths]),
            'sequence_tokens': distribution([e['total'] for e in lengths]),
            'completion_tokens_by_task': {t: distribution(v) for t, v in tasks.items()},
            'lengths': lengths, 'non_thinking_template': 'passed', 'loss_mask_alignment': 'passed_for_all_fitting_examples',
            'collator': collator_result, 'truncation': False,
            'versions': {p: importlib.metadata.version(p) for p in ('transformers', 'tokenizers')},
            'status': 'passed' if not any(e['overlength'] for e in lengths) else 'blocked_overlength'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--examples', type=Path, required=True)
    p.add_argument('--tokenizer', default=MODEL, help='Pinned public model or previously downloaded tokenizer directory')
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--compare-scoped-evidence', action='store_true', help='Compare legacy targets to deterministically reconstructed minimal targets on identical inputs')
    args = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=MODEL_REVISION, token=False,
                                             cache_dir=args.cache_dir)
    rows = [json.loads(line) for line in args.examples.read_text().splitlines()]
    result = {'dataset_sha256': checksum(args.examples), 'before': analyze(rows, tokenizer, args.max_length)}
    if args.compare_scoped_evidence:
        from .tasks import target_from_visible
        scoped = [{**row, 'target': target_from_visible(row)} for row in rows]
        result['after'] = analyze(scoped, tokenizer, args.max_length)
        result['evidence_spans_before'] = sum(len(r['target']['evidence']) for r in rows)
        result['evidence_spans_after'] = sum(len(r['target']['evidence']) for r in scoped)
    atomic_json(args.output, result)
    print(json.dumps({k: {x: y for x, y in v.items() if x != 'lengths'} if isinstance(v, dict) else v for k, v in result.items()}, indent=2))
    return 0 if result.get('after', result['before'])['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
