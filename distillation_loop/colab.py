"""Independent small single-GPU profile with Google Drive persistence."""
import copy
from collections import Counter
from pathlib import Path
from .benchmarks import load_benchmarks
from self_instruct_pipeline.persistence import atomic_json, atomic_jsonl, digest


def colab_configs(parent_config, loop_config, run_dir):
    parent, config = copy.deepcopy(parent_config), copy.deepcopy(loop_config)
    config.update(max_rounds=3, scenarios_per_round=24,
                  development_examples_per_task=5, output_dir=str(Path(run_dir)))
    config['evaluation'] = {
        'mode': 'fixed', 'validation_path': 'data/evaluation/fixed_validation_v1.jsonl',
        'manifest_path': 'data/evaluation/fixed_validation_v1.manifest.json',
        'total_examples': 100, 'subset_seed': 42,
    }
    config['training'].update(max_sequence_length=3072, epochs_per_round=1,
                              max_steps_per_round=20, gradient_accumulation_steps=4,
                              lora_rank=8, lora_alpha=16, replay_previous_rounds=True)
    config['inference'].update(max_new_tokens=1024, batch_size=1)
    config['quality'].update(minimum_new_pair_fraction=1.0,
                             minimum_deterministic_check_rate=.25,
                             deterministic_template_fraction=.25)
    return parent, config


def save_colab_subset(config, schemas, run_dir):
    """Export the exact evaluation inputs/selection manifest beside resume state."""
    rows, _ = load_benchmarks(config, schemas)
    directory = Path(run_dir); directory.mkdir(parents=True, exist_ok=True)
    path = directory/'evaluation_100_manifest.json'
    manifest = {'sha256': digest(rows), 'total': len(rows),
                'per_task': dict(Counter(x['record']['task'] for x in rows)),
                'case_ids': [x['record']['id'] for x in rows],
                'subset_seed': config['evaluation']['subset_seed'],
                'source_dataset': config['evaluation']['validation_path']}
    case_path = directory/'evaluation_100.jsonl'
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

