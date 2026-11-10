"""Small integration-test settings; the ordinary loop still does all the work."""
import copy
from pathlib import Path


def smoke_configs(parent_config, loop_config, output_dir):
    parent, loop = copy.deepcopy(parent_config), copy.deepcopy(loop_config)
    loop.update(max_rounds=1, scenarios_per_round=8,
                development_examples_per_task=1, development_fill_attempts=1,
                output_dir=str(Path(output_dir)))
    loop['evaluation'] = {
        'mode': 'fixed',
        'validation_path': 'data/evaluation/fixed_validation_v1.jsonl',
        'manifest_path': 'data/evaluation/fixed_validation_v1.manifest.json',
    }
    loop.setdefault('quality', {}).update(minimum_deterministic_check_rate=0.0, deterministic_template_fraction=0.0)
    loop['training']['max_steps_per_round'] = 2
    parent['parent']['max_retries'] = 1
    parent['generation'].update(max_retries_per_batch=1, max_consecutive_failed_batches=2,
                                 max_answer_attempts_per_run=1)
    return parent, loop
