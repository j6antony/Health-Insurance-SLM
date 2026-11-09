import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def load_config(path=None):
    path = Path(path or ROOT / 'config/self_instruct.json').resolve()
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or set(config) != {'sampling', 'generation', 'parent', 'student', 'paths'}:
        raise ValueError('Config requires sampling, generation, parent, student, and paths only')
    if any(not isinstance(config[key], dict) for key in config):
        raise ValueError('Configuration sections must be objects')
    s, g, p = config['sampling'], config['generation'], config['parent']
    if set(p) != {'provider', 'model', 'api_key_env', 'max_output_tokens', 'temperature', 'timeout_seconds', 'max_retries'}:
        raise ValueError('Unknown parent setting; credentials must use api_key_env, never config values')
    if s['mode'] != 'balanced':
        raise ValueError('Only balanced sampling is supported')
    positive_int(s['num_seed_examples'], 'num_seed_examples', 6)
    if s['num_seed_examples'] > 8:
        raise ValueError('Use 6-8 demonstrations')
    positive_int(s['random_seed'], 'random_seed', 0)
    if type(s['machine_demonstration_fraction']) not in (float, int) or not 0 <= s['machine_demonstration_fraction'] <= 0.25:
        raise ValueError('machine_demonstration_fraction must be between 0 and 0.25')
    for key in ('batch_size', 'target_accepted_scenarios', 'max_consecutive_failed_batches', 'max_answer_attempts_per_run'):
        positive_int(g[key], key)
    positive_int(g['max_retries_per_batch'], 'max_retries_per_batch', 0)
    for key in ('duplicate_rouge_threshold', 'duplicate_context_threshold', 'semantic_duplicate_threshold'):
        if type(g[key]) not in (float, int) or not 0 < g[key] <= 1:
            raise ValueError(f'{key} must be in (0, 1]')
    if set(g['difficulty_weights']) != {'easy', 'medium', 'hard'}:
        raise ValueError('difficulty_weights requires easy, medium, hard')
    for weights in (g['difficulty_weights'], g['task_weights']):
        if not isinstance(weights, dict) or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in weights.values()):
            raise ValueError('Weights must be finite positive numbers')
    if p['provider'] != 'gemini':
        raise ValueError('This backend implements Gemini; no automatic provider substitution')
    if not isinstance(p['model'], str) or not p['model'].startswith('gemini-') or '/' in p['model']:
        raise ValueError('Specify an explicit Gemini API model identifier')
    if not isinstance(p['api_key_env'], str) or not p['api_key_env'].isidentifier():
        raise ValueError('api_key_env must name an environment variable')
    for key in ('max_output_tokens', 'timeout_seconds'):
        positive_int(p[key], key)
    positive_int(p['max_retries'], 'max_retries', 0)
    if type(p['temperature']) not in (int, float) or not 0 <= p['temperature'] <= 2:
        raise ValueError('temperature must be in [0, 2]')
    if config['student'] != {'model': 'Qwen/Qwen3-8B', 'training_in_separate_pipeline': True}:
        raise ValueError('Student must be Qwen/Qwen3-8B with separate training')
    return config


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path
