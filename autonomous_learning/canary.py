"""Prepare bounded API/GPU canaries without invoking paid services."""
import argparse
import json
from pathlib import Path
from self_instruct_pipeline.parent_backend import OpenAIParent
from self_instruct_pipeline.budget import reservation_usd
from .scenarios import make_scenario
from .rendering import render, verify
from .tasks import project
from .teacher import request, RESPONSE_SCHEMA, RenderQueue
from .storage import Storage, digest, read_json
from .planner import heuristic
from .jobs import prepare
from .model import MODEL, MODEL_REVISION


def api_plan():
    masters = [make_scenario(191119, 0, 'hospitalization', False, 'valid', 7),
               make_scenario(191119, 1, 'diagnostic', True, 'date_conflict', 4),
               make_scenario(191119, 2, 'hospitalization', False, 'unreadable', 5),
               make_scenario(191119, 3, 'diagnostic', None, 'incomplete', 8)]
    settings = read_json(Path(__file__).resolve().parents[1] / 'config/self_instruct.json')['parent']
    settings = {**settings, 'max_output_tokens': 768, 'max_retries': 1, 'reasoning_effort': 'none', 'timeout_seconds': 60, 'retry_network_errors': False}
    requests = []
    for s in masters:
        rendered = render(s)
        verify(s, rendered)
        for d in s['documents']:
            if d['type'] not in ('discharge_summary', 'investigation_report') or d['readability'] != 'readable':
                continue
            messages = request(d)
            body = OpenAIParent.request_body(settings, messages, RESPONSE_SCHEMA)
            # GPT token counts are an estimate, not measured by the Qwen tokenizer.
            input_bytes = len(json.dumps(body, ensure_ascii=False).encode())
            requests.append({'document': d, 'family_id': s['family_id'],
                             'estimated_input_tokens': (input_bytes + 3) // 4,
                             'input_token_reservation_bound': input_bytes + 4096,
                             'max_output_tokens': settings['max_output_tokens'],
                             'reserved_usd_per_attempt': str(reservation_usd(settings, body))})
    reserved = sum(float(r['reserved_usd_per_attempt']) * (settings['max_retries'] + 1) for r in requests)
    return {'version': 1, 'status': 'prepared_not_authorized', 'settings': settings, 'masters': masters,
            'requests': requests, 'planned_unique_requests': len(requests), 'max_attempts': len(requests) * 2,
            'max_api_usd': .02, 'estimated_reservation_all_attempts_usd': reserved,
            'estimated_input_tokens_unique': sum(r['estimated_input_tokens'] for r in requests),
            'max_output_tokens_unique': sum(r['max_output_tokens'] for r in requests),
            'operation': 'document_rendering', 'endpoint': 'POST /v1/responses', 'concurrency': 1,
            'latency': {'measured': None, 'planning_estimate_seconds': [10, 120], 'timeout_bound_seconds_all_attempts': len(requests) * 2 * 60,
                        'note': 'Unmeasured estimate; transport backoff and network may add time.'},
            'expected_projected_examples': len(masters) * 7, 'max_selected_examples': len(masters) * 8,
            'pricing_verified': '2026-11-19', 'pricing_source': 'https://developers.openai.com/api/docs/models/gpt-6-luna',
            'verification': ['same source facts after independent final-text parse', 'strict schema and rule findings unchanged',
                             'exact task-scoped spans; arbitrary prose quarantined'],
            'useful_diversity': 'At least two verified teacher surface forms outside the nine-style deterministic baseline bank, with no fact or label changes. Surface novelty is not a downstream model-quality result.',
            'unreadable_document_is_python_only': True, 'incomplete_document_rendered_without_inventing_fields': True,
            'no_duplicate_billing_promise': 'Completed queue keys are cached; ambiguous interrupted calls remain uncertain and require provider reconciliation.'}


def prepare_api(storage):
    plan = api_plan()
    if plan['estimated_reservation_all_attempts_usd'] > plan['max_api_usd']:
        raise ValueError('Canary reservation exceeds hard budget')
    receipt = storage.write('experiments/api-canary-20261119/plan.json', plan)
    return receipt


def run_api(storage, plan, authorized=False):
    """Explicitly invoked later; never used during canary preparation."""
    if not authorized:
        raise ValueError('API canary is not authorized')
    if plan != api_plan():
        raise ValueError('Prepared canary contract changed; review a new plan')
    queue = RenderQueue(storage)
    keys = [queue.enqueue(r['document'], plan['settings']) for r in plan['requests']]
    # Isolate this canary from unrelated pending jobs on the shared queue.
    import time
    from .storage import digest
    from .dataset import build
    from .diversity import surface_signature
    started = time.monotonic()
    queue.run(plan['settings'], paid_authorized=True, max_calls=plan['max_attempts'],
              max_cost_usd=plan['max_api_usd'], concurrency=1, request_keys=keys)
    from .rendering import render_document, LAYOUT_COUNT
    accepted, quarantine, changed, surfaces = [], [], 0, set()
    novel_documents = 0
    for s in plan['masters']:
        rendered = render(s)
        failed = False
        for item, key in zip(plan['requests'], keys):
            if item['family_id'] != s['family_id']:
                continue
            result = queue.result(key)
            if result['status'] != 'done':
                quarantine.append({'family_id': s['family_id'], 'request_key': key, 'reason': result['status']})
                failed = True
                continue
            for page in rendered['pages']:
                if page['document_id'] == item['document']['id']:
                    text = result['result']['text']
                    if text != page['text']:
                        changed += 1
                        signature = surface_signature({**page, 'text': text})
                        baseline_bank = {surface_signature(render_document(item['document'], style)) for style in range(LAYOUT_COUNT)}
                        if signature not in baseline_bank:
                            novel_documents += 1
                            surfaces.add(signature)
                    page['text'] = text
        if failed:
            continue
        rendered['render_hash'] = digest(rendered['pages'])
        verify(s, rendered)
        accepted.extend(project(s, rendered))
    dataset = build(accepted)
    summary = {'status': 'canary_completed', 'elapsed_seconds': time.monotonic() - started,
               'verified_changed_teacher_documents': changed, 'novel_teacher_documents': novel_documents, 'distinct_novel_surfaces': len(surfaces),
               'useful_diversity_passed': novel_documents >= 2 and len(surfaces) >= 2,
               'statistics': dataset['statistics'], 'quarantine': quarantine,
               'api_usage': 'api_usage/rendering/api_usage.jsonl', 'release_ready': False}
    storage.write('experiments/api-canary-20261119/verified_examples.json', dataset)
    storage.write('experiments/api-canary-20261119/result.json', summary)
    return summary


def prepare_gpu(storage, exported, adapter=None, replay_examples=None):
    plan = {**heuristic(count=8), 'max_gpu_usd': 1.0}
    overrides = {'max_steps_per_round': 2, 'per_device_train_batch_size': 1,
                 'gradient_accumulation_steps': 4, 'learning_rate': 5e-5}
    from .dataset import build, export, migrate_verified_replay
    from .storage import checksum
    for name, expected in exported['manifest']['files'].items():
        if checksum(storage.path(exported['directory']) / name) != expected:
            raise ValueError('Canary source dataset checksum mismatch')
    source_examples = storage.path(exported['directory']) / 'examples.jsonl'
    new_rows = [json.loads(line) for line in source_examples.read_text().splitlines()]
    replay = []
    if replay_examples is not None:
        replay_manifest = Path(replay_examples).parent / 'manifest.json'
        if replay_manifest.exists() and checksum(replay_examples) != read_json(replay_manifest)['files']['examples.jsonl']:
            raise ValueError('Replay source checksum mismatch')
        replay = migrate_verified_replay([json.loads(line) for line in Path(replay_examples).read_text().splitlines()])
    fresh_export = export(storage, build(new_rows, replay=replay, initialization_strategy='fresh'))
    fresh = prepare(storage, fresh_export, plan, training_overrides=overrides,
                    inference_overrides={'max_new_tokens': 1536})
    continued = {'status': 'blocked_adapter_unavailable', 'required': 'verified existing best compatible adapter path'}
    if adapter is not None:
        continued_export = export(storage, build(new_rows, replay=replay, initialization_strategy='continued'))
        cfg = read_json(Path(adapter) / 'adapter_config.json')
        continued_overrides = {**overrides, 'lora_rank': cfg['r'], 'lora_alpha': cfg['lora_alpha'],
                               'lora_dropout': cfg['lora_dropout'], 'use_rslora': cfg.get('use_rslora', False),
                               'learning_rate': 2e-5}
        continued = prepare(storage, continued_export, {**plan, 'adapter_strategy': 'continued'}, adapter,
                            training_overrides=continued_overrides, inference_overrides={'max_new_tokens': 1536})
        continued['dataset'] = continued_export
    fresh['dataset'] = fresh_export
    return storage.write('experiments/gpu-canary-20261119/strategies.json', {
        'status': 'prepared_not_authorized', 'model': MODEL, 'revision': MODEL_REVISION,
        'fresh': fresh, 'continued': continued,
        'replay_source_sha256': checksum(replay_examples) if replay_examples else None,
        'published_continued_candidate': read_json(Path(__file__).resolve().parents[1] / 'config/continued_adapter_candidate.json'), 'max_gpu_usd_proposed': 1.0,
        'steps': 2, 'max_sequence_length': 4096, 'hardware_proposal': 'One 24 GiB CUDA GPU, provider not selected',
        'memory': '4-bit base weight lower bound about 4 GB; quantization metadata, LoRA, optimizer and activations add memory. Peak unmeasured.',
        'approval_required': ['GPU provider', 'hardware', 'hourly rate', 'spending limit', 'training strategy'],
        'old_macro_0_558': 'Previous evaluation contract; not comparable to these seven tasks',
        'continued_dataset_requirement': 'Verified new examples plus task-stratified replay under this evidence policy; legacy labels require explicit revalidation.',
        'fresh_dataset_requirement': 'Cumulative verified approved corpus under this evidence policy.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('prepare-api', 'run-api', 'prepare-gpu'))
    p.add_argument('--root', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--replay-examples', type=Path)
    p.add_argument('--authorize-api', action='store_true')
    args = p.parse_args()
    storage = Storage(args.root)
    storage.initialize()
    with storage.lock():
        if args.operation == 'prepare-api':
            result = prepare_api(storage)
        elif args.operation == 'run-api':
            result = run_api(storage, read_json(storage.path('experiments/api-canary-20261119/plan.json')), args.authorize_api)
        else:
            if args.manifest is None:
                p.error('--manifest is required')
            result = prepare_gpu(storage, {'manifest': read_json(args.manifest),
                                 'directory': str(args.manifest.parent.resolve().relative_to(storage.root.resolve()))}, args.adapter, args.replay_examples)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
