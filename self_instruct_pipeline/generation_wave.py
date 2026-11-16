"""Durable parallel generation with ordered, single-writer admission."""
from collections import Counter
import json
import uuid

from .budget import BudgetExceeded
from .deduplicator import fingerprint
from .parallel import bounded_map
from .parent_backend import ParentError
from .persistence import atomic_json, digest, now
from .prompt_builder import scenario_audit_messages
from .validation import ValidationError, batch_schema, scenario_audit_schema


def _new_wave(pipe, goal, target, task_targets, available):
    counts = pipe.scheduler.counts
    simulated = counts.copy()
    batches = []
    remaining = target - len(pipe.accepted)
    try:
        pipe.scheduler.counts = simulated
        for _ in range(available):
            size = min(remaining, pipe.config['generation']['batch_size'])
            if size <= 0:
                break
            quotas = None
            if task_targets is not None:
                quotas = {}
                for task, wanted in task_targets.items():
                    take = min(size - sum(quotas.values()), max(0, wanted - simulated[task]))
                    if take:
                        quotas[task] = take
                if not quotas:
                    raise ValueError('Task quotas do not cover the remaining generation target')
            plan = pipe.plan(count=size, task_quotas=quotas)
            simulated.update(plan['task_quotas'])
            remaining -= sum(plan['task_quotas'].values())
            batches.append({'id': str(uuid.uuid4()), 'plan': plan})
    finally:
        pipe.scheduler.counts = counts
    return {'goal': goal, 'accepted_before': len(pipe.accepted), 'batches': batches,
            'rng_state': pipe.rng.getstate(), 'seed_exposure': dict(pipe.sampler.exposure)}


def _request(client, store, job, stage, messages, schema):
    path = store.path(f"{job['id']}-{stage}.json")
    key = digest([job['id'], stage, messages, schema])
    if path.exists():
        receipt = json.loads(path.read_text())
        if receipt['key'] != key:
            raise ValueError('Generation wave request receipt changed')
        if 'blocked' in receipt:
            error = receipt['blocked']
            return ParentError(error['message'], metadata=error['metadata'])
        if 'invalid_response' in receipt:
            return ValidationError(receipt['invalid_response'])
        return receipt['result']
    try:
        result = client.generate_json(messages, schema, stage, temperature=0.0 if stage == 'scenario_audit' else None)
    except ParentError as exc:
        if not exc.retryable and exc.category != 'response':
            atomic_json(path, {'key': key, 'blocked': {'message': str(exc), 'metadata': exc.metadata}})
        elif exc.category == 'response':
            atomic_json(path, {'key': key, 'invalid_response': str(exc)})
            return ValidationError(str(exc))
        return exc
    except BudgetExceeded as exc:
        return exc
    except ValidationError as exc:
        atomic_json(path, {'key': key, 'invalid_response': str(exc)})
        return exc
    atomic_json(path, {'key': key, 'result': result})
    return result


def _execute_wave(pipe, wave, workers):
    from .persistence import Store
    replies = Store(pipe.store.path('generation_wave_receipts'))
    client = pipe.client()
    batches = wave['batches']
    journal = {r['batch_id'] for r in pipe.store.read('generation_batches.jsonl')}
    for job in batches:
        if job['id'] not in journal:
            plan = job['plan']
            pipe.store.append('generation_batches.jsonl', {
                'batch_id': job['id'], 'generation_goal': wave['goal'], 'timestamp': now(),
                **{k: plan[k] for k in ('task_quotas', 'difficulty', 'seed_ids')}})

    def generate(job):
        plan = job['plan']
        quotas = plan['task_quotas']
        return _request(client, replies, job, 'scenario_generation', plan['messages'],
                        batch_schema(quotas, sum(quotas.values())))

    raw = bounded_map(generate, batches, workers)
    prepared, fingerprints = [], set()
    for job, records in zip(batches, raw):
        _raise_blocking(pipe, job, records)
        path = replies.path(f"{job['id']}-prepared.json")
        input_hash = digest([job, str(records) if isinstance(records, ValidationError) else records])
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['input_hash'] != input_hash:
                raise ValueError('Prepared scenario audit inputs changed')
            prepared.append((job, saved['audit']))
            fingerprints.update(fingerprint(row) for row in saved['audit']['records'])
            continue
        try:
            if isinstance(records, ValidationError):
                raise records
            valid = pipe._validate_generation(records, job['plan'], job['id'])
        except ValidationError as exc:
            pipe.reject('scenario_validation', str(exc), batch_id=job['id'])
            valid = []
        # Excluded benchmark or historical content must never enter an audit prompt.
        eligible = []
        for row in valid:
            key = fingerprint(row)
            reason = ('held_out_overlap' if ((pipe.exclusion_guard and pipe.exclusion_guard(row))
                                             or pipe.exclusion_dedup.check(row))
                      else 'exact_duplicate' if key in fingerprints else pipe.dedup.check(row))
            if reason:
                pipe.reject('scenario', reason, {'scenario': row}, job['id'])
            else:
                fingerprints.add(key)
                eligible.append(row)
        neighbors = {r['generation_id']: pipe.dedup.neighbors(r) for r in eligible}
        item = {'records': eligible, 'neighbors': neighbors,
                'messages': scenario_audit_messages(eligible, neighbors) if eligible else []}
        atomic_json(path, {'input_hash': input_hash, 'audit': item})
        prepared.append((job, item))

    def audit(entry):
        job, item = entry
        if not item['records']:
            return []
        ids = [r['generation_id'] for r in item['records']]
        return _request(client, replies, job, 'scenario_audit', item['messages'], scenario_audit_schema(ids))

    audits = bounded_map(audit, prepared, workers)
    for (job, item), result in zip(prepared, audits):
        _raise_blocking(pipe, job, result)
        if isinstance(result, ValidationError):
            pipe.reject('scenario_audit', str(result), batch_id=job['id'])
        elif item['records']:
            try:
                pipe._accept_audits(item['records'], job['plan'], job['id'], item['neighbors'],
                                    item['messages'], result)
            except ValidationError as exc:
                pipe.reject('scenario_audit', str(exc), batch_id=job['id'])


def _raise_blocking(pipe, job, result):
    if isinstance(result, (ParentError, BudgetExceeded)):
        pipe._reject_scenario_api(result, job['id'])
        raise result


def generate_waves(pipe, task_targets=None):
    from .pipeline import GenerationExhausted

    target = pipe.config['generation']['target_accepted_scenarios']
    workers = pipe.request_options['workers']
    goal = digest({'target': target, 'task_targets': task_targets, 'mode': 'wave-v1'})
    pending = pipe.store.path('pending_generation_wave.json')
    limits_path = pipe.store.path('generation_wave_limits.json')
    limits = json.loads(limits_path.read_text()) if limits_path.exists() else {}
    if goal not in limits:
        batch = pipe.config['generation']['batch_size']
        limits[goal] = max(6, 3 * ((max(0, target - len(pipe.accepted)) + batch - 1) // batch))
        atomic_json(limits_path, limits)
    failures = 0
    while pending.exists() or len(pipe.accepted) < target:
        if pending.exists():
            wave = json.loads(pending.read_text())
        else:
            used = sum(r.get('generation_goal') == goal for r in pipe.store.read('generation_batches.jsonl'))
            available = min(workers, limits[goal] - used)
            if available <= 0:
                raise GenerationExhausted('Parallel scenario generation batch limit reached')
            wave = _new_wave(pipe, goal, target, task_targets, available)
            atomic_json(pending, wave)
        # Some accepted rows may have made it to disk before an interruption.
        # Prepared prompts and completed provider replies are reused as written.
        print(f"Scenario generation: {len(wave['batches'])} planned batches, "
              f"{len(pipe.accepted)}/{target} accepted before processing.", flush=True)
        _execute_wave(pipe, wave, workers)
        version, internal, gaussian = wave['rng_state']
        pipe.rng.setstate((version, tuple(internal), gaussian))
        pipe.sampler.exposure.clear()
        pipe.sampler.exposure.update(wave['seed_exposure'])
        pipe.checkpoint()
        pending.unlink()
        failures = failures + len(wave['batches']) if len(pipe.accepted) == wave['accepted_before'] else 0
        if failures >= pipe.config['generation']['max_consecutive_failed_batches']:
            raise GenerationExhausted('No usable scenarios in successive parallel generation batches')
    return len(pipe.accepted)
