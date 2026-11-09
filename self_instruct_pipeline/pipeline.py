import json
import random
import uuid
from collections import Counter

from .config import resolve_path
from .coverage_tracker import coverage_state
from .dataset_builder import export_dataset
from .deduplicator import Deduplicator
from .parent_backend import create_parent, ParentError
from .persistence import Store, atomic_json, digest, now, read_jsonl
from .prompt_builder import (generation_messages, load_system_prompt, teacher_messages,
                             verification_messages, scenario_audit_messages)
from .sampler import BalancedSeedSampler
from .scheduler import WeightedDeficitScheduler, apportioned_counts
from .validation import (ValidationError, validate_answer, validate_audit, validate_batch,
                         validate_schema, validate_scenario, batch_schema, scenario_audit_schema, AUDIT_SCHEMA)


class Pipeline:
    def __init__(self, config, parent=None, exclusions=()):
        self.config = config
        self.seeds = read_jsonl(resolve_path(config['paths']['seeds']))
        self.schemas = json.loads(resolve_path(config['paths']['schemas']).read_text())['tasks']
        if len(self.seeds) != 72 or {r['task'] for r in self.seeds} != set(self.schemas) or len(self.schemas) != 18:
            raise ValueError('Expected 72 seeds covering the existing 18 schema tasks')
        if len({r['id'] for r in self.seeds}) != len(self.seeds):
            raise ValueError('Seed IDs must be unique')
        for seed in self.seeds:
            if not isinstance(seed.get('instruction'), str) or not seed['instruction'].strip():
                raise ValueError('Seed instruction must be nonempty')
            if not seed.get('instances'):
                raise ValueError('Each seed needs a demonstration instance')
            for instance in seed['instances']:
                validate_schema(json.loads(instance['output']), self.schemas[seed['task']])
        self.system = load_system_prompt(resolve_path(config['paths']['generation_prompt']))
        ref_path = config['paths'].get('references')
        self.references = json.loads(resolve_path(ref_path).read_text()) if ref_path else []
        if not isinstance(self.references, list) or any(not isinstance(r, dict) or not r.get('id') or not r.get('text') for r in self.references):
            raise ValueError('References must be an array with id and text')
        if len({r['id'] for r in self.references}) != len(self.references):
            raise ValueError('Duplicate reference IDs')
        self.rng = random.Random(config['sampling']['random_seed'])
        self.sampler = BalancedSeedSampler(self.seeds, self.rng)
        self.scheduler = WeightedDeficitScheduler(self.schemas, config['generation']['task_weights'])
        self.store = Store(resolve_path(config['paths']['output_dir']))
        self.parent = parent
        # Held-out records are used only for local duplicate checks. They must
        # never join demonstrations, prompts, or model-based novelty comparisons.
        self.exclusions = list(exclusions)
        self.model_checked = False
        self.accepted = self.store.unique_records('accepted_scenarios.jsonl', 'id')
        if len({r['fingerprint'] for r in self.accepted}) != len(self.accepted):
            raise ValueError('Duplicate accepted scenario fingerprints')
        for row in self.accepted:
            validate_scenario({k: row[k] for k in batch_schema(self.schemas, 1)['items']['properties']}, self.schemas, self.references)
            if row.get('validation_status') != 'scenario_checks_passed':
                raise ValueError('Unvalidated scenario in accepted store')
        self.scheduler.update(self.accepted)
        self.dedup = Deduplicator(self.seeds + self.accepted,
                                 config['generation']['duplicate_rouge_threshold'],
                                 config['generation']['duplicate_context_threshold'],
                                 config['generation']['semantic_duplicate_threshold'])
        self.exclusion_dedup = Deduplicator(self.exclusions,
                                            config['generation']['duplicate_rouge_threshold'],
                                            config['generation']['duplicate_context_threshold'])

    def _contract(self):
        # Target size can increase on resume without invalidating earlier data.
        config = json.loads(json.dumps(self.config))
        config['generation'].pop('target_accepted_scenarios')
        config['paths'].pop('output_dir')
        return digest({'config': config, 'seeds': self.seeds, 'schemas': self.schemas,
                       'system_prompt': self.system, 'references': self.references,
                       **({'exclusion_hash': digest(self.exclusions)} if self.exclusions else {})})

    def initialize_manifest(self):
        path = self.store.path('run_manifest.json')
        contract = self._contract()
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest['contract_hash'] != contract:
                raise ValueError('Configuration/source contract changed; use a new output directory')
        else:
            if self.accepted or self.store.read('validated_answers.jsonl'):
                raise ValueError('Existing records lack a run manifest')
            manifest = {'created_at': now(), 'contract_hash': contract, 'config': self.config,
                        'prompt_hash': digest(self.system), 'verification': 'automatic synthetic silver; not independently validated',
                        'student_training': 'separate Qwen3-8B SFT'}
        manifest.update(updated_at=now(), target=self.config['generation']['target_accepted_scenarios'])
        atomic_json(path, manifest)
        # Restore RNG and exposure for reproducible request planning across resume.
        state = self.store.path('scheduler_state.json')
        if state.exists():
            saved = json.loads(state.read_text())
            def tuples(item):
                return tuple(tuples(x) for x in item) if isinstance(item, list) else item
            self.rng.setstate(tuples(saved['rng_state']))
            self.sampler.exposure.update(saved['seed_exposure'])

    def checkpoint(self):
        coverage = coverage_state(self.accepted, self.scheduler)
        coverage['rejected_records'] = len(self.store.read('rejected_records.jsonl'))
        atomic_json(self.store.path('coverage_ledger.json'), coverage)
        atomic_json(self.store.path('scheduler_state.json'), {
            'rng_state': self.rng.getstate(), 'seed_exposure': dict(self.sampler.exposure), 'updated_at': now()})

    def client(self):
        if self.parent is None:
            from .usage import UsageLogger
            logger = UsageLogger(self.store.directory)
            self.parent = create_parent(self.config['parent'], logger=logger)
        if not self.model_checked:
            result = self.parent.check_model()
            self.model_checked = True
            atomic_json(self.store.path('parent_model.json'), result)
        return self.parent

    def plan(self, count=None):
        generation = self.config['generation']
        count = count or min(generation['batch_size'], max(1, generation['target_accepted_scenarios'] - len(self.accepted)))
        quotas = self.scheduler.next_quotas(count, self.rng)
        difficulty = apportioned_counts(count, generation['difficulty_weights'], self.rng)
        sampling = self.config['sampling']
        seeds = self.sampler.sample(quotas, self.scheduler, self.accepted, sampling['num_seed_examples'],
                                    sampling['machine_demonstration_fraction'])
        messages = generation_messages(self.system, quotas, difficulty, seeds, self.schemas,
                                       self.references, coverage_state(self.accepted, self.scheduler))
        return {'task_quotas': quotas, 'difficulty': difficulty, 'seed_ids': [s['id'] for s in seeds],
                'schemas': list(quotas), 'messages': messages}

    def reject(self, phase, reason, record=None, batch_id=None):
        self.store.append('rejected_records.jsonl', {'id': str(uuid.uuid4()), 'timestamp': now(),
                          'phase': phase, 'reason': reason, 'batch_id': batch_id, 'record': record})

    def generate(self):
        target = self.config['generation']['target_accepted_scenarios']
        failures = 0
        while len(self.accepted) < target:
            plan = self.plan()
            batch_id = str(uuid.uuid4())
            self.store.append('generation_batches.jsonl', {'batch_id': batch_id, 'timestamp': now(),
                              'task_quotas': plan['task_quotas'], 'difficulty': plan['difficulty'], 'seed_ids': plan['seed_ids']})
            accepted_this_batch = []
            for attempt in range(self.config['generation']['max_retries_per_batch'] + 1):
                records = None
                try:
                    client = self.client()
                    count = sum(plan['task_quotas'].values())
                    records = client.generate_json(plan['messages'], batch_schema(plan['task_quotas'], count), 'scenario_generation')
                    validate_batch(records, plan['task_quotas'], plan['difficulty'], self.schemas, self.references)
                    neighbors = {r['generation_id']: self.dedup.neighbors(r) for r in records}
                    audits = client.generate_json(scenario_audit_messages(records, neighbors),
                                                   scenario_audit_schema([r['generation_id'] for r in records]),
                                                   'scenario_audit', temperature=0.0)
                    validate_schema(audits, scenario_audit_schema([r['generation_id'] for r in records]))
                    by_id = {a['generation_id']: a for a in audits}
                    if len(by_id) != len(records):
                        raise ValidationError('Missing/duplicate scenario audits')
                    # Validate every audit reference before committing any item,
                    # so a malformed late audit cannot cause a partially retried batch.
                    for index, row in enumerate(records):
                        duplicate = by_id[row['generation_id']]['duplicate_of']
                        related_ids = {r['id'] for r in neighbors[row['generation_id']]}
                        related_ids.update(r['generation_id'] for r in records[:index])
                        if duplicate and duplicate not in related_ids:
                            raise ValidationError('Novelty audit cites an unknown comparison ID')
                    for row in records:
                        audit = by_id[row['generation_id']]
                        reason = self.dedup.check(row)
                        if self.exclusion_dedup.check(row):
                            reason = 'held_out_overlap'
                        if audit['duplicate_of']:
                            reason = 'semantic_duplicate:' + audit['duplicate_of']
                        if not audit['usable'] or reason:
                            self.reject('scenario', reason or 'scenario_audit_failed', {'scenario': row, 'audit': audit}, batch_id)
                            continue
                        from .deduplicator import fingerprint
                        record = {**row, 'id': str(uuid.uuid4()), 'batch_local_generation_id': row['generation_id'],
                                  'batch_id': batch_id, 'created_at': now(), 'seed_ids': plan['seed_ids'],
                                  'parent': {'provider': self.config['parent']['provider'], 'model': self.config['parent']['model']},
                                  'settings': self.config['parent'], 'prompt_hash': digest(self.system),
                                  'scenario_audit_prompt_hash': digest(scenario_audit_messages(records, neighbors)),
                                  'validation_status': 'scenario_checks_passed', 'scenario_audit': audit,
                                  'coverage_dimensions': audit['coverage_dimensions'], 'fingerprint': fingerprint(row)}
                        self.store.append('accepted_scenarios.jsonl', record)
                        self.accepted.append(record)
                        self.scheduler.update([record])
                        self.dedup.add(record)
                        accepted_this_batch.append(record)
                    break
                except ParentError as exc:
                    self.reject('scenario_api', str(exc), batch_id=batch_id)
                    if not exc.retryable:
                        self.checkpoint()
                        raise
                except (ValidationError, KeyError, TypeError) as exc:
                    self.reject('scenario_validation', str(exc), records, batch_id)
            self.checkpoint()
            print(f'Accepted scenarios: {len(self.accepted)}/{target}; batch accepted: {len(accepted_this_batch)}', flush=True)
            failures = 0 if accepted_this_batch else failures + 1
            if failures >= self.config['generation']['max_consecutive_failed_batches']:
                raise ValueError('Stopped after consecutive failed batches; inspect rejected_records.jsonl')
        return len(self.accepted)

    def answer(self):
        existing = self.store.unique_records('validated_answers.jsonl', 'scenario_id')
        by_id = {r['id']: r for r in self.accepted}
        for answer in existing:
            if answer['scenario_id'] not in by_id or answer['scenario_hash'] != digest(by_id[answer['scenario_id']]):
                raise ValueError('Existing answer does not match its unchanged scenario')
        done = {r['scenario_id'] for r in existing}
        failures = 0
        for row in self.accepted:
            if row['id'] in done:
                continue
            success = False
            for attempt in range(self.config['generation']['max_answer_attempts_per_run']):
                answer, audit = None, None
                try:
                    client = self.client()
                    answer = client.generate_json(teacher_messages(row, self.schemas), self.schemas[row['task']], 'teacher_answer', temperature=0.0)
                    validate_answer(row, answer, self.schemas[row['task']])
                    audit = client.generate_json(verification_messages(row, answer, self.schemas), AUDIT_SCHEMA, 'answer_audit', temperature=0.0)
                    validate_audit(audit)
                    record = {'scenario_id': row['id'], 'scenario_hash': digest(row), 'task': row['task'],
                              'answer': answer, 'audit': audit, 'created_at': now(), 'parent': row['parent'],
                              'teacher_prompt_hash': digest(teacher_messages(row, self.schemas)),
                              'audit_prompt_hash': digest(verification_messages(row, answer, self.schemas)),
                              'settings': {**self.config['parent'], 'temperature': 0.0},
                              'verification_status': 'automatically_checked_silver'}
                    self.store.append('validated_answers.jsonl', record)
                    done.add(row['id'])
                    success = True
                    break
                except ParentError as exc:
                    self.reject('answer_api', str(exc), {'scenario_id': row['id']})
                    if not exc.retryable:
                        raise
                except (ValidationError, KeyError, TypeError) as exc:
                    self.reject('answer_validation', str(exc), {'scenario_id': row['id'], 'answer': answer, 'audit': audit})
            print(f'Validated answers: {len(done)}/{len(self.accepted)}', flush=True)
            failures = 0 if success else failures + 1
            if failures >= self.config['generation']['max_consecutive_failed_batches']:
                raise ValueError('Stopped after consecutive failed answers; inspect rejected_records.jsonl')
        if len(done) != len(self.accepted):
            raise ValueError('Some answers failed automatic checks; resume retries unanswered scenarios')
        return len(done)

    def export(self):
        count = export_dataset(self.store, self.schemas)
        print(f'Exported {count} automatically checked synthetic Qwen3-8B SFT pairs.', flush=True)
        return count
