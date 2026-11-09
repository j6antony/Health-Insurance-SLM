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
        rejections = self.store.read('rejected_records.jsonl')
        answers = self.store.read('validated_answers.jsonl')
        attempted = {r['scenario_id'] for r in self.store.read('answer_attempts.jsonl')}
        attempted.update(r.get('record', {}).get('scenario_id') for r in rejections
                         if r.get('phase', '').startswith('answer') and isinstance(r.get('record'), dict))
        attempted.update(r['scenario_id'] for r in answers)
        attempted.discard(None)
        repaired = sum(r.get('answer_attempt', 1) > 1 for r in answers)
        attempted_repairs = {r['scenario_id'] for r in self.store.read('answer_attempts.jsonl')
                             if r.get('attempt', 1) > 1}
        atomic_json(self.store.path('failure_summary.json'), {
            'accepted_scenarios': len(self.accepted), 'validated_answers': len(answers),
            'answer_acceptance_rate': len(answers) / len(attempted) if attempted else None,
            'repaired_answers': repaired,
            'samples_with_repair_attempts': len(attempted_repairs),
            'repair_success_rate': repaired / len(attempted_repairs) if attempted_repairs else None,
            'failure_types': dict(Counter(r.get('failure_type', r['phase']) for r in rejections)),
            'failure_reasons': dict(Counter(r['reason'] for r in rejections))})
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

    def plan(self, count=None, task_quotas=None):
        generation = self.config['generation']
        count = count or min(generation['batch_size'], max(1, generation['target_accepted_scenarios'] - len(self.accepted)))
        quotas = task_quotas if task_quotas is not None else self.scheduler.next_quotas(count, self.rng)
        count = sum(quotas.values())
        difficulty = apportioned_counts(count, generation['difficulty_weights'], self.rng)
        sampling = self.config['sampling']
        seeds = self.sampler.sample(quotas, self.scheduler, self.accepted, sampling['num_seed_examples'],
                                    sampling['machine_demonstration_fraction'])
        messages = generation_messages(self.system, quotas, difficulty, seeds, self.schemas,
                                       self.references, coverage_state(self.accepted, self.scheduler))
        return {'task_quotas': quotas, 'difficulty': difficulty, 'seed_ids': [s['id'] for s in seeds],
                'schemas': list(quotas), 'messages': messages}

    def reject(self, phase, reason, record=None, batch_id=None):
        if phase.startswith('scenario'):
            failure_type, repairable = ('duplicate' if 'duplicate' in reason.lower() else 'scenario_error'), False
        elif phase.endswith('_api'):
            failure_type, repairable = ('format_error' if 'JSON' in reason else 'api_error'), 'JSON' in reason
        elif isinstance(record, dict) and record.get('audit') is not None:
            failure_type, repairable = 'grounding_error', True
        else:
            failure_type = 'evidence_error' if any(word in reason.lower() for word in ('evidence', 'quote', 'unsupported')) else 'format_error'
            repairable = True
        self.store.append('rejected_records.jsonl', {'failure_type': failure_type, 'repairable': repairable, 'id': str(uuid.uuid4()), 'timestamp': now(),
                          'phase': phase, 'reason': reason, 'batch_id': batch_id, 'record': record})

    def generate(self, task_targets=None):
        target = self.config['generation']['target_accepted_scenarios']
        goal = digest({'target': target, 'task_targets': task_targets})
        limit_path = self.store.path('generation_limits.json')
        limits = json.loads(limit_path.read_text()) if limit_path.exists() else {}
        if goal not in limits:
            needed = max(0, target - len(self.accepted))
            batch = self.config['generation']['batch_size']
            limits[goal] = {'max_batches': max(6, 3 * ((needed + batch - 1) // batch))}
            atomic_json(limit_path, limits)
        batches = sum(r.get('generation_goal') == goal for r in self.store.read('generation_batches.jsonl'))
        failures = 0
        while len(self.accepted) < target:
            if batches >= limits[goal]['max_batches']:
                raise ValueError('Scenario generation attempt limit reached; inspect failure_summary.json and rejected_records.jsonl')
            batches += 1
            quotas = None
            if task_targets is not None:
                counts = Counter(r['task'] for r in self.accepted)
                quotas, remaining = {}, self.config['generation']['batch_size']
                for task, wanted in task_targets.items():
                    take = min(remaining, max(0, wanted - counts[task]))
                    if take:
                        quotas[task] = take
                        remaining -= take
                if not quotas:
                    raise ValueError('Development task targets disagree with generation target')
            plan = self.plan(task_quotas=quotas)
            batch_id = str(uuid.uuid4())
            self.store.append('generation_batches.jsonl', {'batch_id': batch_id, 'generation_goal': goal, 'timestamp': now(),
                              'task_quotas': plan['task_quotas'], 'difficulty': plan['difficulty'], 'seed_ids': plan['seed_ids']})
            accepted_this_batch = []
            for attempt in range(self.config['generation']['max_retries_per_batch'] + 1):
                records = None
                try:
                    print(f'Generating scenarios: accepted={len(self.accepted)}/{target}, batch={batches}/{limits[goal]["max_batches"]}, tasks={plan["task_quotas"]}', flush=True)
                    client = self.client()
                    count = sum(plan['task_quotas'].values())
                    records = client.generate_json(plan['messages'], batch_schema(plan['task_quotas'], count), 'scenario_generation')
                    # Batch shape/quotas must be coherent, but invalid individual
                    # scenarios are quarantined without losing their valid peers.
                    if not isinstance(records, list) or not records or len(records) > 2 * count:
                        raise ValidationError('Generation batch envelope invalid')
                    valid, ids = [], set()
                    task_counts, difficulty_counts = Counter(), Counter()
                    for row in records:
                        try:
                            validate_scenario(row, self.schemas, self.references)
                            if row['generation_id'] in ids:
                                raise ValidationError('Duplicate generation ID within batch')
                            if task_counts[row['task']] >= plan['task_quotas'].get(row['task'], 0):
                                raise ValidationError('Scenario exceeds requested task quota')
                            if difficulty_counts[row['difficulty']] >= plan['difficulty'].get(row['difficulty'], 0):
                                raise ValidationError('Scenario exceeds requested difficulty quota')
                            ids.add(row['generation_id'])
                            task_counts[row['task']] += 1
                            difficulty_counts[row['difficulty']] += 1
                            valid.append(row)
                        except (ValidationError, KeyError, TypeError) as exc:
                            self.reject('scenario_validation', str(exc), row, batch_id)
                    records = valid
                    if not records:
                        break
                    neighbors = {r['generation_id']: self.dedup.neighbors(r) for r in records}
                    audits = client.generate_json(scenario_audit_messages(records, neighbors),
                                                   scenario_audit_schema([r['generation_id'] for r in records]),
                                                   'scenario_audit', temperature=0.0)
                    if not isinstance(audits, list):
                        raise ValidationError('Scenario audit must return an array')
                    audit_ids = Counter(a.get('generation_id') for a in audits if isinstance(a, dict))
                    by_id = {a.get('generation_id'): a for a in audits if isinstance(a, dict)}
                    audit_schema = scenario_audit_schema([r['generation_id'] for r in records])['items']
                    for index, row in enumerate(records):
                        audit = by_id.get(row['generation_id'])
                        try:
                            if audit_ids[row['generation_id']] != 1:
                                raise ValidationError('Missing/duplicate scenario audit')
                            validate_schema(audit, audit_schema)
                            related_ids = {r['id'] for r in neighbors[row['generation_id']]}
                            related_ids.update(r['generation_id'] for r in records[:index])
                            if audit['duplicate_of'] and audit['duplicate_of'] not in related_ids:
                                raise ValidationError('Novelty audit cites an unknown comparison ID')
                        except (ValidationError, KeyError, TypeError) as exc:
                            self.reject('scenario_audit', str(exc), {'scenario': row, 'audit': audit}, batch_id)
                            continue
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

    def answer(self, allow_partial=False, scenario_ids=None, max_sample_attempts=3):
        if type(max_sample_attempts) is not int or max_sample_attempts < 1:
            raise ValueError('max_sample_attempts must be a positive integer')
        existing = self.store.unique_records('validated_answers.jsonl', 'scenario_id')
        by_id = {r['id']: r for r in self.accepted}
        for answer in existing:
            if answer['scenario_id'] not in by_id or answer['scenario_hash'] != digest(by_id[answer['scenario_id']]):
                raise ValueError('Existing answer does not match its unchanged scenario')
        done = {r['scenario_id'] for r in existing}
        feedback_by_id = {}
        previous_failures = Counter()
        attempt_counts = Counter(r['scenario_id'] for r in self.store.read('answer_attempts.jsonl'))
        for rejected in self.store.read('rejected_records.jsonl'):
            proposal = rejected.get('record', {})
            if rejected.get('phase') in ('answer_validation', 'answer_api') and isinstance(proposal, dict):
                previous_failures[proposal.get('scenario_id')] += 1
            if (rejected.get('phase') == 'answer_validation' or (rejected.get('phase') == 'answer_api' and 'JSON' in rejected['reason'])) and isinstance(proposal, dict):
                feedback_by_id[proposal.get('scenario_id')] = {
                    'validation_error': rejected['reason'], 'proposed_answer': proposal.get('answer'),
                    'audit': proposal.get('audit')}
        failures = 0
        for row in self.accepted:
            if row['id'] in done or (scenario_ids is not None and row['id'] not in scenario_ids):
                continue
            success = False
            used = max(attempt_counts[row['id']], previous_failures[row['id']])
            attempts = max(0, max_sample_attempts - used) if allow_partial else self.config['generation']['max_answer_attempts_per_run']
            for attempt in range(attempts):
                answer, audit = None, None
                try:
                    client = self.client()
                    number = used + attempt + 1
                    self.store.append('answer_attempts.jsonl', {'scenario_id': row['id'],
                        'attempt': number, 'timestamp': now()})
                    # Seed historical failures into the persistent count on first resume.
                    if attempt_counts[row['id']] < used:
                        for _ in range(used - attempt_counts[row['id']]):
                            self.store.append('answer_attempts.jsonl', {'scenario_id': row['id'], 'historical': True})
                        attempt_counts[row['id']] = used
                    print(f'Answering {row["task"]}: attempt {number}/{max_sample_attempts if allow_partial else number + attempts - attempt - 1}', flush=True)
                    messages = teacher_messages(row, self.schemas, feedback_by_id.get(row['id']))
                    answer = client.generate_json(messages, self.schemas[row['task']], 'teacher_answer', temperature=0.0)
                    validate_answer(row, answer, self.schemas[row['task']])
                    audit = client.generate_json(verification_messages(row, answer, self.schemas), AUDIT_SCHEMA, 'answer_audit', temperature=0.0)
                    validate_audit(audit)
                    record = {'scenario_id': row['id'], 'scenario_hash': digest(row), 'task': row['task'],
                              'answer': answer, 'audit': audit, 'answer_attempt': number, 'created_at': now(), 'parent': row['parent'],
                              'teacher_prompt_hash': digest(messages),
                              'audit_prompt_hash': digest(verification_messages(row, answer, self.schemas)),
                              'settings': {**self.config['parent'], 'temperature': 0.0},
                              'verification_status': 'automatically_checked_silver'}
                    self.store.append('validated_answers.jsonl', record)
                    done.add(row['id'])
                    success = True
                    break
                except ParentError as exc:
                    self.reject('answer_api', str(exc), {'scenario_id': row['id']})
                    if 'JSON' in str(exc):
                        feedback_by_id[row['id']] = {'validation_error': str(exc), 'proposed_answer': None, 'audit': None}
                    if not exc.retryable or (allow_partial and 'JSON' not in str(exc)):
                        raise
                except (ValidationError, KeyError, TypeError) as exc:
                    self.reject('answer_validation', str(exc), {'scenario_id': row['id'], 'answer': answer, 'audit': audit})
                    feedback_by_id[row['id']] = {'validation_error': str(exc),
                                                 'proposed_answer': answer, 'audit': audit}
            print(f'Validated answers: {len(done)}/{len(self.accepted)}', flush=True)
            failures = 0 if success else failures + 1
            if not allow_partial and failures >= self.config['generation']['max_consecutive_failed_batches']:
                raise ValueError('Stopped after consecutive failed answers; inspect rejected_records.jsonl')
        self.checkpoint()
        if not allow_partial and len(done) != len(self.accepted):
            raise ValueError('Some answers failed automatic checks; resume retries unanswered scenarios')
        return len(done)

    def export(self):
        count = export_dataset(self.store, self.schemas)
        print(f'Exported {count} automatically checked synthetic Qwen3-8B SFT pairs.', flush=True)
        return count
