import copy
import json
import math
from collections import Counter
from pathlib import Path

from self_instruct_pipeline.config import resolve_path
from self_instruct_pipeline.deduplicator import fingerprint
from self_instruct_pipeline.persistence import Store, atomic_json, atomic_jsonl, digest, read_jsonl
from self_instruct_pipeline.pipeline import Pipeline
from self_instruct_pipeline.validation import validate_answer, validate_audit
from .leakage import exclusion_record


def checked_pairs(pipeline):
    records = {r['id']:r for r in pipeline.accepted}
    pairs = []
    for answer in pipeline.store.unique_records('validated_answers.jsonl','scenario_id'):
        record = records[answer['scenario_id']]
        if answer['scenario_hash'] != digest(record) or answer['verification_status'] != 'automatically_checked_silver':
            raise ValueError('Mismatched or unvalidated parent target')
        validate_answer(record,answer['answer'],pipeline.schemas[record['task']])
        validate_audit(answer['audit'])
        pairs.append((record,answer['answer']))
    return pairs


def frozen_development(base_config, loop_config, output_dir, parent=None):
    path = output_dir/'development.jsonl'
    manifest = output_dir/'development_manifest.json'
    schemas = json.loads(resolve_path(base_config['paths']['schemas']).read_text())['tasks']
    per_task = loop_config['development_examples_per_task']
    if path.exists():
        if not manifest.exists():
            raise ValueError('Development data lacks its frozen manifest')
        data=read_jsonl(path)
        if digest(data) != json.loads(manifest.read_text())['hash']:
            raise ValueError('Frozen development set has changed')
        if Counter(x['record']['task'] for x in data) != Counter({t:per_task for t in schemas}):
            raise ValueError('Frozen development coverage disagrees with configuration')
        return data
    config=copy.deepcopy(base_config)
    config['generation']['task_weights']={}
    config['generation']['target_accepted_scenarios']=per_task*len(schemas)
    config['paths']['output_dir']=str(output_dir/'development_generation')
    pipeline=Pipeline(config,parent=parent)
    with pipeline.store.lock():
        pipeline.initialize_manifest()
        progress_path = output_dir/'development_replacements.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'passes': 0}
        # Repair only existing candidates for uncovered tasks, before spending
        # credits generating replacements. Per-sample limits survive reruns.
        pairs = checked_pairs(pipeline)
        counts = Counter(r['task'] for r, a in pairs)
        repair_ids = {r['id'] for r in pipeline.accepted if counts[r['task']] < per_task}
        if repair_ids:
            pipeline.answer(allow_partial=True, scenario_ids=repair_ids)
        # One initial pass plus a bounded number of targeted replacement passes.
        start = progress['passes'] - int(bool(progress.get('active')))
        for attempt in range(start, loop_config['development_fill_attempts'] + 1):
            pairs = checked_pairs(pipeline)
            counts = Counter(r['task'] for r, a in pairs)
            if all(counts[t] >= per_task for t in schemas):
                break
            if progress.get('active'):
                before = set(progress['active']['before'])
                targets = progress['active']['targets']
            else:
                before = {r['id'] for r in pipeline.accepted}
                accepted_counts = Counter(r['task'] for r in pipeline.accepted)
                if attempt == 0:
                    targets = {t: max(per_task, accepted_counts[t]) for t in schemas}
                else:
                    targets = {t: accepted_counts[t] + max(0, per_task - counts[t]) for t in schemas}
                progress['passes'] = attempt + 1
                progress['active'] = {'before': sorted(before), 'targets': targets}
                atomic_json(progress_path, progress)
            pipeline.config['generation']['target_accepted_scenarios'] = sum(targets.values())
            pipeline.generate(task_targets=targets)
            ids = None if attempt == 0 else {r['id'] for r in pipeline.accepted} - before
            pipeline.answer(allow_partial=True, scenario_ids=ids)
            progress.pop('active', None)
            atomic_json(progress_path, progress)
        pairs = checked_pairs(pipeline)
        counts = Counter(r['task'] for r, a in pairs)
        if all(counts[t]>=per_task for t in schemas):
            selected=[];used=Counter()
            for record,answer in pairs:
                if used[record['task']]<per_task:
                    selected.append({'record':record,'reference':answer,
                                     'verification': {'status': 'automatically_checked_silver',
                                                      'method': 'schema_evidence_and_parent_audit',
                                                      'human_reviewed': False}})
                    used[record['task']]+=1
            atomic_jsonl(path,selected)
            atomic_json(manifest,{'hash':digest(selected),'examples_per_task':per_task,
                                  'reference_status':'frozen synthetic development set; not an independent test benchmark'})
            pipeline.checkpoint()
            return selected
    missing = {t: max(0, per_task - counts[t]) for t in schemas if counts[t] < per_task}
    raise ValueError(f'Unable to fill every development task within the configured attempt limit; missing={missing}. Inspect development_generation/failure_summary.json')


class CurriculumData:
    def __init__(self, base_config, development, output_dir, parent=None):
        config=copy.deepcopy(base_config)
        config['paths']['output_dir']=str(output_dir/'self_instruct')
        self.pipeline=Pipeline(config,parent=parent,exclusions=[exclusion_record(x) for x in development])
        self.base_weights=copy.deepcopy(self.pipeline.scheduler.weights)
        self.owns_lock=False
        self.deterministic_fraction = 0.0
        self.export_reference_pairs = True
        self.expanded_templates = False
        self.exclusion_registry = None

    def open(self):
        self.lock=self.pipeline.store.lock()
        self.lock.__enter__();self.owns_lock=True
        self.pipeline=Pipeline(self.pipeline.config,parent=self.pipeline.parent,exclusions=self.pipeline.exclusions)
        self.pipeline.initialize_manifest()
        if self.exclusion_registry is not None:
            self.exclusion_registry.require_clean(self.pipeline.accepted, 'saved candidate pool')
            self.pipeline.exclusion_guard = self.exclusion_registry.match

    def close(self):
        if self.owns_lock:
            self.lock.__exit__(None,None,None);self.owns_lock=False

    def generate(self, target, weights):
        p=self.pipeline
        p.config['generation']['target_accepted_scenarios']=target
        p.scheduler.weights=weights
        p.generate();p.checkpoint()
        return p.accepted

    def verified_pairs(self, target, weights):
        from .trusted_templates import template_pairs
        count = math.ceil(target * self.deterministic_fraction)
        templates = template_pairs(count, self.pipeline.schemas, self.pipeline.exclusions,
                                   expanded=self.expanded_templates,
                                   exclusion_guard=self.pipeline.exclusion_guard) if count else []
        self.pipeline.scheduler.weights = weights
        silver_target = target - len(templates)
        if silver_target:
            self.pipeline.build_pairs(silver_target, export_pairs=self.export_reference_pairs)
        # Template IDs are stable; cumulative replay does not create extra unique data.
        return (templates + checked_pairs(self.pipeline)[:silver_target])[:target]

    def screening_topup(self, deficits, eligible):
        """Reuse verified candidates and request only tasks with a remaining gap."""
        from self_instruct_pipeline.pipeline import GenerationExhausted

        pipeline = self.pipeline
        verified = checked_pairs(pipeline)
        verified_ids = {record['id'] for record, _ in verified}
        unanswered = {record['id'] for record in pipeline.accepted
                      if record['task'] in deficits and record['id'] not in verified_ids}
        if unanswered:
            pipeline.answer(allow_partial=True, scenario_ids=unanswered)
            verified = checked_pairs(pipeline)
        cached = eligible(verified)
        available = Counter(record['task'] for record, _ in cached)
        needed = {task: count - available[task] for task, count in deficits.items()
                  if count > available[task]}
        if needed:
            counts = Counter(record['task'] for record in pipeline.accepted)
            # Include spare candidates for answer audit and context-length failures.
            targets = {task: counts[task] + 2 * count for task, count in needed.items()}
            previous_target = pipeline.config['generation']['target_accepted_scenarios']
            existing = {record['id'] for record in pipeline.accepted}
            pipeline.config['generation']['target_accepted_scenarios'] = (
                len(existing) + sum(targets[task] - counts[task] for task in targets))
            try:
                try:
                    pipeline.generate(task_targets=targets)
                except GenerationExhausted:
                    pass
                cached_ids = {record['id'] for record, _ in cached}
                candidate_ids = {record['id'] for record in pipeline.accepted
                                 if record['task'] in needed and record['id'] not in cached_ids
                                 and record['id'] not in existing}
                if candidate_ids:
                    pipeline.answer(allow_partial=True, scenario_ids=candidate_ids)
            finally:
                pipeline.config['generation']['target_accepted_scenarios'] = previous_target
                pipeline.checkpoint()
            cached = eligible(checked_pairs(pipeline))
        return [pair for pair in cached if pair[0]['task'] in deficits]

    def answer_pairs(self):
        self.pipeline.answer(allow_partial=True);self.pipeline.export();self.pipeline.checkpoint()
        return checked_pairs(self.pipeline)


def repair_development_task(base_config, loop_config, output_dir, task='field_extraction'):
    """Explicit recovery: permit attempt 4 for one latest failed candidate, not a reset."""
    output_dir = Path(output_dir)
    if (output_dir/'development.jsonl').exists():
        return {'status': 'development_already_frozen'}
    config = copy.deepcopy(base_config)
    config['generation']['task_weights'] = {}
    config['paths']['output_dir'] = str(output_dir/'development_generation')
    pipeline = Pipeline(config)
    if task not in pipeline.schemas:
        raise ValueError('Unknown development task')
    with pipeline.store.lock():
        pipeline.initialize_manifest()
        pairs = checked_pairs(pipeline)
        if sum(r['task'] == task for r, _ in pairs) >= loop_config['development_examples_per_task']:
            return {'status': 'task_already_covered', 'task': task}
        done = {r['id'] for r, _ in pairs}
        candidates = [r for r in pipeline.accepted if r['task'] == task and r['id'] not in done]
        if not candidates:
            raise ValueError('No existing candidate to repair; no API calls made')
        candidate = candidates[-1]
        pipeline.answer(allow_partial=True, scenario_ids={candidate['id']}, max_sample_attempts=4)
        return {'task': task, 'scenario_id': candidate['id'],
                'validated': any(r['id'] == candidate['id'] for r, _ in checked_pairs(pipeline))}


def scenario_canary(parent_config, config, *, recovery_id=None):
    """Run one generation batch or re-audit one legacy batch in the existing run."""
    from .benchmarks import load_benchmarks
    from .leakage import ExclusionRegistry, freeze_loss_split

    root = resolve_path(config['output_dir'])
    store = Store(root)
    schemas = json.loads(resolve_path(parent_config['paths']['schemas']).read_text())['tasks']
    with store.lock():
        development, holdout = load_benchmarks(config, schemas)
        split = resolve_path(config['evaluation']['split_path'])
        if json.loads(store.path('evaluation_split.json').read_text()) != json.loads(split.read_text()):
            raise ValueError('Saved evaluation split differs from the original')
        frozen_loss = store.path('heldout_teacher_loss_split.json')
        if not frozen_loss.is_file():
            raise ValueError('Expected the existing run and its frozen teacher-loss split')
        candidates = config.get('teacher_loss_candidates')
        loss = freeze_loss_split(
            frozen_loss, resolve_path(config['historical_training_plan']), development + holdout,
            schemas, candidates_path=resolve_path(candidates) if candidates else None,
            exposure_roots=[resolve_path(path) for path in config.get('historical_exposure_roots', ())])
        registry = ExclusionRegistry(development + holdout + loss['rows'])
        corpus = read_jsonl(resolve_path(config['evaluation']['validation_path']))
        data = CurriculumData(parent_config, corpus + loss['rows'], root)
        data.exclusion_registry = registry
        try:
            data.open()
            pipeline = data.pipeline
            before = len(pipeline.accepted)
            if recovery_id:
                pipeline.recover_legacy_batch(recovery_id)
            else:
                pipeline.config['generation']['target_accepted_scenarios'] = before + 1
                pipeline.generate(one_batch=True)
            new_candidates = pipeline.accepted[before:]
            verified = 0
            if new_candidates:
                sample_id = new_candidates[0]['id']
                pipeline.answer(allow_partial=True, scenario_ids={sample_id}, max_sample_attempts=1)
                verified = sum(record['id'] == sample_id for record, _ in checked_pairs(pipeline))
            return {'accepted_before': before, 'accepted_after': len(pipeline.accepted),
                    'accepted_new': len(new_candidates), 'verified_canary_pairs': verified}
        finally:
            data.close()


def fresh_teacher_pool(pipeline, target, weights, old_ids, *, eligible=lambda pair: True,
                       required_unused=None, used_ids=()):
    """Build up to a fresh, usable pool while retaining interrupted work."""
    from self_instruct_pipeline.pipeline import GenerationExhausted

    pipeline.scheduler.weights = weights
    required_unused = required_unused or {}
    if pipeline.store.path('pending_generation_wave.json').exists():
        pipeline.config['generation']['target_accepted_scenarios'] = len(pipeline.accepted)
        pipeline.generate()
    for attempt in range(4):
        ids = {record['id'] for record in pipeline.accepted if record['id'] not in old_ids}
        pipeline.answer(allow_partial=True, scenario_ids=ids)
        pairs = [pair for pair in checked_pairs(pipeline) if pair[0]['id'] in ids and eligible(pair)]
        available = Counter(r['task'] for r, _ in pairs if r['id'] not in used_ids)
        missing = {task: count - available[task] for task, count in required_unused.items()
                   if count > available[task]}
        targeted = sum(missing.values())
        shortage = max(0, target - len(pairs))
        if (not shortage and not targeted) or attempt == 3:
            return pairs
        task_targets = None
        if targeted:
            # First fill the measured task deficits. Unrestricted slots are
            # filled only after checking how many of those cases are usable.
            task_targets = Counter(r['task'] for r in pipeline.accepted)
            task_targets.update(missing)
        pipeline.config['generation']['target_accepted_scenarios'] = len(pipeline.accepted) + (targeted or shortage)
        try:
            pipeline.generate(task_targets=task_targets)
        except GenerationExhausted:
            if not {record['id'] for record in pipeline.accepted} - ids - set(old_ids):
                return pairs
