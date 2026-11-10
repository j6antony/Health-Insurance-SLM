import copy
import json
from collections import Counter
from pathlib import Path

from self_instruct_pipeline.config import resolve_path
from self_instruct_pipeline.deduplicator import fingerprint
from self_instruct_pipeline.persistence import atomic_json, atomic_jsonl, digest, read_jsonl
from self_instruct_pipeline.pipeline import Pipeline
from self_instruct_pipeline.validation import validate_answer, validate_audit


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
                    selected.append({'record':record,'reference':answer})
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
        self.pipeline=Pipeline(config,parent=parent,exclusions=[x['record'] for x in development])
        self.base_weights=copy.deepcopy(self.pipeline.scheduler.weights)
        self.owns_lock=False

    def open(self):
        self.lock=self.pipeline.store.lock()
        self.lock.__enter__();self.owns_lock=True
        self.pipeline=Pipeline(self.pipeline.config,parent=self.pipeline.parent,exclusions=self.pipeline.exclusions)
        self.pipeline.initialize_manifest()

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
        self.pipeline.scheduler.weights = weights
        self.pipeline.build_pairs(target)
        return checked_pairs(self.pipeline)[:target]

    def answer_pairs(self):
        self.pipeline.answer(allow_partial=True);self.pipeline.export();self.pipeline.checkpoint()
        return checked_pairs(self.pipeline)
