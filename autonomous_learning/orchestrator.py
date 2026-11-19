"""Bounded SQLite workflow. Routine operations survive interactive Codex exit."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import time

from .storage import Storage, StorageError, digest, checksum, read_json
from .planner import heuristic, validate_plan, planner_input
from .scenarios import generate, expand, OPERATIONS
from .rendering import render, verify
from .tasks import project
from .dataset import build, export
from .rules import rule_pack
from .jobs import prepare

STAGES = ('PLAN', 'GENERATE_STATES', 'RENDER', 'VERIFY_RENDERING', 'EXPAND', 'PROJECT_TASKS',
          'VALIDATE_DATASET', 'EXPORT', 'PREPARE_TRAINING', 'TRAIN', 'EVALUATE',
          'PROMOTE_OR_REJECT', 'ANALYZE_FAILURES', 'NEXT_CYCLE')
DEFAULTS = {'max_cycles': 1, 'max_wall_seconds': 3600, 'max_non_improving': 2,
            'master_scenarios': 24, 'seed': 42, 'teacher_enabled': False,
            'max_api_calls': 20, 'max_api_usd': 0, 'max_gpu_usd': 0,
            'gpu_backend': 'manual', 'backup_enabled': False}


class WaitingForArtifact(RuntimeError):
    pass


class State:
    def __init__(self, storage):
        self.storage = storage
        with self.connect() as c:
            c.executescript('''
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, config TEXT NOT NULL, status TEXT NOT NULL,
                control TEXT, started REAL NOT NULL, dry_run INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS stages (
                run_id TEXT NOT NULL, cycle INTEGER NOT NULL, name TEXT NOT NULL,
                input_hash TEXT NOT NULL, output TEXT, output_hash TEXT,
                status TEXT NOT NULL, attempts INTEGER NOT NULL, error TEXT,
                seconds REAL NOT NULL DEFAULT 0, api_cost_usd REAL NOT NULL DEFAULT 0,
                gpu_cost_usd REAL NOT NULL DEFAULT 0, PRIMARY KEY(run_id,cycle,name));
            CREATE TABLE IF NOT EXISTS holdout_access (
                benchmark TEXT NOT NULL, adapter TEXT NOT NULL, accessed REAL NOT NULL,
                PRIMARY KEY(benchmark,adapter));
            ''')

    @contextmanager
    def connect(self):
        self.storage.preflight()
        c = sqlite3.connect(self.storage.path('state/orchestration.sqlite'), timeout=20)
        try:
            with c:
                c.row_factory = sqlite3.Row
                c.execute('PRAGMA synchronous=FULL')
                yield c
        finally:
            c.close()

    def status(self, run_id):
        with self.connect() as c:
            run = c.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
            rows = c.execute('SELECT * FROM stages WHERE run_id=? ORDER BY cycle,rowid', (run_id,)).fetchall()
        return {'run': dict(run) if run else None, 'stages': [dict(row) for row in rows]}

    def control(self, run_id, action):
        if action not in ('pause', 'cancel', 'resume'):
            raise ValueError('Unknown control action')
        with self.connect() as c:
            row = c.execute('SELECT status FROM runs WHERE id=?', (run_id,)).fetchone()
            if not row:
                raise ValueError('Unknown run')
            if row['status'] == 'cancelled':
                raise ValueError('Cancelled runs cannot be resumed')
            c.execute('UPDATE runs SET control=? WHERE id=?', (None if action == 'resume' else action, run_id))

    def reserve_holdout(self, benchmark_id, adapter_sha256, authorized=False):
        if not authorized:
            raise ValueError('Sealed holdout requires explicit release evaluation authorization')
        with self.connect() as c:
            rows = c.execute('SELECT adapter FROM holdout_access WHERE benchmark=?', (benchmark_id,)).fetchall()
            if adapter_sha256 not in {r['adapter'] for r in rows} and len(rows) >= 2:
                raise ValueError('Holdout access limit reached: at most one two-checkpoint comparison')
            c.execute('INSERT OR IGNORE INTO holdout_access VALUES(?,?,?)', (benchmark_id, adapter_sha256, time.time()))


class Controller:
    def __init__(self, storage, config=None):
        self.storage = storage
        self.config = {**DEFAULTS, **(config or {})}
        self.config['gpu_backend'] = os.environ.get('SLM_GPU_BACKEND', self.config['gpu_backend'])
        if 'SLM_BACKUP_ENABLED' in os.environ:
            if os.environ['SLM_BACKUP_ENABLED'].lower() not in ('true', 'false'):
                raise ValueError('SLM_BACKUP_ENABLED must be true or false')
            self.config['backup_enabled'] = os.environ['SLM_BACKUP_ENABLED'].lower() == 'true'
        if set(self.config) != set(DEFAULTS):
            raise ValueError('Unknown orchestrator configuration')
        for key, high in (('max_cycles', 10), ('max_wall_seconds', 604800), ('max_non_improving', 10),
                          ('master_scenarios', 1000), ('max_api_calls', 10000)):
            if type(self.config[key]) is not int or not 1 <= self.config[key] <= high:
                raise ValueError('Invalid run limit: ' + key)
        if self.config['gpu_backend'] != 'manual' or self.config['backup_enabled']:
            raise ValueError('This controller supports manual GPU transfers and local storage; legacy Drive backup remains available separately')
        for key in ('max_api_usd', 'max_gpu_usd'):
            import math
            if type(self.config[key]) not in (int, float) or not math.isfinite(self.config[key]) or self.config[key] < 0:
                raise ValueError('Invalid spending limit')
        self.storage.initialize()
        self.state = State(storage)

    def _stage(self, run_id, cycle, name, inputs, function):
        self.storage.preflight()
        key = digest(inputs)
        with self.state.connect() as c:
            saved = c.execute('SELECT * FROM stages WHERE run_id=? AND cycle=? AND name=?', (run_id, cycle, name)).fetchone()
            if saved:
                if saved['input_hash'] != key:
                    raise ValueError('Stage input changed; create a separate run')
                if saved['status'] == 'completed':
                    receipt = json.loads(saved['output'])
                    output = self.storage.verify(receipt)
                    if digest(output) != saved['output_hash']:
                        raise StorageError('Stage output identity mismatch')
                    return output
                c.execute("UPDATE stages SET status='running', attempts=attempts+1,error=NULL WHERE run_id=? AND cycle=? AND name=?", (run_id, cycle, name))
            else:
                c.execute('INSERT INTO stages(run_id,cycle,name,input_hash,status,attempts) VALUES(?,?,?,?,?,?)',
                          (run_id, cycle, name, key, 'running', 1))
        started = time.monotonic()
        budget_file = self.storage.path('api_usage/render_budget/state.json')
        cost_before = float(read_json(budget_file)['accounted_usd']) if budget_file.exists() else 0.0
        try:
            output = function()
            receipt = self.storage.write(f'experiments/{run_id}/cycle_{cycle:03d}/{name}.json', output)
            artifact_folders = {'GENERATE_STATES': 'master_scenarios', 'RENDER': 'rendered_documents',
                                'EXPAND': 'counterfactuals', 'PROJECT_TASKS': 'task_examples'}
            if name in artifact_folders:
                self.storage.write('datasets/' + artifact_folders[name] + '/' + digest(output) + '.json', output)
            if isinstance(output, dict) and output.get('quarantine'):
                self.storage.write('datasets/quarantine/' + digest(output['quarantine']) + '.json',
                                   {'stage': name, 'records': output['quarantine']})
            self.storage.preflight()
        except BaseException as exc:
            # Error type only: arbitrary provider/document text must not enter logs.
            with self.state.connect() as c:
                c.execute('UPDATE stages SET status=?,error=?,seconds=seconds+? WHERE run_id=? AND cycle=? AND name=?',
                          ('waiting' if isinstance(exc, WaitingForArtifact) else 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                           type(exc).__name__, time.monotonic() - started, run_id, cycle, name))
            raise
        cost_after = float(read_json(budget_file)['accounted_usd']) if budget_file.exists() else 0.0
        gpu_cost = output.get('result', {}).get('estimated_gpu_cost_usd', 0) if name == 'TRAIN' else 0
        with self.state.connect() as c:
            c.execute("UPDATE stages SET status='completed', output=?,output_hash=?,seconds=seconds+?,api_cost_usd=api_cost_usd+?,gpu_cost_usd=gpu_cost_usd+? WHERE run_id=? AND cycle=? AND name=?",
                      (json.dumps(receipt), digest(output), time.monotonic() - started, max(0, cost_after-cost_before), gpu_cost, run_id, cycle, name))
        return output

    def run(self, run_id, *, dry_run=False, proposed_plan=None, authorize_api=False):
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id):
            raise ValueError('Run ID must use 1..80 letters, digits, underscores or hyphens')
        if self.config['teacher_enabled'] and (not authorize_api or self.config['max_api_usd'] <= 0):
            raise ValueError('Teacher generation requires explicit authorization and a positive budget')
        with self.storage.lock():
            contract = {'config': self.config, 'proposed_plan': proposed_plan,
                        'source_sha256': {p.name: checksum(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
            encoded = json.dumps(contract, sort_keys=True)
            with self.state.connect() as c:
                saved = c.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
                if saved and (saved['config'] != encoded or bool(saved['dry_run']) != dry_run):
                    raise ValueError('Run configuration, planner input or source code changed')
                if saved and saved['status'] == 'cancelled':
                    raise ValueError('Run was cancelled')
                if saved is None:
                    c.execute('INSERT INTO runs VALUES(?,?,?,?,?,?)', (run_id, encoded, 'running', None, time.time(), int(dry_run)))
                else:
                    c.execute("UPDATE runs SET status='running' WHERE id=?", (run_id,))
            self.storage.write('rules/snapshots/' + rule_pack()['version'] + '.json', rule_pack())
            previous = None
            feedback = None
            non_improving = 0
            coverage = []
            try:
                for cycle in range(1, self.config['max_cycles'] + 1):
                    results = {}
                    def stage(name, function):
                        info = self.state.status(run_id)['run']
                        action = info['control']
                        if action in ('pause', 'cancel'):
                            raise WaitingForArtifact('control_' + action)
                        if time.time() - info['started'] > self.config['max_wall_seconds']:
                            raise TimeoutError('Run wall-clock limit reached')
                        inputs = {'contract': digest(contract), 'previous_outputs': {k: digest(v) for k, v in results.items()},
                                  'previous_plan': previous, 'cycle': cycle}
                        results[name] = self._stage(run_id, cycle, name, inputs, function)
                        return results[name]
                    def plan_stage():
                        plan = proposed_plan if proposed_plan and cycle == 1 else heuristic(cycle, self.config['master_scenarios'], feedback, previous)
                        if not proposed_plan:
                            plan['max_gpu_usd'] = self.config['max_gpu_usd']
                        if not proposed_plan and self.config['teacher_enabled']:
                            plan['max_api_usd'] = self.config['max_api_usd']
                        return validate_plan(plan, {'max_api_usd': self.config['max_api_usd'], 'max_gpu_usd': self.config['max_gpu_usd']}, previous)
                    plan = stage('PLAN', plan_stage)
                    generated = stage('GENERATE_STATES', lambda: generate(plan['master_scenarios'], self.config['seed'] + cycle,
                                      prior_coverage=coverage, focus=self._case_focus(plan)))
                    coverage = generated['coverage_pairs']
                    def render_stage():
                        output = [render(s) for s in generated['scenarios']]
                        if self.config['teacher_enabled'] and not dry_run:
                            from .teacher import RenderQueue
                            queue = RenderQueue(self.storage)
                            settings = read_json(Path(__file__).resolve().parents[1] / 'config/self_instruct.json')['parent']
                            keys = {}
                            for n, s in enumerate(generated['scenarios']):
                                for d in s['documents']:
                                    if d['type'] in ('discharge_summary', 'investigation_report') and d['present'] and d['readability'] == 'readable':
                                        keys[n, d['id']] = queue.enqueue(d, settings)
                            queue.run(settings, paid_authorized=True, max_calls=self.config['max_api_calls'], max_cost_usd=plan['max_api_usd'])
                            for (n, doc), key in keys.items():
                                answer = queue.result(key)
                                if answer['status'] == 'done':
                                    for page in output[n]['pages']:
                                        if page['document_id'] == doc:
                                            page['text'] = answer['result']['text']
                                    output[n]['renderer'] = 'hybrid-v1'
                                    output[n]['render_hash'] = digest(output[n]['pages'])
                        return {'rendered': output}
                    rendered = stage('RENDER', render_stage)
                    def verify_stage():
                        accepted, rejected = [], []
                        for s, r in zip(generated['scenarios'], rendered['rendered'], strict=True):
                            try:
                                gate = verify(s, r)
                            except ValueError as exc:
                                rejected.append({'scenario': s, 'rendered': r, 'reason': str(exc)})
                            else:
                                accepted.append({'scenario': s, 'rendered': r, 'gate': gate})
                        return {'accepted': accepted, 'quarantine': rejected}
                    verified = stage('VERIFY_RENDERING', verify_stage)
                    def expand_stage():
                        rows, rejected = list(verified['accepted']), []
                        for row in verified['accepted']:
                            for op in OPERATIONS[:plan['variants_per_family']]:
                                s = expand(row['scenario'], op)
                                r = render(s)
                                try:
                                    gate = verify(s, r)
                                except ValueError as exc:
                                    rejected.append({'scenario': s, 'rendered': r, 'reason': str(exc)})
                                else:
                                    rows.append({'scenario': s, 'rendered': r, 'gate': gate})
                        return {'verified': rows, 'quarantine': rejected}
                    expanded = stage('EXPAND', expand_stage)
                    projected = stage('PROJECT_TASKS', lambda: {'examples': [example for row in expanded['verified']
                                                                               for example in project(row['scenario'], row['rendered'])]})
                    def validate_stage():
                        evaluation = []
                        for tier in ('development', 'probes'):
                            for path in self.storage.path('evaluation/' + tier).glob('*.json'):
                                evaluation.extend(read_json(path)['cases'])
                        replay = []
                        for path in sorted(self.storage.path('datasets/approved_sft').glob('*/examples.jsonl')):
                            replay.extend(json.loads(line) for line in path.read_text().splitlines())
                        from .dataset import reject_sealed_leakage
                        for path in self.storage.path('evaluation/release_holdout/exclusions').glob('*.json'):
                            reject_sealed_leakage(projected['examples'] + replay, read_json(path))
                        return build(projected['examples'], plan['max_training_examples_per_family'], evaluation,
                                     replay, plan['replay_fraction'], plan['adapter_strategy'])
                    dataset = stage('VALIDATE_DATASET', validate_stage)
                    exported = stage('EXPORT', lambda: export(self.storage, dataset))
                    def prepare_stage():
                        best = self.storage.path('checkpoints/best/registry.json')
                        adapter = read_json(best)['directory'] if best.exists() and plan['adapter_strategy'] == 'continued' else None
                        with self.state.connect() as db:
                            spent = db.execute('SELECT COALESCE(SUM(gpu_cost_usd),0) FROM stages WHERE run_id=?', (run_id,)).fetchone()[0]
                        remaining = max(0, self.config['max_gpu_usd'] - spent)
                        training_plan = {**plan, 'max_gpu_usd': min(plan['max_gpu_usd'], remaining)}
                        return prepare(self.storage, exported, training_plan, adapter)
                    package = stage('PREPARE_TRAINING', prepare_stage)
                    def train_stage():
                        if dry_run:
                            return {'status': 'not_executed', 'reason': 'No-cost dry run', 'gpu_cost_usd': 0}
                        result = self.storage.path('gpu_jobs/imported_results/' + package['job']['job_id'] + '/result.json')
                        if not result.exists():
                            raise WaitingForArtifact('Import verified manual GPU result for job ' + package['job']['job_id'])
                        return {'status': 'imported', 'result': read_json(result), 'directory': str(result.parent)}
                    training = stage('TRAIN', train_stage)
                    def evaluation_stage():
                        if dry_run:
                            return {'status': 'unavailable', 'reason': 'No trained checkpoint or independent predictions', 'release_ready': False}
                        path = self.storage.path(f'experiments/{run_id}/cycle_{cycle:03d}/comparison_input.json')
                        if not path.exists():
                            raise WaitingForArtifact('Evaluate baseline and candidate, then supply comparison_input.json')
                        comparison = read_json(path)
                        from .evaluation import compare
                        if comparison['candidate']['adapter_sha256'] != digest(training['result']['adapter_files']):
                            raise ValueError('Evaluation candidate differs from imported adapter')
                        return {'status': 'evaluated', 'comparison': compare(comparison['baseline'], comparison['candidate']), **comparison}
                    evaluation = stage('EVALUATE', evaluation_stage)
                    def promotion_stage():
                        if evaluation['status'] != 'evaluated':
                            return {'status': 'not_eligible', 'release_ready': False}
                        from .evaluation import promote
                        best = self.storage.path('checkpoints/best/registry.json')
                        return promote(self.storage, Path(training['directory']) / 'adapter', evaluation['baseline'],
                                       evaluation['candidate'], read_json(best) if best.exists() else None)
                    promotion = stage('PROMOTE_OR_REJECT', promotion_stage)
                    def analyze_stage():
                        summary = planner_input(evaluation.get('candidate'), dataset['statistics'],
                                                dataset['statistics']['rejection_reasons'], previous=plan)
                        summary['rendering_rejections'] = len(verified['quarantine']) + len(expanded['quarantine'])
                        summary['accepted_examples'] = len(dataset['rows'])
                        summary['performance_improvement_per_dollar'] = None
                        summary['api_cost_usd'] = 0 if not self.config['teacher_enabled'] or dry_run else 'see cumulative API usage ledger'
                        return summary
                    feedback = stage('ANALYZE_FAILURES', analyze_stage)
                    non_improving += int(not promotion.get('accepted', False))
                    next_step = stage('NEXT_CYCLE', lambda: {'next_cycle': cycle + 1,
                                     'stop': cycle >= self.config['max_cycles'] or non_improving >= self.config['max_non_improving'],
                                     'reason': 'cycle_or_non_improvement_limit' if not dry_run else 'dry_run_complete'})
                    previous = plan
                    if next_step['stop']:
                        break
                status = 'dry_run_complete' if dry_run else 'completed'
            except WaitingForArtifact:
                control = self.state.status(run_id)['run']['control']
                status = {'pause': 'paused', 'cancel': 'cancelled'}.get(control, 'waiting_for_artifact')
            except BaseException as exc:
                with self.state.connect() as c:
                    c.execute('UPDATE runs SET status=? WHERE id=?', ('interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', run_id))
                raise
            with self.state.connect() as c:
                c.execute('UPDATE runs SET status=? WHERE id=?', (status, run_id))
            return self.state.status(run_id)

    @staticmethod
    def _case_focus(plan):
        return {'missing': plan['focus']['missing_documents'], 'date_conflict': plan['focus']['cross_document_consistency'],
                'incomplete': plan['focus']['field_extraction'], 'unreadable': plan['focus']['evidence_quality']}
