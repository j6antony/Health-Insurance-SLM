"""Score the saved Pass A checkpoint comparison on CPU with an offline preflight."""
import argparse
import json
import os
import re
from collections import Counter
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from statistics import mean

from self_instruct_pipeline.budget import APIBudget, reservation_usd
from self_instruct_pipeline.config import ROOT, load_config
from self_instruct_pipeline.parent_backend import OpenAIParent, create_parent
from self_instruct_pipeline.persistence import Store, atomic_json, atomic_jsonl, digest, read_jsonl
from self_instruct_pipeline.usage import PRICING_SOURCE, RATES, UsageLogger

from .benchmarks import load_saved_split
from .checkpoint_eval import STAGES, checkpoint_config, generation_summary, saved_predictions
from .core import aggregate_scores
from .correction import ask, judge_messages
from .paired import paired_summary
from .review import grade_reviewed
from .semantic import (JUDGMENT_INSTRUCTION, JUDGMENT_OPERATION, JUDGMENT_SCHEMA,
                       SemanticJudge, judgment_payload, read_judgment)

CORPUS = ROOT / 'data/evaluation/fixed_validation_v1.jsonl'
SCHEMAS = ROOT / 'data/self_instruct/task_schemas.json'


def load_comparison(directory, cases, checkpoint_root=None):
    manifest = json.loads((directory / 'manifest.json').read_text())
    ids = [item['record']['id'] for item in cases]
    if manifest['case_ids'] != ids or manifest['benchmark_hash'] != digest(cases):
        raise ValueError('Comparison manifest does not match the frozen development benchmark')
    fingerprints = manifest['checkpoints']
    if (set(fingerprints) != set(STAGES) or any(
            set(record) != {'config', 'weights'}
            or any(not isinstance(h, str) or not re.fullmatch('[a-f0-9]{64}', h) for h in record.values())
            for record in fingerprints.values())
            or len({record['weights'] for record in fingerprints.values()}) != len(STAGES)):
        raise ValueError('Missing, malformed, or duplicate checkpoint identities')
    if not isinstance(manifest['model'], str) or not manifest['model']:
        raise ValueError('Missing base model identity')
    if checkpoint_root is not None:
        model, actual = checkpoint_config({s: checkpoint_root / p for s, p in STAGES.items()},
                                          {'training': {}})
        if model != manifest['model'] or actual != fingerprints:
            raise ValueError('Adapter files do not match the saved checkpoint identities')
    predictions = {}
    for stage in STAGES:
        rows = read_jsonl(directory / f'A_{stage}_strict-reference.jsonl')
        generation = json.loads((directory / f'A_{stage}_generation.json').read_text())
        if len(rows) != len(ids) or len(generation) != len(ids):
            raise ValueError(f'Incomplete Pass A predictions for {stage}')
        for case, row, pred in zip(cases, rows, generation, strict=True):
            record = case['record']
            if (row.get('scenario_id') != record['id'] or row.get('task') != record['task']
                    or not isinstance(pred.get('raw'), str) or type(pred.get('truncated')) is not bool
                    or any(k not in row or row[k] != v for k, v in pred.items())):
                raise ValueError(f'Raw prediction mismatch for {stage}/{record["id"]}')
        expected = manifest.get('reuse_original', {}).get(f'A/{stage}')
        if expected is not None and digest(generation) != expected:
            raise ValueError(f'Original generation hash mismatch for {stage}')
        predictions[stage] = generation
    return manifest, predictions


class JudgmentPlan:
    def __init__(self, directory, schemas, settings, legacy_caches, historical):
        self.store = Store(directory)
        self.schemas = schemas
        self.settings = settings
        self.legacy_caches = legacy_caches
        self.historical = historical
        self.cached = {}
        self.pending = {}
        self.parent = None

    def client(self):
        return self.parent

    def identify(self, item, prediction):
        payload = judgment_payload(item, prediction, self.schemas)
        body = OpenAIParent.request_body(self.settings, judge_messages(payload, JUDGMENT_INSTRUCTION),
                                         JUDGMENT_SCHEMA, temperature=0.0)
        key = digest([SemanticJudge.version, self.settings['provider'], body])
        return key, payload, body

    def inspect(self, item, prediction):
        score, _ = grade_reviewed(item, prediction, self.schemas, {})
        if score is not None:
            return None
        key, payload, body = self.identify(item, prediction)
        path = self.store.path(key + '.json')
        if path.is_file():
            self.cached[key] = read_judgment(path)
        else:
            # The old code used the full generation object; regrade_saved_run
            # used the raw/truncated projection of those same inputs.
            keys = {digest([SemanticJudge.version, item, pred]) for pred in
                    (prediction, {k: prediction[k] for k in ('raw', 'truncated')})}
            matches = [read_judgment(cache / (legacy + '.json'))
                       for cache in self.legacy_caches for legacy in sorted(keys)
                       if (cache / (legacy + '.json')).is_file()]
            matches.extend(self.historical[k] for k in sorted(keys) if k in self.historical)
            if matches:
                if any(value != matches[0] for value in matches[1:]):
                    raise ValueError(f'Conflicting legacy semantic judgments for {item["record"]["id"]}')
                self.cached[key] = matches[0]
        if key in self.cached:
            self.pending.pop(key, None)
            return key
        self.pending.setdefault(key, (payload, reservation_usd(self.settings, body)))
        return key

    def resolve(self, item, prediction):
        key, _, _ = self.identify(item, prediction)
        if key not in self.cached:
            if self.parent is None or key not in self.pending:
                raise ValueError('Unapproved or unplanned semantic judgment')
            payload, _ = self.pending[key]
            judgment = ask(self, payload, JUDGMENT_INSTRUCTION, JUDGMENT_SCHEMA, JUDGMENT_OPERATION)
            atomic_json(self.store.path(key + '.json'), judgment)
            self.cached[key] = judgment
        return self.cached[key]


def historical_judgments(root, cases, generations):
    """Recover validated rubric dimensions from the original frozen scoring journals."""
    from self_instruct_pipeline.validation import validate_schema

    judgments = {}
    for stage, folder in [('initial', root / 'baseline'), ('sft_orpo', root / 'round_01/lighteval')]:
        original, rows, scoring = saved_predictions(folder, cases)
        if scoring != SemanticJudge.version or original != generations[stage]:
            raise ValueError(f'Incompatible historical semantic journal at {folder}')
        for item, prediction, row in zip(cases, original, rows, strict=True):
            if 'semantic_dimensions' in row:
                judgment = row['semantic_dimensions']
                validate_schema(judgment, JUDGMENT_SCHEMA)
                judgments[digest([scoring, item, prediction])] = judgment
    return judgments


def score(cases, generations, plan, store, quality, thresholds, identity):
    judge = SemanticJudge(None, plan.store.directory, plan.schemas, resolve_judgment=plan.resolve)
    runs, rows_by_stage = {}, {}
    for stage, predictions in generations.items():
        rows = []
        for item, prediction in zip(cases, predictions, strict=True):
            record = item['record']
            rows.append({'scenario_id': record['id'], 'task': record['task'],
                         **judge.grade(item, prediction)})
        rows_by_stage[stage] = rows
        atomic_jsonl(store.path(f'{stage}_semantic_cases.jsonl'), rows)
        runs[stage] = {
            **aggregate_scores(rows, plan.schemas, 20), **generation_summary(predictions),
            'grading_basis_counts': dict(Counter(row['grading_basis'] for row in rows)),
            'unsupported_claims': sum(row.get('semantic_dimensions', {}).get('unsupported_claims', False) for row in rows),
            'review_flag_correct_rate': mean(row['review_flag_correct'] for row in rows),
            'per_task_diagnostics': {
                task: {field: mean(row[field] for row in rows if row['task'] == task)
                       for field in ('schema_evidence_valid', 'review_flag_correct', 'reference_evidence_overlap')}
                for task in plan.schemas
            },
        }
    pairs = {}
    for left, right in combinations(generations, 2):
        joined = list(zip(rows_by_stage[left], rows_by_stage[right], strict=True))
        changes = [{'scenario_id': a['scenario_id'], 'task': a['task'],
                     'before': {k: a[k] for k in ('task_score', 'schema_evidence_valid', 'review_flag_correct', 'grading_basis')},
                     'after': {k: b[k] for k in ('task_score', 'schema_evidence_valid', 'review_flag_correct', 'grading_basis')}}
                   for a, b in joined if any(a[k] != b[k] for k in
                   ('task_score', 'schema_evidence_valid', 'review_flag_correct', 'grading_basis'))]
        pairs[f'{right} minus {left}'] = {
            **{field: paired_summary([b[field] - a[field] for a, b in joined])
               for field in ('task_score', 'schema_evidence_valid', 'review_flag_correct')},
            'per_task': {task: paired_summary([b['task_score'] - a['task_score'] for a, b in joined
                                              if a['task'] == task]) for task in plan.schemas},
            'changed_cases': changes,
        }
    from .quality import regression_gate
    for stage, report in runs.items():
        report['case_scores'] = {r['scenario_id']: {'task': r['task'], 'score': r['task_score']}
                                 for r in rows_by_stage[stage]}
    safety = regression_gate(runs['initial'], runs['sft'], quality)
    comparison = pairs['sft minus initial']['task_score']
    delta = comparison['delta']
    validity_ok = runs['sft']['schema_evidence_valid_rate'] >= thresholds['schema_evidence_valid_rate']
    review_ok = runs['sft']['review_flag_correct_rate'] >= runs['initial']['review_flag_correct_rate']
    unsupported_ok = runs['sft']['unsupported_claims'] <= runs['initial']['unsupported_claims']
    decision = {
        'outcome': ('better_within_development_gate' if comparison['conservative_95pct_ci'][0] > 0
                    and safety['accepted'] and validity_ok and review_ok and unsupported_ok else
                    'equal_observed' if delta == 0 else 'worse_observed' if delta < 0 else 'inconclusive'),
        'validity_threshold_met': validity_ok, 'review_flags_nonregressing': review_ok,
        'unsupported_claim_count_nonincreasing': unsupported_ok, 'task_regression_gate': safety,
        'continuation': 'paused_pending_user_acceptance_of_semantic_result_and_explicit_authorization',
    }
    if decision['outcome'] != 'better_within_development_gate':
        decision['investigate'] = ['training data correctness and representativeness', 'student assessment selection',
                                   'review flags and task labels', 'benchmark and judge consistency',
                                   'training and held-out loss', 'learning rate, optimizer and masking']
    result = {'identity': identity, 'scoring': judge.version, 'scope': 'development synthetic silver',
              'runs': runs, 'paired': pairs, 'diagnostic_decision': decision}
    atomic_json(store.path('semantic_comparison.json'), result)
    return result


def run(comparison_dir, split, output, *, corpus=CORPUS, schemas_path=SCHEMAS,
        checkpoint_root=None, legacy_caches=(), historical_run=None, approval=None, max_usd=None):
    from .core import load_loop_config
    schemas = json.loads(schemas_path.read_text())['tasks']
    cases, holdout = load_saved_split(split, corpus, schemas)
    manifest, generations = load_comparison(comparison_dir, cases, checkpoint_root)
    if output.resolve() == comparison_dir.resolve():
        raise ValueError('Use a separate diagnostic output directory')
    if Path('/content/drive/MyDrive').is_dir():
        os.environ.setdefault('DISTILLATION_LOCAL_LOCK_DIR', '/content/distillation-eval-locks')
    store = Store(output)
    settings = load_config()['parent']
    from .colab import distillation_1000_configs
    _, config = distillation_1000_configs(load_config(), load_loop_config(), output)
    prior = historical_judgments(historical_run, cases, generations) if historical_run else {}
    with store.lock():
        plan = JudgmentPlan(store.path('semantic_cache'), schemas, settings, legacy_caches, prior)
        keys = {stage: [plan.inspect(item, prediction)
                        for item, prediction in zip(cases, predictions, strict=True)]
                for stage, predictions in generations.items()}
        references = {key for values in keys.values() for key in values if key is not None}
        tasks = {}
        for stage, values in keys.items():
            counts = Counter(locally_scored=0, cache_hits=0, missing_judgments=0)
            for key in values:
                counts['locally_scored' if key is None else 'cache_hits' if key in plan.cached
                       else 'missing_judgments'] += 1
            tasks[stage] = dict(counts)
        identity = {
            'split_sha256': digest(json.loads(split.read_text())), 'comparison_manifest_sha256': digest(manifest),
            'development_sha256': digest(cases), 'holdout_sha256': digest(holdout),
            'cases_per_checkpoint': len(cases),
            'schema_sha256': digest(schemas), 'checkpoints': manifest['checkpoints'],
            'checkpoint_identity': 'adapter_files_verified' if checkpoint_root else 'recorded_manifest_fingerprints',
            'generations': {stage: digest(rows) for stage, rows in generations.items()},
        }
        identity_path = store.path('inputs.json')
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError('Inputs changed; use a separate diagnostic output directory')
        atomic_json(identity_path, identity)
        attempts = settings['max_retries'] + 1
        reservation = attempts * sum((entry[1] for entry in plan.pending.values()), Decimal(0))
        ledger_path = store.path('judge_budget/state.json')
        ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else {}
        accounted = Decimal(ledger.get('accounted_usd', '0'))
        preflight = {
            'identity': identity, 'scoring': SemanticJudge.version, 'judge_model': settings['model'],
            'judge_provider': settings['provider'], 'request_rates_usd_per_million': list(RATES[settings['provider'], settings['model']]),
            'pricing_source': PRICING_SOURCE, 'per_checkpoint': tasks,
            'unique_cache_hits': len(references & plan.cached.keys()),
            'unique_missing_judgments': len(plan.pending), 'max_attempts_per_judgment': attempts,
            'max_remaining_requests': attempts * len(plan.pending),
            'already_accounted_usd': str(accounted), 'max_additional_usd': str(reservation),
            'max_cumulative_usd': str(accounted + reservation),
            'missing_inputs_sha256': digest(sorted(plan.pending)),
        }
        preflight['approval_hash'] = digest(preflight)
        atomic_json(store.path('cost_preflight.json'), preflight)
        if plan.pending and approval is None:
            return preflight
        if plan.pending:
            if approval != preflight['approval_hash']:
                raise ValueError('Approval does not match the current cost preflight')
            cap = Decimal(str(max_usd))
            if not cap.is_finite() or cap < accounted + reservation:
                raise ValueError('Authorized cumulative cap must cover the complete worst-case preflight')
            calls = ledger.get('calls', 0) + preflight['max_remaining_requests']
            calls = max(calls, ledger.get('limits', {}).get('max_calls', 0))
            budget = APIBudget(store.path('judge_budget'), max_calls=calls, max_cost_usd=cap)
            with budget.activate(allow_increase=True):
                plan.parent = create_parent(settings, logger=UsageLogger(store.path('benchmark_judging')))
                score(cases, generations, plan, store, config['quality'], config['thresholds'], identity)
        else:
            score(cases, generations, plan, store, config['quality'], config['thresholds'], identity)
        return preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('comparison-dir', 'split', 'output'):
        parser.add_argument(f'--{arg}', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, default=CORPUS)
    parser.add_argument('--schemas', dest='schemas_path', type=Path, default=SCHEMAS)
    parser.add_argument('--checkpoint-root', type=Path)
    parser.add_argument('--historical-run', type=Path,
                        help='Original baseline and Round 1 semantic journals from the same judge')
    parser.add_argument('--legacy-cache', dest='legacy_caches', action='append', type=Path, default=[],
                        help='Original source-rubric-v2 cache written using the same judge settings and schema')
    parser.add_argument('--approve-preflight', dest='approval')
    parser.add_argument('--max-usd')
    args = parser.parse_args()
    if (args.approval is None) != (args.max_usd is None):
        parser.error('Paid execution requires both --approve-preflight and --max-usd')
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
