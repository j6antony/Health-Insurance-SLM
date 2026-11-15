"""Model promotion gates and explicit data shortfall accounting."""
from collections import Counter
import math


def regression_gate(baseline, candidate, settings):
    tolerances = settings.get('per_task_max_regression', {})
    default = settings.get('max_task_regression', 0.0)
    failures = {task: baseline['per_task'][task] - candidate['per_task'][task]
                for task in baseline['per_task']
                if baseline['per_task'][task] - candidate['per_task'][task] > tolerances.get(task, default) + 1e-12}
    if 'paired_regression_alpha' in settings:
        before, after = baseline['case_scores'], candidate['case_scores']
        if before.keys() != after.keys() or any(before[k]['task'] != after[k]['task'] for k in before):
            raise ValueError('Paired regression requires the same cases and tasks')
        alpha = settings['paired_regression_alpha'] / len(baseline['per_task'])
        paired = {}
        for task, drop in failures.items():
            changes = [after[k]['score'] - row['score'] for k, row in before.items() if row['task'] == task]
            losses = sum(x < -1e-12 for x in changes)
            gains = sum(x > 1e-12 for x in changes)
            discordant = losses + gains
            p_value = sum(math.comb(discordant, i) for i in range(losses, discordant + 1)) / 2**discordant
            paired[task] = {'drop': drop, 'losses': losses, 'gains': gains, 'p_value': p_value}
        confirmed = {task: row['drop'] for task, row in paired.items() if row['p_value'] <= alpha}
        return {'accepted': not confirmed, 'regressions_exceeding_tolerance': failures,
                'confirmed_regressions': confirmed, 'paired': paired, 'per_task_alpha': alpha}
    return {'accepted': not failures, 'regressions_exceeding_tolerance': failures}


def promotion_gate(baseline, candidate, settings, thresholds, tasks):
    from .core import satisfactory
    try:
        hard = operational_gate(baseline, candidate, settings)
    except SafetyEvidenceError as exc:
        return {'accepted': False, 'reason': str(exc)}
    result = regression_gate(baseline, candidate, settings)
    result['operational_safety'] = hard
    result['no_observed_high_risk_errors'] = not any(hard['safety_events'].values())
    improvement = candidate['macro_task_score'] - baseline['macro_task_score']
    result.update(macro_improvement=improvement,
                  thresholds_met=satisfactory(candidate, thresholds, tasks))
    result['accepted'] = (result['accepted'] and hard['accepted'] and result['no_observed_high_risk_errors']
                          and result['thresholds_met'] and improvement + 1e-12 >= settings.get('minimum_macro_improvement', 0.01))
    return result


def training_coverage(pairs, target, new_count, minimum_new):
    tasks = Counter(r['task'] for r, _ in pairs)
    checked = Counter(r['task'] for r, _ in pairs if r.get('answer_checks'))
    return {'target_verified_pairs': target, 'achieved_pairs': len(pairs),
            'shortfall': max(0, target - len(pairs)), 'target_reached': len(pairs) >= target,
            'new_validated': new_count, 'minimum_new_required': minimum_new,
            'tasks': dict(tasks), 'deterministic_checks_by_task': dict(checked),
            'deterministic_check_count': sum(checked.values()),
            'deterministic_check_rate': sum(checked.values()) / len(pairs) if pairs else 0.0,
            'template_reasoning_patterns': dict(Counter(f"{r['template_family']}:{r['reasoning_variant']}" for r, _ in pairs if r.get('template_family'))),
            'trusted_template_count': sum(bool(r.get('verification_basis')) for r, _ in pairs),
            'case_types': dict(Counter(r.get('case_type', 'unlabeled') for r, _ in pairs)),
            'difficulty': dict(Counter(r.get('difficulty', 'unlabeled') for r, _ in pairs)),
            'distinct_scenarios': len({r['id'] for r, _ in pairs}),
            'reference_status': 'synthetic silver; absent checks mean no deterministic verification'}


def validate_quality_settings(config, tasks=None):
    settings = config.get('quality', {})
    allowed = {'max_task_regression', 'per_task_max_regression', 'minimum_macro_improvement',
               'minimum_new_pair_fraction', 'minimum_deterministic_check_rate', 'deterministic_template_fraction',
               'paired_regression_alpha', 'experimental_safety'}
    if not isinstance(settings, dict) or set(settings) - allowed:
        raise ValueError('Unknown quality setting')
    for name in ('max_task_regression', 'minimum_macro_improvement', 'minimum_new_pair_fraction', 'minimum_deterministic_check_rate', 'deterministic_template_fraction'):
        default = {'minimum_new_pair_fraction': 0.5, 'minimum_macro_improvement': .01}.get(name, 0.0)
        value = settings.get(name, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'quality.{name} must be finite in [0, 1]')
    if settings.get('minimum_new_pair_fraction', .5) == 0:
        raise ValueError('A round must require new data')
    if 'paired_regression_alpha' in settings:
        alpha = settings['paired_regression_alpha']
        if type(alpha) not in (int, float) or not 0 < alpha < 1:
            raise ValueError('paired_regression_alpha must be in (0, 1)')
    overrides = settings.get('per_task_max_regression', {})
    if not isinstance(overrides, dict) or (tasks is not None and set(overrides) - set(tasks)):
        raise ValueError('Unknown regression task')
    for value in overrides.values():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid task regression tolerance')
    experimental_limits(settings)
    return settings


EXPERIMENTAL_LIMITS = {
    'minimum_validity': .94,
    'maximum_validity_decline': .015,
    'maximum_incomplete_rate': .03,
    'task_warning': .05,
    'single_task_rejection': .15,
    'multiple_task_rejection': .10,
    'maximum_regressing_tasks': 3,
}


def experimental_limits(settings):
    overrides = settings.get('experimental_safety', {})
    if not isinstance(overrides, dict) or set(overrides) - EXPERIMENTAL_LIMITS.keys():
        raise ValueError('Unknown experimental safety limit')
    limits = {**EXPERIMENTAL_LIMITS, **overrides}
    for key, value in limits.items():
        if key == 'maximum_regressing_tasks':
            if type(value) is not int or value < 0:
                raise ValueError(f'Invalid experimental safety limit {key}')
        elif type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Invalid experimental safety limit {key}')
    return limits


class SafetyEvidenceError(ValueError):
    """An automated promotion cannot be evaluated from the saved evidence."""


def operational_gate(comparator, candidate, settings):
    """Reject observed safety regressions independently of sample significance."""
    from .core import validate_report

    limits = experimental_limits(settings)
    for summary in (comparator, candidate):
        validate_report(summary, comparator['per_task'], 1)
    before, after = comparator.get('case_scores'), candidate.get('case_scores')
    flags = ('valid', 'incomplete', 'unsupported_claims', 'missed_required_review')
    for rows in (before, after):
        if not isinstance(rows, dict) or not rows or any(
                not isinstance(key, str) or not key.strip() or not isinstance(row, dict)
                or not isinstance(row.get('task'), str) or any(type(row.get(flag)) is not bool for flag in flags)
                for key, row in rows.items()):
            raise SafetyEvidenceError('Promotion requires complete per-case safety evidence')
    if before.keys() != after.keys() or any(row['task'] != before[key]['task'] for key, row in after.items()):
        raise SafetyEvidenceError('Promotion requires matching case IDs and tasks for safety evidence')
    for field in ('benchmark_hash', 'scoring', 'evaluation_protocol'):
        value = comparator.get(field)
        if not isinstance(value, str) or not value.strip() or value != candidate.get(field):
            raise SafetyEvidenceError(f'Promotion requires nonempty, matching {field} for safety evidence')
    for summary, rows in ((comparator, before), (candidate, after)):
        counts = Counter(row['task'] for row in rows.values())
        totals = Counter()
        for row in rows.values():
            value = row.get('score')
            if type(value) not in (float, int) or not 0 <= value <= 1:
                raise SafetyEvidenceError('Invalid per-case score for safety evidence')
            totals[row['task']] += value
        if (counts != summary['examples_per_task'] or counts.keys() != summary['per_task'].keys()
                or any(abs(totals[t] / counts[t] - summary['per_task'][t]) > 1e-9 for t in counts)
                or abs(sum(row['valid'] for row in rows.values()) / len(rows)
                       - summary['schema_evidence_valid_rate']) > 1e-9):
            raise SafetyEvidenceError('Per-case safety evidence disagrees with score summary')

    declines = {t: value - candidate['per_task'][t] for t, value in comparator['per_task'].items()}

    def declining(limit):
        return {t: drop for t, drop in declines.items() if drop > 0 and drop + 1e-12 >= limit}

    def losses(tasks):
        return sorted(key for key, row in after.items()
                      if row['task'] in tasks and row['score'] < before[key]['score'])

    reasons = []
    def reject(code, cases, **details):
        reasons.append({'code': code, 'cases': sorted(cases), **details})

    validity = candidate['schema_evidence_valid_rate']
    decline = comparator['schema_evidence_valid_rate'] - validity
    if validity + 1e-12 < limits['minimum_validity']:
        reject('validity_below_floor', (k for k, row in after.items() if row.get('valid') is False), rate=validity)
    if decline > limits['maximum_validity_decline'] + 1e-12:
        reject('validity_decline', (k for k, row in after.items() if not row.get('valid', True)
                                    and before[k].get('valid')), decline=decline)
    incomplete = [k for k, row in after.items() if row.get('incomplete')]
    incomplete_rate = len(incomplete) / len(after)
    if incomplete_rate > limits['maximum_incomplete_rate'] + 1e-12:
        reject('incomplete_generation', incomplete, rate=incomplete_rate)
    single = declining(limits['single_task_rejection'])
    if single:
        reject('single_task_regression', losses(single), tasks=single)
    multiple = declining(limits['multiple_task_rejection'])
    if len(multiple) > limits['maximum_regressing_tasks']:
        reject('multiple_task_regression', losses(multiple), tasks=multiple)
    for field, reason in [('unsupported_claims', 'new_unsupported_claim'),
                          ('missed_required_review', 'new_missed_required_review')]:
        introduced = [k for k, row in after.items() if row.get(field) and not before[k].get(field)]
        if introduced:
            reject(reason, introduced)
    return {'accepted': not reasons, 'reasons': reasons, 'limits': limits,
            'incomplete_rate': incomplete_rate, 'case_evidence_complete': True,
            'task_regression_warnings': declining(limits['task_warning']),
            'safety_events': {field: sorted(k for k, row in after.items() if row.get(field))
                              for field in flags[2:]}}


def diagnostic_decision(comparator, candidate, paired, settings, *, quality_target=.98):
    safety = operational_gate(comparator, candidate, settings)
    statistical = regression_gate(comparator, candidate, settings)
    delta = paired['delta']
    practical_limit = settings.get('minimum_macro_improvement', .01)
    interval_key = next((key for key in ('clustered_bootstrap_95pct_ci', 'bootstrap_95pct_ci',
                                        'conservative_95pct_ci') if key in paired), None)
    if interval_key is None:
        raise ValueError('Semantic diagnostic requires a paired uncertainty interval')
    lower, upper = paired[interval_key]
    measured = candidate['macro_task_score'] - comparator['macro_task_score']
    if (not all(type(value) in (int, float) and math.isfinite(value) for value in (delta, lower, upper))
            or not -1 <= lower <= upper <= 1 or abs(measured - delta) > 1e-9):
        raise ValueError('Invalid paired change or confidence interval')
    outcome = ('positive_interval' if lower > 0 and delta > 0 else
               'negative_interval' if upper < 0 and delta < 0 else 'inconclusive')
    return {'outcome': outcome, 'observed_macro_change': delta,
            'practical_effect_threshold': practical_limit, 'practical_gain_observed': delta >= practical_limit,
            'quality_target': quality_target,
            'quality_target_met': candidate['schema_evidence_valid_rate'] >= quality_target,
            'operational_safety': safety, 'statistical_regression': statistical,
            'uncertainty_interval': interval_key,
            'experimental_continuation_eligible': safety['accepted'] and outcome != 'negative_interval',
            'experimental_promotion_eligible': (safety['accepted'] and statistical['accepted']
                                                and delta >= practical_limit and delta > 0),
            'release_eligible': False,
            'continuation': 'requires_explicit_authorization'}


def require_experimental_result(result, quality):
    """Recompute the pre-training decision using the active safety limits."""
    try:
        before, after = (result['runs'][stage] for stage in ('initial', 'sft'))
        paired = result['paired']['sft minus initial']['task_score']
        if 'clustered_bootstrap_95pct_ci' not in paired:
            raise ValueError('Semantic SFT-only result requires a source-family bootstrap report')
    except (KeyError, TypeError) as exc:
        raise ValueError('Semantic SFT-only result requires investigation or a fresh diagnostic report') from exc
    decision = diagnostic_decision(before, after, paired, quality)
    if not decision['experimental_continuation_eligible']:
        codes = ', '.join(reason['code'] for reason in decision['operational_safety']['reasons'])
        raise ValueError(f'Semantic SFT-only result requires investigation before training: {codes or decision["outcome"]}')
    return decision
