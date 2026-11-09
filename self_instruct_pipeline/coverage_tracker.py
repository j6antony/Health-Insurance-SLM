from collections import Counter

TARGETS = {'supported_or_no_issue': 8, 'insufficient_ambiguous_unreadable': 8,
           'conflicting_or_misleading': 6, 'conditional_logic_or_date': 4,
           'injection_or_unsupported_action': 2}


def coverage_state(records, scheduler):
    window = records[-40:]
    counts = Counter()
    for record in window:
        counts.update(set(record.get('coverage_dimensions', [])))
    total = sum(scheduler.counts.values())
    weight_total = sum(scheduler.weights.values())
    return {
        'accepted_total': len(records), 'window_size': 40, 'window_accepted_count': len(window),
        'task_counts': dict(scheduler.counts), 'task_weights': scheduler.weights,
        'task_deficits_at_current_total': {
            t: round(total * scheduler.weights[t] / weight_total - scheduler.counts[t], 3)
            for t in scheduler.tasks},
        'difficulty_counts': dict(Counter(r['difficulty'] for r in records)),
        'case_type_counts': dict(Counter(r['case_type'] for r in records)),
        'skill_counts': dict(Counter(s for r in records for s in set(r['skills']))),
        'observed_coverage_counts': {k: counts[k] for k in TARGETS},
        'remaining_targets': {k: max(0, target - counts[k]) for k, target in TARGETS.items()},
        'coverage_basis': 'automated source-text scenario audit; overlapping dimensions, not independently verified',
        'priority_skills': [k for k, target in TARGETS.items() if counts[k] < target],
    }
