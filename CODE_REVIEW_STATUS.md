# Distillation loop review status

- Branch for this review: `codex/distillation-corrections`
- Last external review reference supplied in the brief: `b61ea8b29cc88794e6ef8427b3d7d950f772faea`
- Corrective implementation under review: `a64bd67f8d489c465d096ef0deed8867dfc51301`
- Latest external outcome: pending. The supplied brief was the review context.

| Confirmed issue | Resolution | Verification |
| --- | --- | --- |
| 98% validity target blocked experimental continuation | Separate 94% experimental floor, relative decline and incomplete limits from the 98% quality target | `test_high_uncertainty_and_9472_validity_can_allow_an_experiment`, saved-artifact CPU preflight |
| Positive conservative interval required for continuation | Report observed effect, task-stratified paired bootstrap and Hoeffding interval. Safe inconclusive outcomes can be explicitly authorized | `test_bootstrap_is_repeatable_and_paired_within_tasks`, complete local semantic replay |
| Statistical tests could mask substantial deterioration | Run hard per-case and task safety checks independently, with case IDs and reasons in the decision | `test_task_and_safety_events_reject_despite_inconclusive_sign_test`, `test_multiple_task_deterioration_and_duplicate_stop_labels` |
| Latest candidate could start the next round, with comparisons against the original baseline | Compare against the accepted best, persist the next starting adapter and selection record together, and preserve rejected artifacts | `test_resume_and_rejections_use_current_best`, existing evaluation and backup interruption tests |

The final CPU suite ran 199 tests. 195 passed and 4 were skipped. All four
notebooks parsed, matched the tracked ZIP checksum, and the packaged CPU
modules imported. The saved preflight reported 94.72% SFT validity, a decline
of 1.11 points from initial, and 2.78% incomplete. There are 286 pending
judgments, with 573 cached and a conservative ceiling of USD 8.236834500.
The full semantic decision remains pending and no model improvement is claimed.

Held-out loss has no independent candidates. The exclusion registry now checks
available earlier training plans, correction records, and prediction receipts
against their scenario records. Additional exposure journals require input
from their owners. The repeat-judge plan and analyzer run offline. Collection
of new repetitions and a per-scenario concurrent paid workflow remain deferred.
The existing cumulative budget and approval guards passed their mock tests.
No paid calls, new training, or historical artifact changes were made.
