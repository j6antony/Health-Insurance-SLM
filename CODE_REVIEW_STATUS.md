# Distillation loop review status

- Branch for this review: `codex/distillation-corrections`
- Last external review reference supplied in the brief: `b61ea8b29cc88794e6ef8427b3d7d950f772faea`
- Corrective implementation under review: `a64bd67f8d489c465d096ef0deed8867dfc51301`
- Latest external outcome: Round 1 returned clarifications F1–F5 against the initial corrective submission. Re-review is pending.

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


## External review round 1

Corrective implementation: `76c1db54a63788e19de8a2339a7b0a355163dd68`, based
on review tip `38281b2ab6dea15defef598bd3ef9fbdce4dc06e`.

| Finding | Resolution | Targeted verification |
| --- | --- | --- |
| F1: Candidate restore and cache provenance | Restore immediately before evaluation, bind caches to verified configuration and weight contents, and track the runtime's successful restore. Refuse legacy live caches without that binding. | Interrupted SFT receipt inference, unchanged hash reuse, changed weights, and unbound cache |
| F2: Promotion without complete evidence | Require matching benchmark, scoring protocol and per-case safety evidence in development and reviewed release selection. Recheck previously accepted reviewed release receipts on resume. | Aggregate-only, empty identities, and mismatched case IDs are denied |
| F3: Paired uncertainty ignored source families | Keep the fixed cases, resample whole source families within tasks, weight task means equally, and report both intervals and seed. | Unequal family sizes and task sizes give the expected point estimate and interval |
| F4: Stale Colab extraction | Verify ZIP and extracted files before either diagnostic subprocess. Stage and replace stale code safely. | Older extraction is replaced and a bad ZIP is rejected even with an existing project |
| F5: Implicit backup redirect or destination | Resolve one configured destination, record a blocked experimental GitHub upload, and distinguish local archive verification from Drive-mount verification. | No redirect or uploader call, local archive label, and unmounted Drive rejection |

Seven focused tests passed before the full CPU run. The full suite ran 206
tests, with 202 passing and four skipped. Verified notebook syntax, bundle
checksum and integrity, and isolated CPU imports. No live GPU trainer,
mounted Drive, or paid API was exercised. Existing frozen artifacts were
preserved. The SFT-only semantic result and external re-review both remain
pending. The previous deferred concurrency and held-out loss work is unchanged.

## Semantic safety investigation checkpoint

The 46 flagged development cases reconcile with the saved diagnostic and frozen
sources. The provisional audit records 31 source-supported errors, five apparent
judge false positives, one possible rubric issue and nine second-review cases.
The 200 selected historical pairs passed the frozen split exclusion check.
Review the audit, root cause analysis, and blocked one-round design in `reports/`
and `config/experiments/`. The original diagnostic still blocks continuation.

The dedicated `distillation_colab_semantic_remediation_round1.ipynb` loads the
versioned experiment specification from the verified bundle. It requires all
200 new verified pairs and leaves release scoring disabled. Launch requires a
qualifying semantic result and explicit continuation authorization.

The user reported independent review complete and authorized making the old
diagnostic advisory for the dedicated one-round experiment. That notebook
still requires explicit launch authorization. Candidate safety and rollback
are enforced by the existing controller.

The controlled Colab run completed its baseline but stopped during generation.
A raw scenario carries a batch-local generation ID and has no permanent ID until
acceptance. The exclusion registry now applies content, family and near-duplicate
checks at that stage. Unexpected programming errors propagate without a
generation retry. HTTP 400 invalid_prompt was a separate provider rejection
with no disclosed cause. Its sanitized metadata is retained without retry.

The original rejection journal contains 263 candidate rows across 33 attempts.
Recovery can freshly validate and audit a selected legacy rejection under the
same frozen exclusions. Canary and recovery use the existing cumulative budget
and live Supabase sync. A one-batch canary also verifies one teacher target
before the operator resumes training.
