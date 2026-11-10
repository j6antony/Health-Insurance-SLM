# Automatic mode is now the default

Full notebook runs use `evaluation.mode = "automatic"` and `TEST_RUN = False`.
The existing pipeline generates and freezes synthetic validation references (30
per task) and checks them with schema/evidence validation and the parent audit.
No manually reviewed input files, human reviewer metadata, or adjudication queue
are required in this mode. Failed answers still require bounded repair or replacement.
Generated records are `automatically_checked_silver` with `human_reviewed: false`.

Automatic training and GitHub checkpoint uploads proceed normally. The best
validation checkpoint is restored and recorded as `selected_checkpoint`. The
independent-gold `promotion` gate is not claimed as satisfied. This setting does
not turn synthetic references into independent clinical ground truth.
Use the new `distillation_loop_automatic_v1` output folder for a full run; previous
run folders remain intact. API budgets are still enforced. Preflight lists the
540-reference generation target rather than requiring missing reviewed files.

The contract below applies only when explicitly selecting `evaluation.mode = "reviewed"`.

# Reviewed evaluation and adjudication

The default run requires `validation.jsonl` and `test.jsonl`, each with at least
30 independently reviewed examples for each of the 18 tasks (540 per split).
Teacher-generated examples must not be relabeled as independently reviewed gold.

Run the read-only readiness check before spending API credits or loading a model:

```sh
python3 -m distillation_loop.preflight
# Data/configuration only, on a machine without the training environment:
python3 -m distillation_loop.preflight --skip-runtime
```

A benchmark JSONL line has `record`, `reference`, and:

```json
{"review": {"status": "approved", "reviewer": "reviewer-id", "source_group": "case-family-id"}}
```

`record` contains `id`, `task`, `instruction`, and `scenario.documents`/`context`,
using the existing task answer schemas for `reference`. Reasoning tasks also need
a reviewed rubric, for example:

```json
{"rubric": {"decision": "authorization_required", "required_facts": ["Authorization is required before submission."]}}
```

Review the decision, facts, exceptions, and abstention against applicable source
rules. Reviewer metadata is an attestation, not proof. Optional
`accepted_references` contains other **complete, independently approved answers**
using the same schema. Do not add arbitrary model outputs to this allowlist.

Keep case/source families in one split. Include appropriately deidentified,
realistic records with missing documentation, contradictions, ambiguous policies,
AND/OR rules, numeric/date boundaries, and abstention. The loader rejects repeated
IDs, near duplicates, shared source groups, and identical normalized page text
across validation/test, even if document IDs change. Local duplicate checks exclude
both splits from new teacher scenarios and executable templates. Heuristic overlap
checks cannot establish semantic independence or eliminate pretraining contamination.

## Scoring and human adjudication

Document classification checks the class **and** review flag. Extraction scores
supported field/value/review tuples. Evidence must pass source validation.
Matching codes do not automatically approve a contradictory prose explanation.

For reviewed reasoning tasks, full substantive agreement with a reviewed answer
or approved alternative can pass automatically; evidence spans and array order may
differ when independently source-valid. Other wording is **ungraded pending review**,
not automatically wrong or automatically correct. The evaluator writes
`review_queue.jsonl`, freezes generations in `generation.json`, and raises
`ReviewRequired`. No incomplete score report can select a checkpoint.

A reviewer assesses each queued answer against its documents and rubric, then
writes `adjudications.jsonl` in that evaluation directory, with one line per key:

```json
{"review_key":"key-from-queue","reviewer":"independent-reviewer-id","reason":"Explain the source-supported assessment.","decision_correct":true,"facts_correct":true,"abstention_correct":true,"unsupported_claims":false}
```

All four Boolean dimensions and a reviewer/reason are required. Duplicate or stale
keys are rejected. A passing judgment requires correct decision, facts and
abstention, with no unsupported claims. Resume the same loop: saved predictions
and completed round training are reused. Keys bind the full benchmark/rubric,
prediction and scoring version; cached generations also bind the checkpoint.
`adjudications_hash` in the score report records the review inputs used. Grading
basis counts distinguish reference agreement from human judgments; dimension
rates are reported for adjudicated answers with an explicit denominator.

This workflow supports semantic review without pretending that lexical similarity,
a synonym list, or the teacher's self-approval establishes factual correctness.
It requires actual independent reviewer work for unfamiliar reasoning answers.

## Selection and release

Validation alone controls feedback and checkpoint selection. By default a task
may not regress relative to the initial adapter; tolerances can be explicitly
configured in `quality.per_task_max_regression`. Final-test labels never influence
feedback, ranking, or early stopping.

After selection, the initial and selected adapters run on the frozen test set.
Promotion requires configured quality thresholds, no regression beyond tolerance,
and at least 0.01 macro improvement. `promotion` records the result;
`release_checkpoint` points to the selected adapter only if it passes, otherwise to
the initial adapter. The runtime restores the release checkpoint. Test results do
not trigger another tuning round. These are observed-score gates, not statistical
proof of generalization; inspect errors and sampling uncertainty before deployment.

Use a new output directory after changing scoring, schemas, templates, or quality
settings. Frozen split hashes and round input hashes are checked on resume.

## Deterministic training coverage

The default curriculum reserves 25% of its cumulative target for executable
fictional-rule templates: ledger subtraction (including exact settlement, zero and credit balances),
calendar-date counting (including leap and year boundaries), and Boolean
AND/OR criteria. Python computes references independently of the teacher. Exact
answer checks pin their computed conclusions, missing-information/review flags,
and task-specific fields. These narrow synthetic skills are **not reviewed clinical
or real-world benchmarks** and are not sent to the teacher for self-approval.

The other 75% remains automatically checked teacher silver. A round must achieve
its full new-example target and at least 25% deterministic check coverage by
default. `training_coverage.json` reports achieved counts, shortfall, task mix,
checks by task, template counts, and distinct template reasoning patterns.
Changing amounts or dates does not establish a new reasoning skill. Tune the
mixture only after examining task errors; these templates cover three tasks, not
all 18 capabilities. Increasing epochs cannot replace missing diverse source data.

Optional independently authored `record.answer_checks` have `path` (nonempty keys
or nonnegative array indices), `operation`, and nonempty `operands`. Operations:
`equals` (one exact typed operand), `sum`, `subtract`, `all`, `any`, `date_offset`
(ISO start date and integer calendar-day offset). They stay outside target payloads.
Rules must be authored from trusted source facts and explicit applicability rules;
this DSL does not interpret natural-language policies or business-day calendars.
Absent assertions mean no deterministic verification. All synthetic data remains
silver even when arithmetic checks pass.

## Smoke runs

Use `smoke_configs` for a bounded GPU integration run. It explicitly selects
`synthetic_smoke`, one round, and two training steps; synthetic results cannot
promote a release adapter. The ordinary smoke evaluator still uses native
Lighteval. `tests/test_quality_hardening.py` also exercises the real controller,
Pipeline, reviewed evaluator, curriculum, checkpoint selection and resume on CPU,
with only teacher/model boundaries replaced by fixtures. This test does not verify
CUDA execution, actual weight learning, or actual model quality.
