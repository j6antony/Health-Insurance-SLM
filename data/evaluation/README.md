# Reviewed evaluation inputs

The default loop requires `validation.jsonl` and `test.jsonl`, each with at least
30 independently reviewed examples for every task (540 per split for 18 tasks).
These files are deliberately not populated with teacher-generated “gold”.

Each JSONL line has `record`, `reference`, and
`review: {"status": "approved", "reviewer": "reviewer-id", "source_group": "case-family-id"}`.
Use the existing scenario and task answer schemas. Reviewers must verify the
answer against applicable source rules; metadata is an attestation, not proof.
Include realistic, appropriately deidentified cases with missing documentation,
conflicts, AND/OR rules, date and numeric boundaries, and abstention. Keep all
variants of one case/source family in one split. The loader rejects repeated IDs,
near-duplicate cases, shared source groups, and identical source documents across
splits. Training generation excludes both splits through the existing deduplicator.
This is a useful overlap guard, not a guarantee against semantic leakage.

Validation selects the best checkpoint. Test labels never influence feedback,
checkpoint ranking, or early stopping. Only after selection, the initial adapter
and selected adapter are evaluated on test; `final_comparison` records task deltas
and regressions. Freeze the split before the experiment; do not repeatedly tune
against the final report. Resume checks both split hashes.

Free-text scoring now requires normalized reference agreement, including issue
and missing-information lists. This conservative metric rejects paraphrases;
it does not establish clinical correctness. Use expert error review before making
performance claims. ROUGE is no longer a primary correctness metric.

For an integration run only, set `evaluation.mode` to `synthetic_smoke` and
`development_examples_per_task` to 3, using a new output directory. Smoke thresholds
are engineering checks, never evidence of real-world model quality.

Optional independently authored `record.answer_checks` entries have `path` (list
of answer keys/indices), `operation`, and `operands`. Operations: `equals` (one
operand), `sum`, `subtract`, `all`, `any`, `date_offset` (ISO start date, calendar-day
offset). Example: `{"path":["requires_review"],"operation":"any","operands":[false,true]}`.
These checks remain outside target_payload and run on teacher and student answers.
They must be authored from independently verified source facts. They do not parse
natural language or infer policy applicability. Existing unannotated synthetic
samples still rely on the teacher audit and remain silver. No independent reviewer
or clinical ground truth is supplied automatically by this code.

Inspect each round's `training_coverage.json` before increasing volume or epochs.
The default generation budget is unchanged: increase high-quality distinct data
only after auditing coverage and baseline-versus-adapter errors.
