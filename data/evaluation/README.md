# Standalone inference review

`inference_review_v1.jsonl` contains 100 synthetic cases across all 18 tasks,
with five or six cases per task. It includes 18 newly authored cases and 82 cases
from the current fixed holdout. Older adapters may have used those holdout
families for development. Keep this file for evaluation.

In a Kaggle session with GPU and Internet enabled, run from the extracted
project directory. Use the Python environment with GPU PyTorch and install the
inference dependencies:

```sh
python -m pip install transformers==4.57.3 peft==0.18.0 accelerate==1.11.0 bitsandbytes==0.48.2
CUDA_VISIBLE_DEVICES=0 python -m distillation_loop.inference_review \
  --output /kaggle/working/inference_review_predictions.jsonl
```

The runner fetches the newest compatible adapter from
`j6antony/Health-Insurance-SLM` GitHub Releases, including prereleases. It
verifies the downloaded ZIP's SHA-256 and records the release tag and hash in
every result. For a private repository, pass a `GITHUB_TOKEN` with Contents
read access through the environment using Kaggle Secrets. Use `--adapter`
with a local adapter directory or `--base` to explicitly select other weights.
The command runs without a teacher API key.

For the Colab notebook's Unsloth generation backend, install
`requirements-colab.txt` with the existing CUDA PyTorch version constrained,
then add `--unsloth`. The runner restores the adapter with its saved LoRA rank,
alpha, and dropout. Results record the generation backend. Use a fresh output
filename for a first Unsloth run.

To publish the JSONL as an asset on the adapter's GitHub release, add
`--upload-to-github` and use a `GITHUB_TOKEN` with Contents write access to
`j6antony/Health-Insurance-SLM`. The predictions inherit the release's
visibility. The command prints the verified asset URL. If the upload fails
after inference finishes, rerun the same command with the same output path
to upload the saved file.

Each result includes the case ID, task, full model input, model and adapter
identity, and the raw completion with any truncation or context error.
Results are saved after each case. Use a new output filename for a new
inference run. Share the prediction JSONL or its release asset for individual
assessment.

# Bundled fixed evaluation: default full-run workflow

The saved dataset is `fixed_validation_v1.jsonl`, with its companion
`fixed_validation_v1.manifest.json`. It contains **540 cases: 30 for each of 18
tasks**, arranged as 10 task-specific families and 3 challenge variants. Variants
cover the base task, an unrelated conflicting case that must not be merged, and
untrusted document-embedded instructions. Inputs are unique; shared families are
intentional. This is a synthetic regression suite, not 540 independent real-world
patients or an expert-reviewed medical benchmark.

`evaluation.mode = "fixed"` is the default. The loader checks the dataset version,
content hash, schema hash, all task counts, reference schema/evidence, provenance,
and unique IDs/inputs. It reads the file directly; it does not regenerate it or
make parent API calls. A missing, altered or incompatible file is an error with
**no fallback to paid generation**. Both files are explicitly included by Git's
ignore rules so they travel with the repository. Sync the updated repository into
Kaggle before running the notebook; merely updating the notebook is insufficient.

The full notebook defaults to `TEST_RUN = False` and the
`distillation_loop_adaptive_v1` run directory. A seeded split reserves two source
families per task for the final holdout. All three variants of each reserved family
stay together. The loop uses 432 development cases for checkpoint selection and
108 final cases to compare the initial and selected adapters. It excludes the
entire corpus from training generation through local duplicate checks and saves
the chosen case IDs, family IDs and hashes in `evaluation_split.json`.

The 100-case Colab and Kaggle profiles sample development cases after reserving
the same final holdout. Exact reference-based prose scoring is conservative and
can score valid paraphrases low. Synthetic evaluation supports experimental
checkpoint selection. Promotion requires independent reviewed evidence.

Smoke runs deterministically select one saved example per task (18 total), also
without paid evaluation-data generation. `automatic` and `synthetic_smoke` modes
remain available explicitly for legacy/generated-development workflows. Select
`reviewed` only to use actual independently reviewed benchmark inputs.

## Reproducibility and limits

`python3 -m distillation_loop.build_fixed_evaluation` is a maintainer command that
reproduces the checked-in dataset locally, without API access. It is **not** called
by training. Treat a released version as immutable; changes require a new dataset
version and a new run folder. The manifest describes the corpus and provenance.

The source records use explicitly fictional/nonbillable catalogs, exercise
policies and handling rules. They cover source grounding, negation, missing and
unreadable values, conflicting sources, date applicability, AND/OR, exceptions,
calculation boundaries, attribution, planned versus completed actions, scope and
injection. No real code meanings, medical guidelines or legal obligations are
asserted as authoritative. Mechanical checks and authored references do not
establish real-world generalization or eliminate semantic leakage. The bundled
holdout is isolated within a fresh run. Earlier versions used the entire corpus
for development, so an adapter selected by an earlier run may already reflect
performance on the reserved families.

Run `python3 -m distillation_loop.preflight --skip-runtime` to confirm that the
432 development cases and 108 holdout cases load with `generated_validation_target: 0` before a Kaggle run.

## Optional independently reviewed mode

The rest of this document describes explicit reviewed mode, not the default.

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
