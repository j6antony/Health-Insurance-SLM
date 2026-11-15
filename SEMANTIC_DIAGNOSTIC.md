# Saved checkpoint semantic diagnostic

Use `semantic_diagnostic_colab.ipynb` in a CPU Colab runtime. Upload the matching
`colab_distillation_bundle.zip` from the repository if prompted, mount Drive, and run the
first code cell. It checks the split, all three Pass A prediction files, their
generation artifacts and the original adapter hashes. It produces
`cost_preflight.json` without contacting the judge. The second code cell takes
the approved preflight hash and cumulative USD ceiling before loading a key
and submitting missing judgments. Run only one Colab VM per output directory.
Use the same output directory to resume. Re-run the preflight to get a fresh
hash and updated remaining cost before resuming an interrupted paid run.

## Required and optional files

| Input | Expected files |
| --- | --- |
| Saved checkpoint comparison | `manifest.json`, `A_initial_strict-reference.jsonl`, `A_sft_strict-reference.jsonl`, `A_sft_orpo_strict-reference.jsonl`, and the corresponding `A_initial_generation.json`, `A_sft_generation.json`, `A_sft_orpo_generation.json` |
| Original benchmark | `evaluation_split.json` from the original run and the matching benchmark corpus. The default corpus and task schemas are bundled. External copies can be passed via `--corpus` and `--schemas`. |
| Optional independent adapter verification | `initial_adapter/`, `round_01/sft/adapter/`, and `round_01/preference/adapter/`, each with `adapter_config.json` and `adapter_model.safetensors`. Pass their common root via `--checkpoint-root`. |
| Optional historical judgment reuse | Original `baseline/` and `round_01/lighteval/`, each containing `scores.json`, `predictions.jsonl`, and `generation.json`. Pass their common root via `--historical-run`. Pass compatible original `benchmark_judging/` directories via repeatable `--legacy-cache` flags. |

The historical judgment journals must use the same rubric and judge settings.
The diagnostic compares raw generations and validates every reused judgment.
It recomputes semantic scores from the raw outputs and rubric dimensions.
The strict-reference score fields carry their original meaning.

## Local CPU workflow

Run the command from the repository root with the saved files in the indicated directories.

```sh
python3 -m distillation_loop.semantic_eval \
  --comparison-dir /path/to/saved-comparison \
  --split /path/to/original-run/evaluation_split.json \
  --historical-run /path/to/original-run \
  --output /path/to/semantic-diagnostic
```

With explicit approval, repeat the command with
`--approve-preflight <approval_hash> --max-usd <approved_cumulative_cap>`.
Set `OPENAI_API_KEY` through the environment. The ceiling uses the repository's
Standard model pricing, the exact missing request bodies, maximum output
tokens and all configured transport retry attempts. The journal reserves each
attempt before submission and retains the reservation for unknown usage.

An actual preflight on the saved three-way comparison reused 573 unique
judgments. Another 199 checkpoint-case rows were scored locally. The remaining
286 unique judgments were all for SFT-only. At four attempts each the cumulative
ceiling was USD 8.236834500. Colab recomputes the exact number from its caches.

## Outputs for the continuation decision

Return `semantic_comparison.json`, `cost_preflight.json`, and the three
`*_semantic_cases.jsonl` files. Include an explicit continuation decision.
The report records paired SFT versus initial task-score, validity, and review
flag changes, task-specific regressions, and judge diagnostics. If the result
is equal, worse, or inconclusive, investigate training targets, student
screening, task labels, benchmark and judge agreement, training and held-out
loss, and the existing SFT optimizer, learning rate, and completion mask
before changing the training setup.

For independent calibration, provide a development-only JSONL of labels
and run `python3 -m distillation_loop.calibration --report <semantic_comparison.json>
--labels <labels.jsonl> --output <calibration.json>`. Each label binds
`checkpoint`, `scenario_id`, `comparison_identity_sha256`, and
`prediction_sha256` to a named `reviewer`, the four Boolean rubric dimensions
under `judgment`, and Boolean `review_flag_correct` and `schema_evidence_valid`
decisions. The report includes coverage, agreement, and case-level discrepancies.

## Held-out diagnostic and release controls

Supply the original `round_01/training_plan.json` with all 200 pairs.
`python3 -m distillation_loop.leakage --training-plan <training_plan.json>
--split <evaluation_split.json> --output <diagnostic-folder> --candidates <verified.jsonl>`
checks the plan and freezes a diagnostic split before training. Candidate rows
include `record`, `answer`, `scenario_hash`, `verification_status` with value
`automatically_checked_silver`, and the passing `audit`. Use independently
sourced candidates with no prior training or student-conditioned exposure.
The split is bound to the history, benchmark, and candidate identities. A
shortfall is recorded when fewer than 50 independent pairs qualify. The
revised training controller can read this candidate JSONL via
`teacher_loss_candidates` and saves completion-token-weighted loss before
and after SFT. It reports the task composition and sequence-length exclusions.

The fixed release split is separate from development selection and is
evaluated only by an explicit controlled release request after a provisional
checkpoint exists. It uses a shared reservation beside the source split and
stores outputs separately. Synthetic holdout evidence leaves
`release_checkpoint` unset. At six cases per task, one change can move a task
score by 16.7 points. An overall one-point change is also weak evidence with
108 cases. The release report uses paired intervals clustered by source
family. Stronger claims require a larger independently validated or
human-reviewed benchmark.
