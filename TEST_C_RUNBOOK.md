# Test C — paired seven-task development evaluation

Test C is inference only. It never trains, repairs predictions, changes frozen targets, or promotes an adapter. The user separately approved one Colab NVIDIA L4, a maximum US$1 worker credit valuation, and a 3,600-second worker watchdog, followed by verified result transfer and GPU release. Setup/transfer/idle time remains outside the worker watchdog and must be managed by the operator.

Artifacts live under `/Volumes/Extreme SSD/Codex Distillation Loop/experiments/test-c-baseline-comparison/`.

## Frozen benchmark and scoring

Canonical benchmark ID: `417e71e939eef930474f7233f0f32c1ff075120b4ffd6384e62d9c444e81e055`. The immutable `benchmark.json` and `benchmark-manifest.json` were written before inference. There are 28 distinct source families, four cases for each of seven tasks, all four claim categories within every task, and four examples of each selected scenario state overall. Existing references are reconstructed from final visible text, rule facts and inventories.

All 28 references passed actual Qwen tokenizer and TRL collator checks. Maximum full reference sequence: 2,263 tokens; overlength: zero; non-thinking template and completion-only masks passed. Inference separately checks each prompt plus the 1,536-token generation budget fits the 4,096-token context.

Protocol `in-generator-family-content-separated-v1` explicitly permits the shared `closed-grammar-v1` template family **only for unreviewed in-generator development**. It retains source-family, case-ID and normalized content separation, with the existing 0.98 SequenceMatcher threshold. It excludes 1,386 available training/replay rows across nine content-bound sources (32 distinct families). The legacy adapter's complete historical training manifest is unavailable; separation from unknown historical corpora cannot be certified. Same-generator cases are not independently authored transfer probes.

Version-1 benchmark imports retain their strict template exclusion. The version-2 exception rejects probes, release holdouts, independent authorship labels and independent-review claims. Future controller training excludes development families/content without banning every shared generator template. Independent and sealed exclusion checks remain unchanged.

Scoring reuses `document-verification-grounded-v3` without changing its strict canonical-result/evidence criteria. Strict task correctness includes canonical reason wording and result order. Auxiliary rule/status or structured-result correctness excludes reason wording and order and is reported separately. No raw response is repaired or normalized for strict scoring.

Evidence offset/source-reference validity, strict proof relevance/sufficiency, findings, review flags, truncation, failures and per-task/macro scores are separate diagnostics. Semantic relevance/sufficiency are reported with their assessable denominator; malformed responses are not mistaken for semantically inspected text. Full free-text unsupported-assertion checking is unavailable; deterministic structured-result extras are a separately labeled metric. Existing v3 critical-error counts are a proxy for schema-valid wrong canonical results or missed review, with malformed answers counted separately.

Paired comparison uses the same frozen cases and inference configuration. It records strict gains/losses/ties, task deltas and the existing family-clustered/task-stratified bootstrap plus conservative bound. Four cases per task, shared generator structure, potential floor effects and incomplete historical lineage limit confidence. No scores are directly compared with the legacy 79.5% or 0.558 contracts.

## Immutable checkpoints and execution

Baseline: `/Volumes/Extreme SSD/Codex Distillation Loop/checkpoints/candidates/e8e5edc7-drive-transfer/adapter`.

Candidate: `/Volumes/Extreme SSD/Codex Distillation Loop/gpu_jobs/imported_results/c03c14696b8ab4b394266f83d850641ae3e71f9eaa3cc749a34de12489834fc5/adapter`.

Both adapters are rank-32 RSLoRA, alpha 32, dropout zero. Exact config/weights hashes and the full source checksums are in `package-receipt.json`. Adapter configs omit a saved base revision, so the job and loaded model identity explicitly pin `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`.

Prepare or resume the existing immutable benchmark/package:

```bash
python3 -m autonomous_learning.paired_evaluation prepare \
  --root '/Volumes/Extreme SSD/Codex Distillation Loop'
```

The first transfer package is `package-cdcfc799fed5.zip`, SHA-256 `cbe5a01c9f27063d09d8ebc8b584989634dff1d76a827960d67baae203f04ff6`. The notebook is `notebooks/test_c_paired_evaluation.ipynb`, with an SSD copy alongside the package. It installs the Test B-compatible pinned dependencies in an isolated Python 3.10–3.12 environment, using `venv --without-pip` and host pip's `--python` option when system ensurepip is absent.

Run package/tokenizer preflight from the extracted package on the GPU machine:

```bash
python -m autonomous_learning.evaluation_worker . --validate-only
```

Then execute only the separately approved inference job:

```bash
python -m autonomous_learning.evaluation_worker . \
  --authorize-inference --gpu-hourly-rate-usd "$APPROVED_CREDIT_VALUATION_USD_PER_HOUR" \
  --max-gpu-usd 1 --max-seconds 3600
```

Both arms use the same pinned tokenizer/chat template, thinking disabled, context 4,096, generation cap 1,536, greedy decoding, seed 43 and batch size one. One quantized base remains loaded; the runtime clears GPU caches and restores the complete verified adapter state before each arm. It never merges weights or invokes a training function. A failed single-case OOM stops rather than changing inference settings. Responses are durably written after every case. Resume validates identities and skips completed case IDs; the cumulative runtime budget cannot be reset by a normal resume.

The return archive contains both raw prediction files, per-case/per-task metrics, paired comparison, failure analysis, at least seven representative comparisons, aggregate planner feedback, budget/runtime records, source/benchmark bindings and a per-file return checksum manifest. Verify the download SHA-256 and every return-manifest entry, then re-score locally against the original SSD checkpoints. Only after the results are safely preserved should the Colab runtime be disconnected and deleted.

The planner feedback is aggregate only, explicitly development labeled, and severity/frequency weighted. Its suggested curriculum has zero spending authorization and is never submitted to a training cycle. Sealed holdout contents never enter this feedback.
