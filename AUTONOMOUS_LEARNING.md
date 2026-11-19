# Rule-grounded autonomous learning

This additive research pipeline generates **fictional synthetic document-verification** data. It never determines coverage, medical necessity, approval, denial, or payment. Labels are synthetic silver, not expert gold. Existing self-instruct commands, notebooks, API accounting, and backup integrations remain available.

The source repository stays on the Mac. Runtime artifacts default to `/Volumes/Extreme SSD/Codex Distillation Loop`. The directory must already exist; there is no fallback. Override with an explicitly created directory via `SLM_ARTIFACT_ROOT` or `--root`. Python 3.10+ and the standard library are sufficient for generation, tests, packaging and scoring saved predictions.

## Local commands

Run from the repository root:

```bash
python3 -m autonomous_learning storage-check
python3 -m autonomous_learning init
python3 -m autonomous_learning dry-run --run-id local-dry-001
python3 -m autonomous_learning status --run-id local-dry-001
python3 -m autonomous_learning resume --run-id local-dry-001
python3 -m autonomous_learning report --run-id local-dry-001
```

`dry-run` actually generates states, renders and verifies documents, expands variants, projects tasks, applies QA, exports a training dataset, and creates a transfer package. TRAIN is explicitly recorded as **not executed**; evaluation is **unavailable** and promotion **not eligible**. A completed dry run is not a model-quality result.

Standalone commands print artifact receipts. Substitute the real paths printed by the previous command:

```bash
python3 -m autonomous_learning generate --count 24 --seed 42
python3 -m autonomous_learning verify --input "$MASTER_JSON"
python3 -m autonomous_learning expand --input "$MASTER_JSON" --operation remove_signature
python3 -m autonomous_learning build --input "$VERIFIED_OR_EXPANDED_JSON"
python3 -m autonomous_learning stats --input "$DATASET_MANIFEST"
python3 -m autonomous_learning prepare-training --input "$DATASET_MANIFEST"
```

Other interventions: `remove_document`, `restore_document`, `make_unreadable`, `repair_readability`, `introduce_date_mismatch`, `fix_dates`, `remove_field`, `satisfy_condition`, `make_valid`. Every intervention recomputes rule findings and repeats rendering verification.

For isolated testing:

```bash
mkdir -p /tmp/health-slm-artifacts
SLM_ARTIFACT_ROOT=/tmp/health-slm-artifacts python3 -m autonomous_learning dry-run --run-id isolated
python3 -m unittest discover -s tests -v
```

This is an explicit override, never an automatic response to an absent SSD. `.env.autonomous.example` is a reference; export variables yourself. `SLM_STORAGE_BACKEND` supports only `local`, `SLM_GPU_BACKEND` only `manual`. Backup automation is disabled in this controller; the existing `distillation_loop.drive_backup` remains available separately.

## Architecture and contracts

The `autonomous_learning` package contains:

| Module | Responsibility |
| --- | --- |
| `storage.py` | Volume/free-space checks, root containment, writer lock, atomic writes/copies, fsync, checksums |
| `rules.py` | Versioned fictional checklists for hospitalization, outpatient, diagnostic and pharmacy claims |
| `scenarios.py` | Seeded feasible-state generation, pairwise coverage, stable families and counterfactual lineage |
| `rendering.py` | Three text layouts, field aliases/date formats, final-text parsing and exact evidence offsets |
| `tasks.py` | Seven new task schemas and strictly separated student payloads and targets |
| `dataset.py` | Independent target reconstruction, scope/schema/evidence checks, deduplication, family caps, replay, leakage exclusion, exports |
| `teacher.py` | Selective existing OpenAI integration, persistent queue/cache, budget capture, bounded threads, quarantine, offline Batch exchange |
| `planner.py` | Strict JSON plan validation, bounded focus changes, aggregate feedback and a heuristic baseline |
| `jobs.py` / `gpu_worker.py` | Portable training packages, verified imports, QLoRA worker and tokenizer preflight |
| `evaluation.py` | Benchmark imports, saved-prediction scoring, optional LightEval metric, paired comparison and best-pointer promotion |
| `experiments.py` | Validation of matched comparison-arm manifests |
| `orchestrator.py` | Bounded SQLite state machine and manual external-artifact waits |
| `__main__.py` | CLI |

Task contract `document-verification-v1` deliberately differs from incompatible legacy schemas. The tasks are classification, field extraction, missing documents, evidence quality, conditional requirements, cross-document consistency and short finding explanations. The student sees supplied documents, inventory, claim facts, fictional rules, instruction and response schema. It never sees scenario lineage, findings, generation seed, verification status or hidden targets in its prompt.

The rule pack defines presence, readability, required patient/date fields, signatures, conditional referral applicability, patient-ID mismatches and service-date consistency. Unknown applicability/signatures and unreadable documents trigger review; missing and unreadable evidence remain distinct. Each finding records predicates, relevant facts, evidence references and the pack version. It is a narrow document checklist, not an insurer policy.

The generator guarantees an initial valid control and coverage of all four categories for plans of at least four masters. Pairwise coverage is measured over feasible category/condition/case/layout dimensions. It is not exhaustive clinical coverage. Family caps apply jointly to new rows and replay. Continued adapters use replay stratified by task at a fixed 25% target; bootstrap or limited pools can produce an explicit replay shortfall. Fresh adapters train on the cumulative approved corpus, with duplicates and family caps applied, rather than a replay subsample. Inspect the replay policy and shortfall before training.

## Persistence, interruptions and storage

`state/orchestration.sqlite` stores stage input/output hashes, status, attempts, safe error type, runtime and API/GPU cost fields. Full typed output contracts live in checksummed stage artifacts. Completed stages are verified before reuse. Source/config/planner changes reject stale resume; use a new run ID after changing implementation or configuration.

The controller executes PLAN → GENERATE_STATES → RENDER → VERIFY_RENDERING → EXPAND → PROJECT_TASKS → VALIDATE_DATASET → EXPORT → PREPARE_TRAINING → TRAIN → EVALUATE → PROMOTE_OR_REJECT → ANALYZE_FAILURES → NEXT_CYCLE.

Limits in `config/autonomous_learning.json` bound cycles, wall time (including manual external waits), consecutive non-improvement, calls and spending. Prepared GPU limits deduct already imported worker cost from the run allowance. Pause/cancel/wall-time checks apply between major stages, not instantaneous preemption of every local operation. An SSD loss or changed volume identity fails closed; reconnect the same drive and resume. No earlier datasets or adapters are automatically deleted. A crash after a completed stage write but before its SQLite commit reexecutes an idempotent operation. A crash after best-pointer promotion is also recoverable.

```bash
python3 -m autonomous_learning start --run-id experiment-001 --config "$RUN_CONFIG" --plan "$CURRICULUM_JSON"
python3 -m autonomous_learning pause --run-id experiment-001
python3 -m autonomous_learning resume --run-id experiment-001 --config "$RUN_CONFIG"
python3 -m autonomous_learning cancel --run-id experiment-001
```

A cancelled run is terminal. A manual-training wait is expected, not a successful training stage. The operator resumes after importing verified external results. SQLite control requests can be written while the worker holds its artifact lock. An incomplete result-copy directory is retained for inspection, not silently removed.

All writers should use the provided CLI/lock. POSIX advisory locks cannot prevent an unrelated program from ignoring them. Directory fsync is attempted; unsupported filesystem errors are handled explicitly. Moving to another disk/device requires an explicit reviewed migration, not editing the storage identity silently.

## Teacher rendering and spending

Paid generation defaults to disabled and zero dollars. The queue reuses `OpenAIParent`, `APIBudget`, `UsageLogger` and the optional Supabase ledger. Transport retries include provider rate-limit delays and jitter; every request attempt is budgeted before sending. Request identity includes normalized synthetic input, model/settings, response contract and renderer version. A queue item interrupted in flight becomes `uncertain` and is never automatically resubmitted. Reconcile the provider result before any manual retry.

The accepted text grammar permits heading-preserving, field-preserving line variations. **Unrestricted clinical prose is not currently accepted.** This conservative gate checks final text independently; it does not accept a document merely because metadata agrees. A validated semantic audit pathway is still needed before allowing rich narrative variations. Failed teacher samples are quarantined; valid peers continue. The hybrid controller can retain a verified deterministic version when a teacher variation fails; the queue retains that rejection.

To enable paid rendering later, set positive API limits and `teacher_enabled: true` in a separate operator-reviewed config; use `--authorize-api` on `start`/`resume`. The plan must remain within those limits. For a standalone selective queue run:

```bash
python3 -m autonomous_learning render-api --input "$MASTER_JSON" --config "$API_ENABLED_CONFIG" --authorize-api
python3 -m autonomous_learning batch-export --output "$BATCH_REQUESTS_JSONL"
python3 -m autonomous_learning batch-import --input "$COMPLETED_BATCH_RESPONSES_JSONL"
```

Batch export/import verifies queued IDs and final facts, but **does not submit, poll or budget an external Batch job**. External submission and Batch billing reconciliation remain operator responsibilities. Exporting Batch requests does not authorize submitting them. The synchronous budget is cumulative across this artifact root and cannot be increased implicitly. Provider-side exactly-once delivery is not promised after a network ambiguity.

## GPU handoff

Preparation requires no CUDA, API key or model download. A package contains exact dataset files/hashes, source hashes, the pinned training requirements, model name/revision, LoRA configuration, seed, output location, adapter lineage and a job ID. Continued adapters must match base model and LoRA rank; restore performs full weight compatibility checks. Fresh training initializes one adapter on the base; it does not stack LoRAs.

After explicit authorization for training:

1. Choose an operator-approved positive GPU limit in the plan/config **before** preparing the job. Default jobs have a zero GPU limit and cannot train.
2. Copy `package.zip` to a separately provisioned CUDA machine. Verify its outer SHA-256 against the local package receipt before extraction. Extract into a new directory and install its `requirements-training.txt` against a compatible CUDA torch installation.
3. From the extracted directory, validate tokenizer lengths without loading model weights:

```bash
python -m autonomous_learning.gpu_worker . --validate-only
```

4. Only after authorizing the actual spend, run the real worker, supplying approved numeric budget and hourly rate values:

```bash
python -m autonomous_learning.gpu_worker . --authorize-training --max-gpu-usd "$GPU_BUDGET" --gpu-hourly-rate-usd "$GPU_HOURLY_RATE"
```

The runtime watchdog derives a time allowance from the supplied budget/rate and refuses to mark an over-budget worker complete. Its cost is an estimate for this worker, not provider billing or idle VM time. You must stop paid infrastructure yourself. CUDA calls can delay Python signal handling; this is not a provider-enforced spending cap.

5. Copy the entire `result/` directory back to a chosen local transfer location. Import it:

```bash
python3 -m autonomous_learning import-results --input "$RETURNED_RESULT_DIRECTORY" --job "$PACKAGE_RECEIPT_JSON"
python3 -m autonomous_learning resume --run-id experiment-001 --config "$RUN_CONFIG"
```

Imports bind job, dataset, adapter hashes, remote reload/inference verification, versions and bounded GPU cost. They do not execute remote files. Remote statements are trusted worker attestations, not cryptographically signed third-party proof. Re-import is idempotent. Unexpected or changed results are rejected and never replace a best adapter.

The worker reuses `StudentRuntime`, non-thinking Qwen prompts, `tokenize_teacher_pair`, completion-only loss and `train_tokenized`/SFTTrainer. It restores the produced adapter from disk and performs a nonempty inference check before emitting `result.json`. This smoke check is not a quality score. Current model revision `main` is resolved/recorded at load time; pin an immutable revision before a publishable controlled experiment. GPU execution and new-contract tokenizer lengths have not been live-tested here.

## Evaluation and promotion

No independent evaluation corpus is fabricated. Import a JSON object with exactly `version: 1`, `tier`, `process`, `independently_reviewed` and `cases`. Tier is `development`, `probes`, or `release_holdout`; process is `in_generator` or `independent`. Each case contains `id`, `family_id`, `template_family`, `contract_version`, `task`, `payload` and `target` using the schemas in `tasks.py`. Source provenance must be accurate. Independent examples can use different prose/layouts; they need not use the renderer grammar. Human reference correctness remains the independent author's responsibility.

```bash
python3 -m autonomous_learning import-benchmark --input "$BENCHMARK_IMPORT_JSON"
python3 -m autonomous_learning evaluate --benchmark "$IMPORTED_BENCHMARK_JSON" --checkpoint "$ADAPTER_DIRECTORY" --input "$GENERATIONS_JSON"
```

Saved generations require `benchmark_id` (canonical benchmark hash), `adapter_sha256` (canonical hash of `adapter_files()`), `contract_version`, `model_name`, `inference_config`, and `predictions` keyed by every case ID. Each prediction contains `raw` JSON text and `truncated`. `evaluation.evaluate_runtime(runtime, benchmark, checkpoint)` produces this contract using a restored, already loaded runtime. `make_lighteval_metric()` supplies a native optional LightEval metric for these new tasks. Existing legacy evaluation remains unchanged.

Exact structured scoring is deliberately conservative. It reports task macro scores, schema/evidence validity, missing-document recall, inconsistency detection, review-flag correctness, critical errors, process-specific accuracy and source-family clustered paired intervals. Arbitrary paraphrases of explanations may need reviewed alternative references; no semantic-equivalence score is invented.

Probes require independent authorship. A release holdout additionally requires independent review and `--authorize-release-evaluation`. Access is logged and limited to two distinct adapters per benchmark, permitting one paired comparison. One-way sealed exclusion fingerprints protect later datasets without forwarding holdout contents to the planner.

When the controller waits at EVALUATE, produce baseline/candidate reports on identical cases with identical inference settings, and place an object with `baseline` and `candidate` reports at `experiments/RUN/cycle_NNN/comparison_input.json`. Resume. Promotion reuses existing operational/regression gates and paired intervals, verifies checkpoint identity, requires all seven tasks with at least two cases each, quality floors, no observed critical errors and no evidence decline. It atomically updates a provisional-best pointer while retaining previous metadata and all adapter artifacts. Synthetic scores alone never establish release readiness.

## Curriculum and controlled comparisons

```bash
python3 -m autonomous_learning planner-input --output "$PLANNER_INPUT_JSON"
python3 -m autonomous_learning compare --input "$COMPARISON_SPEC_JSON"
```

`planner-input` emits the empty baseline interface. In real runs, `ANALYZE_FAILURES.json` contains allowed aggregate feedback, coverage/statistics and quarantine counts. Use that file to author a proposed plan; submit it with `start --plan`. The strict plan keys and bounds are defined in `planner.py`. The controller never requires this interactive Codex session to remain open. A separate unattended Codex planner client is not deployed; it would require credentials, billing and an independently configured service. Do not pass holdout case contents to a planner.

A comparison spec has `mode` (`equal_training_budget` or `equal_api_cost`) and `arms` with A/direct_teacher and B/scenario_reuse. Required fields and exact matching checks are in `experiments.py`. Equal-training mode requires matched accepted count and measured token exposure; both modes require matching starting adapter, training configuration and benchmark. Equal-cost mode requires equal API ceilings. The validator records measured generation time, cost and family diversity supplied by each arm; it does not claim savings or execute the arms.

**Automated execution of the legacy teacher arm, fair token-matching/subsampling, planner A/B runs and adapter-strategy experiments remain deferred.** Existing direct self-instruct generation has 18 legacy tasks and cannot be silently treated as these seven new task contracts. A reviewed common-task adapter is required for a valid head-to-head experiment.
