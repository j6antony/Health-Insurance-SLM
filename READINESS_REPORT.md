# Autonomous learning readiness — November 19, 2026

**Subsequent live-test update:** the original preparation report below is historical. Test A live rendering completed with its results saved under the SSD's `experiments/api-canary-20261119/analysis.md`. Test B continued-adapter GPU training, reload, inference, transfer and import have now passed on a single Colab NVIDIA L4; see [the measured GPU results](GPU_CANARY_TEST_B_RESULTS.md). The GPU runtime was released after verified result transfer. Neither test establishes downstream model-quality improvement or independent-evaluation readiness.

The interrupted work is complete through all no-cost validation and paid-canary preparation. **No paid OpenAI generation, pretrained Qwen3-8B weight loading, or GPU training was performed.** The existing 24-family experiment was not regenerated. Its immutable exported examples were reused for tokenizer/evidence comparison and explicitly revalidated replay.

Work remains on `feature/autonomous-agent-learning`, based on reviewed commit `f176d2752783a0c61bc32f9779d336d3ead8e90a`. No merge or change to `collab` is included. The completion message supplies the new pushed commit SHA; `git log -1 --oneline` identifies the commit containing this report.

## Readiness answers

| Question | Answer and verified boundary |
| --- | --- |
| 1. Is deterministic generation functioning correctly? | Yes for the fictional checklist: valid controls, all four claim categories, feasible states, explicit intervention targets and recomputed findings pass tests and the SSD pilot. |
| 2. Is the data diverse enough for a controlled pilot? | Yes for an engineering pilot: eight synthetic families, 31 represented versions, seven tasks, five observed layout IDs and positive/negative/uncertain cases. This is not evidence of broad clinical or insurer generalization. |
| 3. Are evidence targets minimal and relevant? | Task-specific canonical proofs are independently reconstructed. Readability-driven review cites unreadability; typed inventories support quote-free absence; classification labels are hidden from its inventory. Alternative valid proof styles may be conservatively rejected by exact-anchor evaluation. |
| 4. Does real Qwen tokenizer preflight pass? | Yes for existing paired data, the 64-example pilot, fresh 256-example corpus and continued 85-example dataset. No overlength rows, no truncation; real non-thinking template, completion masks, Dataset, TRL collator and CPU SFTConfig passed. |
| 5. Is the teacher renderer ready for a paid canary? | Prepared and mocked end to end, with hard budget/cache/recovery controls and a novelty test against all nine Python styles. Live service behavior and useful diversity remain unmeasured; explicit approval is still required. |
| 6. Is the GPU worker ready for a live canary? | Packages, dataset lengths, configuration and native CPU RSLoRA scaling are verified. Ready for an approved engineering smoke test; CUDA loading, bitsandbytes memory, training/reload and full-model inference remain untested. |
| 7. What is missing for independent evaluation? | A separately authored/reviewed corpus with accurate source/family/template provenance and reviewed supporting evidence. A complete authoring/import/review kit was prepared; no independent cases or approvals were invented. |
| 8. What must happen before real SFT? | Approve provider, one CUDA GPU, hourly rate, spending ceiling and one strategy; use the verified package; run its tokenizer preflight; then perform the bounded CUDA load/train/reload smoke test. A meaningful quality experiment additionally needs independent evaluation and a baseline on the new contract. |

## Evidence-quality changes and measured savings

Evidence policy is `minimal-task-evidence-v3`; external task JSON shapes remain `document-verification-v1`. The new policy gets separate example identities, preserving old exports. Unreviewed semantic cases receive unavailable macro/per-task/transfer scores, are ineligible for promotion, and do not drive curriculum adaptation. The scorer is versioned `document-verification-grounded-v3` so old/new metrics cannot be silently paired.

- Classification cites identifying headings only.
- Extraction cites the supplied field/value statements and unreadability when it explains review/abstention.
- Contradictions cite two conflicting dates; agreement/insufficiency uses the necessary dates, inventory and quality markers.
- Missing-document tasks receive a verified typed inventory and scoped requirement references. Their negative/absence findings can have zero document quotes. Classification inventories stay untyped, avoiding label leakage.
- Conditional findings cite the actual applicability fact and conditional rule, adding document proof only when applicable.
- Explanations combine the minimal proof of their public findings, including source-text references for genuinely absent fields rather than fabricated quotes.

Every source reference is resolved and every quote is checked against document ID, page and exact character bounds. Offsets alone do not establish grounding. Tests distinguish correct findings with irrelevant/insufficient evidence, wrong findings with valid-looking quotes, correct/incorrect review flags, unsupported assertions, and malformed/incomplete output. Unfamiliar prose without reviewed reference anchors has **semantic grounding unavailable**; no equivalence score is fabricated.

Paired measurements below use the **same 192 pre-existing inputs** and the real pinned Qwen tokenizer. Only target evidence/reference construction changed; no new 24-family generation was run.

| Measurement | Before | After |
| --- | ---: | ---: |
| Evidence spans | 2,994 | 792 |
| Prompt tokens | 168,612 | 168,612 |
| Completion tokens, including EOS | 161,923 | 60,306 |
| Full sequence tokens | 330,535 | 228,918 |
| Longest sequence | 2,598 | 2,163 |
| Overlength examples at 4,096 | 0 | 0 |

Evidence spans decreased **73.5%** and completion-token exposure decreased **62.8%** on these paired targets. This is a measured token reduction, not a model-quality or dollar-savings claim. Legacy untyped inventories still require heading evidence; they are not silently upgraded when reconstructing targets.

## Scenario, selection and rendering fixes

The old first `valid` case could have unknown referral applicability. Valid controls now exclude unresolved applicability; a manually supplied valid/unknown combination is explicitly classified as `unknown_applicability`. The existing report's example also incorrectly called its referral requirement not applicable. The preserved SSD record actually has `referral_required: true`; its saved removal variant correctly removes the referral. The historical report was corrected against those records, not a regenerated example.

Counterfactuals now validate target and comparison document IDs. Defaults name the category's required document rather than taking the last inventory entry. Optional referrals cannot be removed as required evidence unless applicable. A missing document cannot silently become readable while applying an unrelated signature/date intervention. No-op interventions are rejected; expansion searches for distinct meaningful alternatives. A global `make_valid` operation is explicitly a complete repair, not a one-factor counterfactual.

Selection greedily balances task/outcome/intervention/state/category/layout/control coverage under the unchanged family cap, with seeded content-based tie breaking. Near-duplicate screening includes visible claim context and separates different outcomes/review flags; a change in applicability is not discarded just because document text is similar.

The deterministic grammar now supports nine controlled styles: aliases/abbreviations, reversed and sectioned ordering, five date formats, pipe tables, alternate separators, bounded sentence-like field lines, fixed administrative distractors and known label OCR variants. Arbitrary prose, altered values, duplicate/conflicting fields and unsupported statements are rejected. OCR variation is label-level only; there is no claim of robust arbitrary OCR recovery.

## Final bounded SSD pilot

The final run uses an explicitly isolated descendant of the authorized SSD root, keeping the eight-family pilot separate from cumulative canary replay. Earlier iteration artifacts, datasets and checkpoints were retained. Only this final run is counted below; repeated family IDs are not counted as new independent families.

- Artifact root: `/Volumes/Extreme SSD/Codex Distillation Loop/experiments/quality-pilot-20261119-release/artifacts`.
- Run ID: `quality-final`; fixed generator seed 191120 (configured seed 191119 plus cycle 1).
- Eight unique **synthetic** master families; all four categories. These are not independently authored evaluation families.
- 31 total scenario versions: eight masters plus 23 meaningful counterfactuals. Every family has at least two variants. All 31 versions are represented in selected data.
- 217 projected candidates; 64 selected SFT examples; maximum eight per family.
- 153 exclusions: 135 family-cap selections, 18 near duplicates and 0 exact duplicates. Duplicate rate: 8.29% of candidates.
- 236 selected target spans. All JSONL targets were reconstructed and validated against final text; dataset checksums verified.
- Faithfulness failures: 0; target/evidence verification rejection rate: 0.0% for this narrow synthetic pilot.
- Controls: 15 valid-control examples and 49 issue/uncertain examples. All five canonical statuses occur.
- 5 observed layout IDs out of nine supported. Normalized surface forms per document type: `{"claim_form": 5, "discharge_summary": 6, "hospital_bill": 2, "investigation_report": 6, "invoice": 1, "pharmacy_invoice": 2, "prescription": 5, "referral": 3}`. Invoice variation remains sparse; counts are observed, not assumed from feature support.
- Stage runtime: about 0.73 seconds, excluding environment provisioning, tokenizer checks and adapter transfer packaging.
- Resume reexecuted zero completed stages. Paid API calls: zero. GPU training: none.

| Task | Selected examples |
| --- | ---: |
| conditional_requirements | 9 |
| cross_document_consistency | 9 |
| document_classification | 8 |
| evidence_quality | 9 |
| field_extraction | 9 |
| finding_explanation | 11 |
| missing_documents | 9 |

## Real tokenizer and native training-pipeline checks

Model/tokenizer revision: [`b968826d9c46dd6066d109eabc6255188de91218`](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218). Only the public tokenizer was downloaded. No full Qwen model weights were loaded.

| Dataset | Examples | Sequence p50 | Sequence p95 | Max sequence | Max completion | Overlength |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pilot | 64 | 1088 | 1826 | 1919 | 974 | 0 |
| Fresh | 256 | 1070 | 1873 | 2163 | 1068 | 0 |
| Continued | 85 | 1077 | 1866 | 1978 | 998 | 0 |

The context stays **4,096**. Canary inference allows 1,536 output tokens because revalidated replay includes completions above the old 1,024 cap; this does not increase the training context. Prompt, completion, EOS, attention padding and completion-only loss masks were checked using the actual pinned chat template and native TRL collator. GPU-side preflight avoids initializing an Accelerate CPU device state before CUDA training.

Real token totals are stored in separate content-bound receipts; immutable dataset manifests are not rewritten after measurement. Pilot totals are 56,659 prompt + 19,942 completion = 76,601 sequence tokens per one pass. They are not claims about exposure during an unexecuted two-step training run.

## Recovered 79.5% adapter and Strategy A/B preparation

The [Drive benchmark report](https://drive.google.com/file/d/1d4Bu9TnPyki1wsU9dcN2eGBNhNx_Lhd6/view) identifies the earlier comparator as `e8e5edc7a75f82507db3c3fe0afca721728cd5f8abe03a80fa997139970fbc0e` and explicitly retains it over the 78.1% v3 candidate. **79.5% is 114.5/144 half-credit points on directly Codex-reviewed legacy 18-task cases. It is not an expert-gold score, a score over all 3,600 semantic cases, or a seven-task macro score.** The report itself records unblinded/reused diagnostics and limited independence. The approximate prior 0.558 macro figure is also not directly comparable.

The [matching Drive adapter folder](https://drive.google.com/drive/folders/1_JyOyaLQhCMpVyu8JbtWrFw71IxtF1Za) and two transfer parts were traced through the saved benchmark identity. The parts were reconstructed on the SSD as a safetensors file, not executed or loaded as a model. Verified exact files:

- Config SHA-256: `1fe1fdbfa6b611c63f9d7dc84d431f2b3685338d91678510f579b24ae002d5df`.
- Weights SHA-256: `154c508775106e8ffe15e224c9f256131852d384e4efaea6af2bdf3d35e99c6f`.
- Their legacy combined identity matches `e8e5edc7...` exactly. The saved adapter config has `revision: null`, so its historical base revision is unproven. The continuation job explicitly pins the new base revision. See `GPU_CANARY_TEST_B.md` for the verified package, local tokenizer preflight and cloud execution notebook.

This checkpoint uses **rank-32 RSLoRA, alpha 32, dropout 0.0**. The worker now preserves that scaling variant instead of silently using rank-16 standard LoRA. Native PEFT on a tiny random CPU Qwen-shaped fixture verifies all seven target modules and scaling `32/sqrt(32)`. This is not a load test of the full saved adapter on quantized Qwen.

| Setting | Strategy A: continued | Strategy B: fresh |
| --- | --- | --- |
| Initialization | Exact recovered 79.5% adapter | Pinned base, new standard LoRA |
| Rank / alpha / dropout | 32 / 32 / 0.0, RSLoRA | 16 / 32 / 0.05 |
| Data | 64 new + 21 stratified replay = 85 | 64 new + all 192 revalidated replay = 256 |
| Replay provenance | Legacy targets explicitly rebuilt from final text under policy v3 | Same rebuilt corpus, cumulative |
| Learning rate | 0.00002 | 0.00005 |
| Microbatch / accumulation | 1 / 4 | 1 / 4 |
| Optimizer steps | 2 maximum | 2 maximum |
| Context | 4,096 | 4,096 |
| Proposed worker limit | $1, not authorized | $1, not authorized |

The proposals are two alternative engineering canaries, not a matched performance experiment. Only one should be selected after approval. No GPU provider or hourly rate has been approved. Proposed hardware is one 24 GiB CUDA GPU. Four-bit base weights alone have an approximate 4 GB lower bound; quantization metadata, adapters/gradients/optimizer and sequence activations add memory. Actual peak VRAM, training throughput, CUDA compatibility and adapter reload remain unmeasured. The watchdog is a worker-runtime cost estimate and does not stop provider idle billing; the operator must shut down paid infrastructure.

## Prepared OpenAI rendering canary — NOT EXECUTED

- Four synthetic master families: hospitalization and diagnostic cases, readable report/discharge text, an unreadable control, an incomplete report and date contradiction/unknown applicability.
- Model `gpt-6-luna`; operation `document_rendering`; `POST /v1/responses`; reasoning disabled; no tools or coverage decisions.
- 3 unique requests, at most 6 attempts, concurrency 1. Unreadable evidence stays Python-only; missing fields must remain absent.
- Estimated unique input tokens: about 1,144, using UTF-8 bytes/4, **not measured GPT tokenization**. Unique output cap: 2,304; all-attempt output cap: 4,608.
- Conservative pre-request reservation across all attempts: about $0.006518. Hard cumulative ceiling: **$0.02**. Standard pricing $0.10/M input and $0.50/M output was checked on [OpenAI's model page](https://developers.openai.com/api/docs/models/gpt-6-luna) on November 19; the budget helper also reserves the higher cache-write input rate and byte-based framing bound.
- Planning latency estimate: 10–120 seconds, unmeasured; six 60-second request timeouts plus transport backoff are possible. No latency measurement is claimed.
- Acceptance: strict JSON, independent final-text parse, exact facts/absence/signature conditions and unchanged rule findings. Unsupported prose or changed facts are quarantined; valid peers continue.
- Utility criterion: at least two verified novel surface forms **outside the nine-style Python baseline bank**. Reproducing another existing Python layout does not pass. Surface novelty is not a measured student improvement.
- Expected projections if all families pass: 28; actual accepted examples remain unknown. No training has been scheduled from them.
- Completed queue keys are cached. Ambiguous network failures retain budget reservations and remain uncertain without automatic network retry; provider reconciliation is required. Provider-side exactly-once billing is not promised. Existing per-attempt usage/Supabase integration is reused.

## Independent evaluation preparation

The existing 540 fixed-validation cases are explicitly locally authored synthetic data without human review; the 100 inference-review cases span the old 18 tasks. The Drive 144-case review is a reused, unblinded legacy diagnostic, not an independent new-task probe set. None was relabeled as independent transfer evidence or used to create new release-holdout training cases.

`evaluation/probes/authoring-kit-20261119/` contains task schemas, import/case contracts and a review checklist with source provenance, author/reviewer identity, case hashes, evidence relevance and leakage requirements. It creates zero cases. External authorship/review is still required; transfer and release scoring remain unavailable. Family/template/content exclusions, sealed-holdout fingerprints, access logging/limits, paired intervals and promotion regression safeguards are preserved.

## Tests, operational issues and modified files

Final command, using the isolated native CPU environment, ran **259 tests: 259 passed, zero skipped**, in about 17.6 seconds. This includes all prior functional tests and 21 new quality tests. Native packages exercised: `{"accelerate": "1.11.0", "datasets": "4.4.1", "lighteval": "0.11.0", "peft": "0.18.0", "tokenizers": "0.22.2", "torch": "2.15.0", "transformers": "4.57.3", "trl": "0.25.1"}`. Tiny random CPU SFT/PEFT fixtures are not Qwen training and incurred no GPU cost.

Intermediate failures exposed Mac internal temporary-space pressure, AppleDouble sidecars after moving the task-specific environment to exFAT, legacy POSIX-temp fixture assumptions, and stale notebook bundle checksums after backward-compatible student/backend changes. The task-owned environment was moved onto the authorized SSD, its metadata sidecars were removed, full tests used POSIX `/tmp` after space was freed, and the legacy source ZIP/notebook checksum bindings were rebuilt. Runtime benchmark discovery ignores AppleDouble metadata. Actual storage/pilot runs and packages were verified on the SSD. No unrelated data, previous dataset or checkpoint was deleted.

Code/configuration/documentation changes:

- `autonomous_learning/evidence.py (new)`
- `autonomous_learning/tasks.py`
- `autonomous_learning/dataset.py`
- `autonomous_learning/scenarios.py`
- `autonomous_learning/rendering.py`
- `autonomous_learning/diversity.py (new)`
- `autonomous_learning/evaluation.py`
- `autonomous_learning/planner.py`
- `autonomous_learning/jobs.py`
- `autonomous_learning/gpu_worker.py`
- `autonomous_learning/tokenizer_preflight.py (new)`
- `autonomous_learning/model.py (new)`
- `autonomous_learning/canary.py (new)`
- `autonomous_learning/evaluation_preparation.py (new)`
- `autonomous_learning/teacher.py`
- `autonomous_learning/orchestrator.py`
- `autonomous_learning/__main__.py`
- `distillation_loop/student.py`
- `self_instruct_pipeline/parent_backend.py`
- `config/autonomous_learning.pilot.json (new)`
- `config/continued_adapter_candidate.json (new)`
- `tests/test_autonomous_quality.py (new)`
- `tests/test_autonomous_learning.py`
- `colab_distillation_bundle.zip and its four notebook checksum bindings`
- `AUTONOMOUS_LEARNING.md, IMPLEMENTATION_REPORT.md, README.md, READINESS_REPORT.md`

No real patient data or credentials were added to manifests or Git. The Git repository stays in the local development folder; datasets, tokenizer cache, recovered adapter, environment and prepared jobs stay on the SSD. No Drive file was edited or shared.

## Exact artifacts and next actions

- Final pilot summary: `/Volumes/Extreme SSD/Codex Distillation Loop/experiments/quality-pilot-20261119-release/artifacts/../verified_summary.json`.
- Selected pilot dataset manifest: `/Volumes/Extreme SSD/Codex Distillation Loop/experiments/quality-pilot-20261119-release/artifacts/datasets/approved_sft/867727cbace8c74aa588baec403c09d0477831e66e3e12414be7131085314313/manifest.json`.
- Evidence comparison: `/Volumes/Extreme SSD/Codex Distillation Loop/logs/quality-20261119-final-evidence-comparison.json`.
- Full passing log: `/Volumes/Extreme SSD/Codex Distillation Loop/logs/quality-20261119-full-native-tests.log`.
- Token receipts: `{"continued": "/Volumes/Extreme SSD/Codex Distillation Loop/manifests/tokenization/51cb0864daa7680acd3afd5d68d31276bac2289cc24bf7921cbc850e81252810-b968826d.json", "fresh": "/Volumes/Extreme SSD/Codex Distillation Loop/manifests/tokenization/688374dc44bd45b04393a40218f1ded7ed1e7400604cf430840d9d3ed73879bd-b968826d.json", "pilot": "/Volumes/Extreme SSD/Codex Distillation Loop/manifests/tokenization/867727cbace8c74aa588baec403c09d0477831e66e3e12414be7131085314313-b968826d.json"}`.
- API plan: `/Volumes/Extreme SSD/Codex Distillation Loop/experiments/api-canary-20261119/plan.json`.
- GPU strategies and manifests: `/Volumes/Extreme SSD/Codex Distillation Loop/experiments/gpu-canary-20261119/strategies.json`.
- Strategy A package: `/Volumes/Extreme SSD/Codex Distillation Loop/gpu_jobs/packages/c03c14696b8ab4b394266f83d850641ae3e71f9eaa3cc749a34de12489834fc5/package.zip`; SHA-256 `19ec9e08f6bd85eb7ee585dbb62939db7c3fbcaa588f70f853b89b82a5751e2d`.
- Strategy B package: `/Volumes/Extreme SSD/Codex Distillation Loop/gpu_jobs/packages/f93e0a79fa49496a8fd73599b18343b510ed37c38d5b20e56ce1efcb1ea51fbf/package.zip`; SHA-256 `a3d62ec829e080c004eb4008ddea4e558f809e080f56ab951f694aa34915178b`.
- Verified adapter: `/Volumes/Extreme SSD/Codex Distillation Loop/checkpoints/candidates/e8e5edc7-drive-transfer/adapter`.

Inspect or resume the final no-cost run from the repository with:

```bash
python3 -m autonomous_learning status --config config/autonomous_learning.pilot.json --root '/Volumes/Extreme SSD/Codex Distillation Loop/experiments/quality-pilot-20261119-release/artifacts' --run-id quality-final
python3 -m autonomous_learning resume --config config/autonomous_learning.pilot.json --root '/Volumes/Extreme SSD/Codex Distillation Loop/experiments/quality-pilot-20261119-release/artifacts' --run-id quality-final
```

Next choose whether to authorize the **$0.02 API canary**. GPU execution separately requires explicit provider/hardware/rate/budget/strategy approval. After approval, verify the chosen archive hash, extract on the approved CUDA machine, install the packaged requirements, run `python -m autonomous_learning.gpu_worker . --validate-only`, and only then use its existing `--authorize-training --max-gpu-usd ... --gpu-hourly-rate-usd ...` interface within the prepared limit. Keep result import/evaluation/promotion separate and never treat the prior 79.5% as a new seven-task score.

Unattended Codex service deployment, automatic legacy-teacher A/B execution, arbitrary clinical-prose verification, and independently reviewed release data remain outside the verified functionality. No full operational or deployment readiness is claimed.
