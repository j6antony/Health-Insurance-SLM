# Fixed evaluation update

The default now loads the saved 540-case synthetic benchmark instead of generating
validation through the API. Both full and smoke modes have zero evaluation-data
construction calls. The schema/evidence/hash/coverage checks and controller path
are covered by tests. The corpus has 10 families and 3 challenge variants per task;
it is development data, not independently reviewed clinical gold. See
[data/evaluation/README.md](data/evaluation/README.md). Earlier missing-reviewed-data
notes below describe explicit reviewed mode, not the current fixed default.

# Distillation loop hardening — November 10, 2026

Re-reviewed against the updated “Distillation Loop Recommendations” chat and
adversarial cases, rather than treating the earlier patch as finished.

## Defects fixed

- A matching document type could override a wrong review flag. Both now count.
- Matching codes could hide an incorrect explanation or missing information.
- Exact prose matching penalized valid paraphrases. Reviewed alternatives pass;
  unfamiliar reasoning goes to hash-bound independent adjudication with frozen
  generations. Incomplete judgments cannot produce a model-selection score.
- Final-test regressions were reported but did not prevent release. Validation
  regressions now exclude candidates; final promotion has quality, improvement
  and per-task regression gates, falling back to the initial adapter on failure.
- The smoke helper inherited reviewed mode and could not run without full gold
  data. It now explicitly selects synthetic smoke mode and forbids promotion.
- Resuming a partially filled round could regenerate a different training batch.
  Accepted round inputs are frozen before training, checked by digest, and reused
  with completed checkpoints. Insufficient batches stay recoverable, not frozen.
- API/network/auth/budget failures could consume sample or audit repair allowances.
  Service failures now have separate journals and preserve quality retry counts.
  An exhausted service request aborts for resume rather than spending through
  siblings; nonretryable sample refusals do not discard valid sibling answers.
- Optional checks had no enforced coverage. The default mixed curriculum now has
  a 25% independently computed fictional-rule component, a coverage gate, and
  explicit task/pattern accounting. The full new-example target is required by
  default; an intentional partial-data policy must be configured explicitly.
- Rule inputs accepted empty Boolean sets, mismatched Boolean/numeric types,
  nonfinite numbers, or fractional date offsets. These now fail validation.
- Split overlap checks depended on document IDs/order. Normalized individual page
  content is now checked across splits; malformed records and leaked context
  labels are rejected. Explicit evaluation mode and quality settings are checked.

## Validation and limits

Run `python3 -m unittest discover -s tests`, then the read-only readiness check
`python3 -m distillation_loop.preflight` on the intended training machine.
Validation result: 137 tests run, 134 passed and 3 native-dependency tests skipped.
Compilation and Git whitespace checks passed. The CPU integration test runs the real Pipeline/curriculum/controller/evaluator
through baseline, one training round, selection, final comparison, release and
completed-run resume. Teacher and model calls are test doubles; the reference
fixtures are explicitly test-only and are not shipped as gold evaluation data.
Adversarial tests exercise pending review/resume, changed inputs, retry recovery,
regression rejection, shortfalls and baseline fallback.

Actual reviewed validation/test files are still absent. This macOS environment
has no PyTorch/Transformers/TRL/PEFT/Lighteval installation and no CUDA training
runtime. A real GPU smoke run and meaningful baseline-versus-adapter experiment
remain unverified. Code tests cannot create independent expert judgments or prove
that a model improves healthcare decisions.

The executable templates intentionally test narrow fictional rules; they are not
clinical validation and do not make the teacher-generated remainder independent.
Regression gates use observed scores and do not imply statistical significance.
Heuristic deduplication cannot guarantee semantic leakage prevention. The system
therefore blocks promotion without reviewed evidence rather than calling an
unmeasured model production-ready.
