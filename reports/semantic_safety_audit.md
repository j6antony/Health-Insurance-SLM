# Semantic safety audit

All 46 packet rows reconcile with the union of 39 new unsupported-claim flags, four new missed-review flags, and 18 losses in severely regressing tasks. The verifier matched the exact flag sets and saved scores, both raw predictions and generation metadata, original source text, reference answers, source families and accepted alternatives. It also verified diagnostic identities for the split, manifest and all three generations. The frozen split remains 360 development cases and 108 release cases.

The companion JSONL records the initial-to-SFT semantic difference, source excerpts and provisional disposition for every flagged case. It includes 31 source-supported errors, five apparent judge false positives, one possible rubric issue and nine cases requiring a second read. All 46 classifications are provisional Codex judgments. Zero have independent human confirmation. The saved rubric provided flags and structured judgments rather than per-assertion chains of reasoning, so the audit explains the evidence directly.

The strongest errors include using documentation silence to assert an appeal never occurred, asserting nonpayment despite equally ranked paid and unpaid entries, changing four review flags to false with required evidence still missing, and issuing opposite conclusions under the same explicit OR rule. Embedded directives sometimes appear as governing rules or as part of conclusions. Several answers also manufacture uncertainty about sources that explicitly state a fact. One medical-necessity answer repeats until incomplete.

Candidate false positives include abstaining from coding when no finding is documented, preserving a conflict between policies without assigning precedence, and applying a documented exception without guaranteeing reimbursement. The ICD expiry case may expose a rubric threshold for explaining the consequence of an already stated date mismatch. Leave all saved scores and gate decisions in place pending independent review.

From the repository root, run the reconciler against the original files to repeat the identity and contamination checks:

```sh
PYTHONPATH=. python3 tools/reconcile_semantic_audit.py \
  /path/to/semantic_safety_review_pack.jsonl /path/to/semantic_comparison.json \
  /path/to/evaluation_split.json /path/to/checkpoint_eval /path/to/training_plan.json
```
