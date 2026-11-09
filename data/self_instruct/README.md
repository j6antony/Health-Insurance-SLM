# Claims Self-Instruct seed pack

72 original AI-authored seed tasks spanning all 18 subtasks identified in the earlier claims-distillation dataset. This is a reviewable starting set, not independently validated gold or a measured improvement over the previous 35-example set.

## Files

- `seed_tasks.jsonl`: Self-Instruct seed record layout extended with an explicit `task` field: id, name, task, instruction, instances with string input/output, is_classification. Each input includes its response schema. Every row has exactly one of the 18 task labels defined in `TASKS.md` and `task_schemas.json`.
- `seed_tasks.annotated.jsonl`: the samX
- `SEED_EXAMPLES.md`: readable instructions, scenarios, and reference answers for all 72 seeds.
- `TASKS.md`: definitions and seed counts for all 18 task labels.
- `task_schemas.json`: copied from the prior project's `training_set/metadata/us_distillation_medgemma_v3/task_schemas.json`. This preserves its task contracts; schema version is independent of this seed pack.
- `GENERATION_PROMPT.md`: task-and-scenario generator prompt, sampling advice, and separate teacher-answer stage.
- `validate_seeds.py`: dependency-free structural/schema-subset and evidence checks. It does not validate medical, legal, or business correctness.

## Coverage and design

| Scope | Seeds |
|---|---:|
| Claim scrubbing/completeness | 12 |
| Document classification | 8 |
| Field extraction | 7 |
| Each of the other 15 subtasks | 3 |
| Total | 72 |

The other subtasks are diagnosis/supply code explanation, clinical coding, denial explanation, appeals-process Q&A, coverage Q&A, eligibility/benefits, patient billing, compliance, support chat, policy/claim-history/support-call summarization, appeal drafting, and medical-necessity documentation support.

Cases cover negative controls, present-but-unreadable documents, required-field scope, unknown conditional triggers, AND/OR alternatives, exceptions, wrong-date sources, conflicting identifiers/statuses, unsupported requests, and injected document instructions. Paired contrasts include absent vs unreadable, required vs inapplicable conditional documents, supplied vs missing sources, and documented vs undocumented criteria. These differences matter more than changing names or amounts.

Reference answers demonstrate compact JSON, evidence fidelity, uncertainty, and appropriate task boundaries. For common-schema tasks, evidence sometimes quotes an entire short supplied record to preserve relevant qualifications. It is not a long-context evidence-localization benchmark. Names and identifiers are invented. No real patient records, real payer rules, real clinical thresholds, or actual code descriptions were added.

## Scope decisions

Sources reviewed: chats **Self Instruct Implementation Guide** and **API Distillation Plan**; this workspace's README; and the adjacent health-insurance-slm README, US pilot dataset READMEs, and task schemas. The guide's seven categories underspecify the earlier dataset's 18 subtasks, so this pack covers the broader scope and prioritizes document intelligence.

This workspace currently describes Qwen3-8B as the student, while earlier chats describe MedGemma as the student. These seeds are model-independent. They do not choose or change the training architecture, trainer, or notebook.

Classification follows the prior deployed seven-label contract: hospital_bill, discharge_summary, prescription, investigation_report, claim_form, other, unknown. A policy maps to other under this contract even though a broader project taxonomy also exists. Changing the taxonomy requires changing the seed answers, response schema, and downstream consumer together.

Missing extraction fields are omitted rather than emitted as null or invented zeros, matching the inherited fields-array schema. An empty fields array is valid when nothing requested is safely extractable. Ambiguous source text may be preserved with requires_review=true. Extraction outputs cannot carry document_id in this inherited schema; keep target-document scoping in the input and provenance outside the model answer. Multi-document provenance is stronger in the common evidence schema.

Diagnosis and supply-code examples use explicitly fictional, non-billable DX-/EQ- catalogs. They teach lookup, date applicability, negation, and abstention only. They do NOT teach real ICD-10-CM or HCPCS knowledge. For production coding generation, inject approved official catalog/guideline excerpts and their effective dates; never expand the fictional catalogs into real-looking code facts. Similarly, medical-necessity examples assess fictional documentation rules, not clinical necessity.

## Use in the loop

The active automated runner is now `run_self_instruct.py`; see `../../SELF_INSTRUCT.md`. It does not require human approval and exports only scenario/answer pairs that pass its automatic checks. The manual-review steps below remain optional quality guidance; the original seeds retain their honest `ai_drafted_unreviewed` status. The runner does not automatically export those seed reference answers as training targets.

1. Review the seed outputs and task contracts. Every row starts `ai_drafted_unreviewed`; record corrections and approval in the annotated file or a separate review log. No reviewer approval is claimed here.
2. Use `seed_tasks.jsonl` if adapting the original repository's seed loader. The record shape extends the [original seed file](https://github.com/yizhongw/self-instruct/blob/main/data/seed_tasks.jsonl) with `task`; retain this field in sampling, routing, and evaluation. A strict legacy consumer may need to ignore that additional field. No runtime compatibility claim is made for its legacy API implementation.
3. For a custom generator, sample 6-8 examples from the annotated pack, including contrasting behaviors. Pass only task, instruction, input, and output as demonstrations; keep review/provenance metadata out of model inputs. Prefer reviewed seeds once available.
4. Generate instructions and scenarios using `GENERATION_PROMPT.md`. Apply semantic and structural novelty checks, source checks, and a rolling coverage ledger. Keep the original reviewed seeds anchored in the pool; cap machine-generated exemplars at about one-quarter of a generation prompt initially. That sampling ratio is an experiment setting, not an established optimum.
5. Generate and validate teacher answers in a separate stage. Convert only accepted scenario/answer pairs to SFT examples; train with instruction plus input as the prompt and the reviewed answer as the completion. Do not include reference answers in the prompt. This pack is not a trainer-ready messages export.
6. Use development feedback to request new failure skills. Never feed held-out test items, reference answers, or near-copy scenarios into generation. All these seeds and derivatives belong to the generation/training pool; none constitutes an independent evaluation set.

The seed counts emphasize coverage, not an intended training distribution. Set generation quotas explicitly. The initial document-product example uses 3 scrubbing, 2 classification, 2 extraction, 1 denial task per batch of 8; rotate remaining subtasks across later batches. Broader model training can instead use the prior eight-bucket weights. Choose quotas based on the deployment scope and development performance rather than sampling uniformly from seed rows.

## Review criteria

For each seed, verify: the task belongs in scope; the instruction and schema agree; the response uses only supplied facts; every quote exists at the recorded source/page; missing vs unreadable vs unknown is correct; exceptions and conditional triggers are honored; arithmetic and dates are correct; and the answer does not invent a code, policy, clinical conclusion, completed action, or final claim decision. A review flag signals uncertainty or human workflow review; it is not a calibrated confidence score.

Known limitations: compact authored examples are easier and more correlated than real documents; most examples have one instance; no model inference or training was run; structural validation cannot prove domain correctness. Add reviewed long documents, multi-page layouts, approved real code references, and richer dialogue in subsequent generation rounds. No OCR, PDF processing, or image-understanding capability is established by these text seeds.

Validate locally:

```sh
python3 data/self_instruct/validate_seeds.py
```
