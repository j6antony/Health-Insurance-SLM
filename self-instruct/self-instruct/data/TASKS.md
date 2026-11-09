# Explicit task labels

Every example in both JSONL files has exactly one top-level `task` string. This is the primary operation requested of the model, not the scenario topic, difficulty, or classification answer. The exact string also selects its output schema in `task_schemas.json`.

| Task value | Intended operation | Seeds |
|---|---|---:|
| `icd_code_explanation` | Explain a supplied diagnosis-code entry and its applicability from a source catalog. Current seeds use fictional catalogs, not actual ICD entries. | 3 |
| `hcpcs_code_explanation` | Explain a supplied supply/equipment-code entry, units, and distinctions. Current seeds use fictional catalogs, not actual HCPCS entries. | 3 |
| `clinical_coding` | Select diagnosis-code candidates supported by the note and supplied coding rules; preserve uncertainty and negation. | 3 |
| `denial_explanation` | Interpret an existing denial notice, its evidence, and supported administrative next steps. | 3 |
| `appeals_process_qa` | Answer procedural appeal questions, including deadlines and submission routes, from applicable rules. | 3 |
| `policy_coverage_qa` | Answer a specific coverage-condition question using applicable policy clauses, versions, and exceptions. | 3 |
| `eligibility_benefits_qa` | Interpret membership dates, benefit limits, utilization, and network-dependent benefits. | 3 |
| `patient_billing` | Explain or reconcile patient amounts using supplied benefits, EOBs, invoices, and payments. | 3 |
| `compliance_qa` | Apply supplied administrative privacy/access/compliance rules or identify missing authority. | 3 |
| `support_chat` | Respond to the latest conversational turn using dialogue context and available capabilities. | 3 |
| `claim_scrubbing` | Check claim fields, documents, conditional requirements, consistency, and submission readiness against supplied rules. | 12 |
| `document_classification` | Assign the document's primary type using the fixed seven-label taxonomy and source evidence. | 8 |
| `field_extraction` | Extract requested populated fields with exact evidence and original page attribution. | 7 |
| `policy_summarization` | Condense supplied policy terms while preserving limits, conditions, exceptions, and version scope. | 3 |
| `claim_history_summarization` | Summarize a claim's events and latest supported status while preserving conflicts and claim boundaries. | 3 |
| `support_call_summarization` | Summarize a supplied transcript with speaker attribution, commitments, completed events, and action owners. | 3 |
| `appeal_drafting` | Write a source-grounded appeal draft for human review, using placeholders for unavailable details. | 3 |
| `medical_necessity_support` | Map supplied documentation to explicit criteria and identify gaps; do not decide clinical necessity or authorize care. | 3 |
| **Total** | | **72** |

Example of the main seed format (abridged):

```json
{
  "id": "seed_031",
  "name": "required_field_absent",
  "task": "claim_scrubbing",
  "instruction": "Check only the required claim fields listed in the supplied checklist.",
  "instances": [{"input": "...scenario and response schema...", "output": "...reference JSON..."}],
  "is_classification": false
}
```

Keep `task` in orchestration metadata for quotas, schema routing, dataset filtering, and per-task evaluation. It is not an additional required field in the model's answer. `is_classification` retains its narrower role in the original Self-Instruct generation flow; it does not identify the other 17 tasks.

Choose labels by requested operation: explaining a denial is `denial_explanation`, answering how to appeal it is `appeals_process_qa`, and writing the appeal is `appeal_drafting`. Similarly, answering one coverage question differs from summarizing a policy. Use secondary skills to describe cross-document reasoning or arithmetic without changing the primary label.
