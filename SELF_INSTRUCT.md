# Automated healthcare Self-Instruct

The active pipeline uses **GPT-6 Luna** as the parent for scenario generation, reference answers, and separate automated audits. The downstream student is **Qwen3-8B**. No student weights, SFT training, DSPy, LightEval, or GOLD are loaded by this pipeline.

The latest **Self Instruct Implementation Guide** specification was used for the implementation. OpenAI GPT-6 Luna is now configured through the Responses API. Gemini remains an explicitly selectable backend; there is no silent provider fallback. Model access is checked using the configured account before generation.


## Run

Python 3.10+ on macOS/Linux; the active pipeline uses only the standard library. The imported upstream `requirements.txt` is legacy and is not needed for this runner.

Set `OPENAI_API_KEY` in the environment using your usual secret manager or shell. Keys are not loaded from config or stored in repository files. No key value belongs in source code or configuration. On Kaggle, use Add-ons → Secrets and enable access to the secret named OPENAI_API_KEY.

```sh
python3 run_self_instruct.py dry-run
python3 run_self_instruct.py generate
python3 run_self_instruct.py answer
python3 run_self_instruct.py export
```

Or run the stages together:

```sh
python3 run_self_instruct.py all
python3 run_self_instruct.py resume
```

`resume` reconstructs state, finishes generation to the configured target, answers only unanswered scenarios, and rebuilds exports. Dry-run requires no credential and never calls the API. It prints task/difficulty quotas, selected seed IDs, relevant schema keys, and the complete two-message generation request. The first fenced text block of `data/self_instruct/GENERATION_PROMPT.md` is the system prompt; documentation outside it is excluded.

Configuration: `config/self_instruct.json`. Paths resolve relative to the workspace, regardless of the working directory. Defaults are 120 accepted scenarios, batches of 8, 8 demonstrations, and weighted balanced scheduling. `claim_scrubbing` has weight 3; classification and extraction have weight 2; the remaining 15 tasks have weight 1. Weights are experiment settings, independent of the number of seed rows. Retry/failure limits, token budget, temperature, seed, and duplicate thresholds are configurable.

Isolated live check:

```sh
python3 run_self_instruct.py all --target 2 --batch-size 2 --output-dir outputs/my_smoke
python3 run_self_instruct.py resume --target 4 --batch-size 2 --output-dir outputs/my_smoke
```

Increase `--target` when resuming. Other configuration/source changes require a fresh output directory so existing records cannot silently change meaning. In particular, preserve `--batch-size 2` when resuming an isolated run started with that setting.

## Implementation and data flow

- `scheduler.py`: weighted-deficit quotas from cumulative accepted counts. Temporary per-batch reservations do not count as accepted data.
- `sampler.py`: full original demonstrations from explicit task labels, across 6-8 distinct categories; priority goes to requested/underrepresented tasks. At most 25% may be validated generated scenario-only examples, with no invented answers. There is only balanced mode.
- `prompt_builder.py`: distinct scenario, source-usability/novelty, teacher-answer, and answer-audit prompts. Teacher and student inputs use an explicit allowlist, excluding generation metadata and seed answers.
- `validation.py`: exact scenario/answer contracts, quotas, reference IDs, source quotes, extraction page/value checks, code-candidate membership, and uncertainty flags. It uses the schema subset required by the current 18 schemas and fails on unsupported keywords instead of silently ignoring them.
- `deduplicator.py`: exact fingerprints, configurable ROUGE-L and context overlap, and reasoning-contrast guards. A source-text model audit also compares semantic novelty with nearest same-task examples and candidates in the batch. `semantic_duplicate_threshold` controls additional lexical shortlisting for that audit, not a calibrated embedding distance. No embedding model is downloaded.
- `coverage_tracker.py`: per-task weights/deficits, difficulty, case types, skills, and overlapping coverage dimensions over the last 40 accepted scenarios. Coverage dimensions come from the separate source-text audit rather than solely from generated `case_type` labels.
- `parent_backend.py`: OpenAI Responses and Gemini REST clients, environment authentication, explicit model check, JSON validation, finite transient retries, rate-limit backoff, complete-response checks, certificate validation, token usage/error-code logging. No raw HTTP error bodies, request headers, or credentials are logged.
- `pipeline.py` and `persistence.py`: bounded generation, append-and-fsync persistence, global IDs, provenance, source/config hashes, file locking, restart reconstruction, and audit quarantines. Corrupt/truncated JSONL is reported rather than silently discarded.
- `dataset_builder.py`: only schema/evidence/audit-passing answer pairs are exported. Source hashes are checked again. Metadata remains outside the messages sent to the student.

The existing bootstrap entry point and generation/answer/export shell scripts forward to this one active implementation. The old classification-detection and GPT-3 fine-tuning shell stages are explicitly disabled. The upstream Python source is retained as reference, but the active path never imports `gpt3_api.py` or uses `davinci`.

## Outputs

Under the configured output directory:

| File | Content |
|---|---|
| `accepted_scenarios.jsonl` | Validated novel scenarios, original batch-local IDs, global IDs, seeds, parent/settings, source-audit outcomes |
| `validated_answers.jsonl` | Answers joined to scenario IDs/hashes, exact evidence checks, separate answer audit, verification status |
| `rejected_records.jsonl` | Rejection phase, reason, quarantined scenario/answer where available |
| `generation_batches.jsonl` | Planned quotas and parent-seed provenance |
| `coverage_ledger.json` | Accepted/rejected coverage and remaining rolling targets |
| `scheduler_state.json` | RNG/exposure checkpoint; accepted counts are rebuilt from durable records |
| `api_usage.jsonl` | Operations, retry/error status, timing, provider token usage |
| `run_manifest.json` | Contract/config hashes, timestamps, target, validation limitations |
| `parent_model.json` | Confirmed API model and output limit |
| `qwen3_8b_sft.jsonl` | task/instruction/input/output pairs |
| `qwen3_8b_chat.jsonl` | user/assistant messages suitable for downstream Qwen chat-template preprocessing |
| `export_metadata.jsonl` | Row IDs/tasks, parent and verification status, separate from model inputs |

Generated output directories are ignored by Git. Original seeds are not automatically added to the SFT export. Held-out evaluation items are never read or used for bootstrapping.

## Validation limits

No human approval is required. An accepted pair is labelled `automatically_checked_silver`, never gold. Scenario schema checks, exact evidence, and same-provider model audits do not establish medical/legal correctness. A model can repeat its own error during review. General natural-language arithmetic, date interpretation, policy applicability, and unsupported-action checks rely partly on the separate audit; explicit numeric constraints also receive deterministic checks. Missing-information examples can legitimately pass when the response accurately preserves uncertainty.

Coding and medical-necessity seeds contain explicitly fictional nonbillable catalogs/training criteria; they teach grounded reasoning, not authoritative medicine or coding knowledge. To generate real code/policy tasks, supply a JSON reference pack through `paths.references`, containing immutable excerpts with `id`, `text`, version, jurisdiction, and applicable dates. Referenced excerpts and metadata must appear in target-visible scenario context/documents. Actual source applicability still needs reliable source curation. Without such references, generation is restricted to synthetic rules and explicit uncertainty.

Failed or unverifiable answers remain quarantined and are excluded from export; `answer` returns a failure if any accepted scenarios lack a passing answer. There are finite per-call, per-batch, and consecutive-failure limits. A clean run is not a benchmark result or evidence of student improvement.

## Tests

```sh
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s self-instruct/self-instruct/tests -v
python3 data/self_instruct/validate_seeds.py
```

The tests use mocked parents and require no paid API calls. The live smoke output is in `outputs/self_instruct_gemini_smoke`; it is a small integration check, not the full default 120-scenario dataset.

Official API references used: [Google Gemini API](https://ai.google.dev/api), [model metadata](https://ai.google.dev/api/models?hl=en), and [generateContent](https://ai.google.dev/api/generate-content?hl=en).

OpenAI uses JSON object mode with a private `result` transport wrapper so array responses and arbitrary scenario context remain compatible with the existing contracts. The original JSON schema is included in the instructions and enforced locally before returning the unwrapped result. This does not claim native strict-schema constrained decoding; malformed outputs are rejected and follow the bounded retry policy. Reasoning effort defaults to medium, with temperature omitted while reasoning is enabled.

When switching providers, use a new output directory. Existing Gemini logs remain historical evidence only, not a live OpenAI verification. OpenAI request formatting, usage logging, refusal/truncation handling, authentication, and retry behavior are covered by mocked tests. Official references: [Responses API](https://developers.openai.com/api/docs/guides/text), [JSON outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses), and [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna).
