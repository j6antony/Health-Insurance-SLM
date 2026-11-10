# Notebook training and evaluation loop

Open `distillation.ipynb` and run its cells in order on a CUDA GPU. Attach the full repository so the notebook can import `self_instruct_pipeline` and `distillation_loop`. The notebook installs `requirements-training.txt`, obtains credentials from the environment/Kaggle Secrets, and loads Qwen3-8B once in 4-bit mode. The existing inline Hugging Face credential has been removed.

The parent is GPT-6 Luna through the OpenAI Responses API. Provide `OPENAI_API_KEY` in Kaggle Secrets or the environment. The public Qwen student loads anonymously with `token=False`; Hugging Face login is not required. No credential is embedded in this notebook or configuration.

## Sequence

The notebook calls one function, `run_loop(student_runtime, parent_config, loop_config)`. Its main flow is a plain bounded `for` loop in `distillation_loop/controller.py`; there are no injected factories or multi-phase state machine. Model-specific training and evaluation helpers stay in their own files.

1. Prepare trainable LoRA adapters on the frozen quantized Qwen3-8B base.
2. Create or load a frozen development set: three parent-checked synthetic examples for each of the 18 tasks (54 total by default). References are stored separately from student inputs.
3. Evaluate the starting student with Lighteval custom metrics. Stop without training if all targets already pass.
4. Ask Self-Instruct for new scenarios, using weighted deficits and the previous development per-task scores to prioritize weaknesses.
5. Run the existing student on those prompts. Save raw and cleaned responses, including schema/evidence failures and truncation flags. Cleaning removes complete thinking/code-fence wrappers; it does not repair malformed JSON or invent fields.
6. Generate and validate parent reference answers in separate calls. Train on those parent completions using TRL SFTTrainer. The student forward pass supplies the predictions whose cross-entropy loss is optimized against those labels. Sampled student text is diagnostic data, never an incorrect self-training target.
7. Save the updated LoRA adapter and evaluate the same frozen development set through Lighteval.
8. Stop when all score gates pass or when the round cap is reached. Otherwise generate the next round with higher weights for weak tasks. Restore the best saved adapter at the end.

## Defaults and bounds

Edit `config/distillation_loop.json` or the notebook's configuration cell **before starting a new run**:

| Setting | Default |
|---|---:|
| Maximum training rounds | 5 |
| New scenarios per round | 72 |
| Development examples per task | 3 |
| Required macro task score | 0.90 |
| Required minimum score on every task | 0.80 |
| Required schema/evidence-valid rate | 0.98 |
| Maximum optimizer steps per round | 100 |
| Epochs per round | 1 |
| Inference completion budget | 1,024 tokens |
| Total context/training sequence budget | 4,096 tokens |

These are editable experiment targets, not clinically established acceptance criteria. Optimizer steps are capped at the smaller of the step limit and the number of steps for the configured epochs. Generation also retains the parent pipeline's finite API retries, batch retries, and consecutive-failure limits. Development coverage filling has a separate finite attempt limit. API/model/training failures stop explicitly; they are never interpreted as satisfactory scores.

Initial development setup and each training round make paid parent calls. The notebook does not run these calls simply by importing modules or previewing a request. Maximum rounds includes training rounds only; baseline evaluation happens once before them. Reaching the cap returns `max_rounds_reached`, not a claim that targets were achieved.

## SFT behavior

The notebook uses pinned TRL 0.25.1, Transformers 4.57.3, PEFT 0.18.0, and Lighteval 0.11.0 APIs. It requires a CUDA-enabled torch installation (provided by Kaggle); the parent-generation CLI still needs only the standard library. It does not install the imported upstream legacy GPT-3 requirements.

The non-thinking Qwen chat prefix is identical in inference and training. Teacher JSON completions and EOS are appended to that prefix, with a completion mask that sets every prompt label to `-100`. The actual TRL collator is checked before training for both prompt and completion masks. Overlength examples are reported rather than silently truncated. Nonfinite training loss prevents a checkpoint from being committed.

Only LoRA parameters train. Each round starts a new optimizer on the current adapter; optimizer state is not continued between rounds. By default, all accepted teacher examples from earlier rounds are replayed to reduce forgetting; disable `replay_previous_rounds` to train on newly generated pairs only. Student diagnostics are excluded from trainer labels and prompts.

The notebook uses one process on GPU 0. Batch size is one, with gradient accumulation and checkpointing. Qwen3-8B at the selected context length still needs sufficient GPU memory; CPU mode is not a substitute for the configured bitsandbytes QLoRA run. Full Qwen GPU training was not executed in the local desktop environment.

## Lighteval scoring

`distillation_loop/evaluation.py` uses the real Lighteval `Doc`, `ModelResponse`, `SampleLevelMetricGrouping`, sample computation, and corpus aggregation interfaces on generations from the already-loaded student. It does not load another copy of Qwen or run a standard public benchmark suite.

Scores are task-specific and lie in [0, 1]:

- Classification: exact document-label accuracy, gated by schema and source evidence.
- Extraction: F1 over field name, normalized value, and review flag; source quote/page checks must pass.
- Clinical coding: suggested-code-set F1, gated by source membership, schema/evidence, and the review flag.
- Free-text tasks: ROUGE-L over the answer and any draft/criteria text, gated by schema/evidence, review-flag agreement, and missing-information/no-issue distinctions.

The stop gate requires the mean of per-task scores, the minimum per-task score, and the overall schema/evidence-valid rate to pass together. Every task must have the configured number of examples. Missing tasks, inconsistent summaries, nonfinite scores, or truncated outputs cannot pass. All-valid JSON alone is insufficient.

Free-text overlap is conservative and wording-sensitive. It does not prove semantic equivalence, sound medical judgment, or real coverage accuracy. The frozen development set is parent-checked synthetic silver data, not an independently validated test benchmark. Repeatedly tuning on its aggregate scores can overfit it; reserve an independently reviewed final test set for later reporting.

Development prompts and answers never enter student training, generator demonstrations, or model-based novelty comparisons. Training candidates are compared locally against held-out records for exact/near overlap. Only per-task aggregate scores influence subsequent generation weights. This catches direct copying, but cannot guarantee the absence of all semantic/template correlation.

## Persistence and recovery

Kaggle writes to `/kaggle/working/distillation_loop_openai`; locally, the notebook defaults to `outputs/distillation_loop_openai`. The notebook overrides the output path so an attached read-only repository can still be used.

- `development.jsonl` and its manifest freeze the evaluated examples and answers.
- `loop_state.json` records completed rounds, score history, stop reason, last completed adapter, and best adapter.
- `self_instruct/` contains the reusable parent scenario/answer stores and usage/error logs.
- Each `round_NN/` contains student diagnostics, generation feedback weights, SFT metrics/adapter, and Lighteval predictions/scores.

Rerun the final notebook cell with the same configuration to resume. A completed run does not generate/train again. An unfinished round restarts from the last completed predecessor adapter, including retraining if evaluation had not finished. Only whole completed rounds are resumed; there is no phase-by-phase recovery machinery. Configuration, student identity, or frozen-development changes fail clearly; use a new output directory for a changed experiment. Keep the output directory or attach its saved copy when restarting Kaggle sessions.

## Checks

```sh
python3 -m unittest discover -s tests -v
```

Tests cover cleaning, label masks, gates, coverage, adaptive weights, threshold/cap stopping, best-model restoration, and recovery after training/evaluation interruptions. Optional `test_native_training.py` checks the actual pinned Lighteval metrics and a one-step tiny CPU SFT model without API calls or model downloads when training dependencies are installed. These CPU checks do not establish Qwen3-8B CUDA/QLoRA compatibility or training improvements.

Primary API references: [TRL SFTTrainer](https://huggingface.co/docs/trl/v0.25.1/en/sft_trainer), [Lighteval custom metrics](https://huggingface.co/docs/lighteval/main/en/adding-a-new-metric), and the inspected published Lighteval 0.11.0/TRL 0.25.1 wheels.

## Switching from Gemini on Kaggle

Upload the updated repository files, including `self_instruct_pipeline/parent_backend.py`, `pipeline.py`, `config.py`, `config/self_instruct.json`, and the notebook. In Kaggle **Add-ons → Secrets**, create a secret named exactly **OPENAI_API_KEY**, paste your OpenAI key as its value, and enable notebook access. Do not paste the key into a Python cell or config file.

Run the updated credential cell, then model/settings cells. Use a fresh output directory such as `/kaggle/working/distillation_smoke_openai` for a smoke test, or keep the notebook's new OpenAI-specific default folder. Existing Gemini experiment folders cannot be resumed under a different parent configuration.

The backend uses standard-library HTTP, so no OpenAI SDK installation is required. `reasoning_effort` defaults to `medium`. Temperature is omitted while reasoning is enabled, as required by GPT-6. The same backend handles scenarios, reference answers, and audits. Model availability is checked with your account at runtime; local mocked tests do not prove account access.

## API usage and cost tracking

Every API attempt automatically appends to `api_usage.jsonl` and an Excel-readable
`api_usage.csv` inside `development_generation/` or `self_instruct/`. Rows include
UTC start/end timestamps, operation (generation, teacher, audit, or model check),
model/provider, attempt number, HTTP/response status, duration, response ID,
input/cached/output/reasoning/total tokens, estimated USD cost, and the pricing
rates/source/date. Credentials and prompt/response contents are not included.
Transport success can still have `response_status=incomplete`; reported tokens
are charged in the estimate even when the pipeline rejects the generated answer.
Retries have separate rows. Missing usage means unknown cost, never a zero-cost
assumption. Successful model-access checks are marked non-generation with $0 cost.

The loop exports a combined `RUN_DIR/api_usage.csv` and prints a cost summary on
completion or a Python exception. Per-call files remain available if Kaggle is
terminated abruptly. The last notebook cell refreshes the combined CSV manually:

```python
from self_instruct_pipeline.usage import export_usage
print(export_usage(RUN_DIR))
```

For the default notebook, download
`/kaggle/working/distillation_loop_openai/api_usage.csv` from Kaggle output files
and open it in Excel. A custom smoke run saves it in your chosen `RUN_DIR`.
Keep the JSONL journals with the run when resuming; CSVs are rebuilt from these
journals without duplicating calls. Existing historical rows without timestamps
or provider information cannot recover those missing details.

The estimate uses GPT-6 Luna Standard global text rates verified November 9,
2026: $0.10 input, $0.01 cached input, $0.50 output, and $0.125 cache writes per
million tokens. Reasoning tokens are included in output tokens and are not added
a second time. The documented long-context multipliers apply above 272,000 input
tokens. Prices are in `self_instruct_pipeline/usage.py`; update them if pricing or
models change. Unknown models/providers retain token counts but leave cost blank.
This is an estimate for this code's calls, not an account billing statement;
regional/tier premiums, other applications, and unreported usage after network
failures require checking your provider dashboard.

Pricing source: https://developers.openai.com/api/docs/models/gpt-6-luna

## Run the integration test first

The notebook now starts with `TEST_RUN = True` in its settings cell. Upload the
updated complete repository to Kaggle, enable a CUDA GPU and Internet, enable
access to the `OPENAI_API_KEY` secret, and run notebook cells in order. The preview
cell also runs a local student-generation check before any paid API request.

The smoke settings use a separate `/kaggle/working/distillation_smoke_openai`
folder: 1 round, 8 new training scenarios, 18 frozen development examples (one per
task), and at most 2 optimizer steps. All 18 evaluation tasks are retained; this
checks the same code path as a full run. Generation and answer retries are reduced.
The small development set checks integration, not reliable model quality.

The run cell activates one shared API budget for the entire test: 120 HTTP
attempts (including checks and retries) and $0.50 estimated USD. Before each
request it persists a conservative allowance for UTF-8 input size plus framing
and the maximum output-token budget, using Standard global text rates without a
cache discount. Reported usage replaces that allowance after the call; unknown
usage or an interrupted request retains it. The next call is refused if its
allowance would exceed the limit. `BudgetExceeded` stops immediately and is not
caught as a retryable provider error. Calls explicitly request Standard processing.
This is a local estimate, not a provider-enforced account spending limit.

`api_budget/state.json` preserves counts and accounted cost across kernel restarts
and reruns. Do not delete it to resume, and do not run multiple notebooks against
the same output folder. An old run without budget state is refused: choose a fresh
folder rather than ignoring its historical usage. Limits are fixed once requests
start. Full-run defaults have a separate $5 / 2,000-attempt allowance.

After the smoke run, inspect:

- `api_usage.csv`: calls, reported tokens, estimated costs, and unknown-cost rows.
- `loop_state.json`: normally `completed_rounds: 1` and `max_rounds_reached` or
  `thresholds_met`. If the initial model already passes the thresholds, training
  is skipped as usual; that outcome alone does not verify SFT.
- `round_01/`: training metrics,
  saved adapter, student outputs, and Lighteval scores.

If the budget or validation stops the test, inspect the reason before increasing
limits; repeated failures should be fixed first. No automatic budget increases
occur. A smoke run cannot establish full-run GPU memory requirements or guarantee
high scores. Before a full experiment, **restart the kernel**, set `TEST_RUN = False`,
and rerun from the top so the full experiment starts from the base student rather
than the smoke-trained adapter. The full run uses its separate output folder.

## Filtering failed answers and replacing evaluation cases

The training loop keeps only answers passing both local validation and grounding
audit. A rejected answer no longer aborts the whole answer pass: the bounded
attempts finish, and the valid subset is exported and trained. At least half of
each round's new examples (rounded up, minimum one) must pass; otherwise training
stops to avoid running a round mostly on old replay data. Each round records
counts in `training_filter.json`. Raw failed answers remain in the rejection log.
Authentication, request errors, and API-budget exhaustion still stop the run.

Evaluation retains exact coverage of all 18 tasks. After the initial answer pass,
it generates replacements only for missing task counts, validates their answers,
and retains earlier successful examples. `development_fill_attempts` now caps the
number of replacement passes AFTER the initial pass (one replacement pass for the
smoke test, three with default full settings). These passes are recorded in
`development_replacements.json` before calls and persist across resumes; an
interrupted pass resumes its saved targets instead of consuming another pass. The shared API budget
still covers every call. If coverage remains incomplete after this limit, the
loop stops with an error rather than evaluating a smaller task set. Once frozen,
`development.jsonl` is reused unchanged after every training round.

These changes can resume an existing run with the same configuration and output
folder. Copy the updated `distillation_loop` and `self_instruct_pipeline` folders
to Kaggle and restart the kernel. Keep the run's output and budget files intact.


## Failed-sample handling (Handle Failed Samples chat)

The training loop now uses **filter → repair with feedback → reverify → replace**.
Good scenarios and answers are saved independently. An invalid scenario or scenario
audit no longer discards its valid batch peers. Batch-envelope errors still reject
the request when samples cannot be identified safely. Multi-page documents use
unique `(document_id, page_number)` pairs; the same document ID may occur on several
pages. Evidence must still match that document and, for extraction, the stated page.

For partial answer passes used by the training loop, each sample gets at most
**three attempts total: one initial answer plus two repairs**. Every repair must
pass local checks AND a fresh grounding audit. The most recent failed proposal
and validation/audit feedback are supplied to the next attempt. Formatting failures
are regenerated as JSON; evidence and grounding failures are corrected against
sources, never automatically edited or accepted. This total replaces the legacy
`max_answer_attempts_per_run` behavior in partial passes, including smoke mode.
Standalone strict answer runs retain their previous per-run setting.

`answer_attempts.jsonl` reserves attempts before requests; historical rejection
records also count. Restarting a kernel does not reset sample attempts. A crashed
request conservatively consumes one attempt. Network/HTTP errors and API-budget
exhaustion stop the run instead of silently consuming repair cycles. Invalid
model-generated JSON may use the remaining repair allowance.

Development setup first repairs existing candidates for missing tasks, even if
replacement passes were previously exhausted. It then generates replacements only
for remaining missing task counts when replacement allowance remains. An active
replacement pass saves its task targets and resumes after interruption. Frozen
complete development data is unchanged. Training retains the earlier behavior of
using validated subsets, subject to the minimum acceptance requirement.

`failure_summary.json` reports validated answer counts, sample acceptance rate,
failure types/reasons, repaired answers, and repair success rate. Rejection rows
include `failure_type` and `repairable`; old rows retain their phase labels. Counts
measure automated synthetic verification, not independent factual accuracy.

Scenario generation also has a persistent per-goal batch ceiling: at least six
batches, or three times the ideal batch count for larger requests. Each batch's
existing request retry limit and consecutive-failure limit still apply. Progress
prints before generation and answer calls, showing tasks and attempt counts.
`generation_limits.json` and `generation_batches.jsonl` preserve this ceiling across
resumes. The API call/cost budget is unchanged; increasing training `max_rounds`
does not increase development-data repair or replacement allowances.

To update Kaggle, copy both `distillation_loop` and `self_instruct_pipeline` folders,
restart the kernel, and retain the same configs, output directory, and budget files.
Previously failed development candidates may now be repaired within their remaining
three-attempt allowance without resetting the replacement counter. If both sample
and replacement limits are exhausted, setup still stops and lists missing tasks.

## Verified-pair target correction

Each training round now requests a cumulative target of verified pairs through
`Pipeline.build_pairs`, rather than stopping at the same number of accepted
scenarios. Exhausted failed answers are replaced within three replacement passes;
good existing pairs are retained. Each training run's `self_instruct/` directory
contains `pair_build_state.json` and `pair_build_report.json` with the requested and
actual pair counts and an explicit partial/completed status. The API budget is
unchanged. If replacement limits leave a partial dataset, the existing minimum
new-example requirement still applies before SFT. Training records the actual
sample IDs used, so replay and later rounds do not count rejected candidates as
trained examples. Student generations remain diagnostics and never labels.

This addresses the follow-up review in the Handle Failed Samples chat: the normal
CLI now exports good partial results and builds toward a verified-pair target,
strict mode also obeys cumulative retry limits, and malformed audits retry without
regenerating the locally valid answer. Earlier descriptions of scenario-count
training targets or uncapped strict per-run answer retries are superseded by this
behavior. The standalone `generate` command still targets scenarios only.

## Kaggle multi-GPU training crash

A traceback through `torch.nn.parallel.data_parallel` followed by bitsandbytes
`CUBLAS_STATUS_EXECUTION_FAILED` shows unintended DataParallel replication in this
single-process, single-GPU notebook. The model is loaded in 4-bit on GPU 0; Trainer
must not replicate it across Kaggle's other visible GPU. The first setup cell now
sets `CUDA_VISIBLE_DEVICES=0` before torch imports. Student setup requires exactly
one visible GPU, and SFT checks that Trainer uses one GPU and one process.

After an illegal-memory-access error, restart the kernel before reloading the
student; changing visibility after CUDA initializes does not repair the context.
Keep the same run folder, generated data, adapters, and budget files when resuming.
The saved development set and verified training pairs are reused. Cleanup now
preserves the original training exception if emptying the damaged CUDA cache also
fails. This fixes the observed replication configuration; full CUDA validation
still requires rerunning on Kaggle.
