# Chatgpt-luna and qwen3 based distillation loop implementation

## Purpose
The purpose of this is to fine-tune qwen3-8B using LLM(gpt6-luna) using a distillation approach. It will be implemented in a way where minimal human interaction is required as many tools will be used to distill qwen until the desired results are required.

## Tools/Approaches
- DSPY: for trained adapter evaluation and next prompt generation
- LightEval: benchmarking the trained adapters
- Self-Instruct: self-instruct will allow the training loop to generate prompts and scenarios upon which the model can be trained on. It is a methodology and not a tool.
- SFTTrainer: Will be used for loss calculation and back-propegation
  
## Implementation overview
![Architecture Diagram](assets/MedGemma%20Training%20Implementation%20Loop.png)

## Self-Instruct seeds

[Seed pack and usage guide](data/self_instruct/README.md): 72 synthetic seed tasks across 18 claims-related subtasks, with reference outputs, task schemas, a scenario-generation prompt, and a validation script. These are AI-authored examples awaiting domain review.

## Automated Self-Instruct pipeline

[Run guide](SELF_INSTRUCT.md) · [Configuration](config/self_instruct.json)

The active parent is `gpt-6-luna` through the OpenAI Responses API; the downstream student remains Qwen3-8B. The pipeline uses weighted balanced sampling, structured scenarios, separate teacher answers, automatic validation/audits, resumable storage, and SFT/chat JSONL exports. Human approval is optional; exported data is automatically checked synthetic silver, not independently validated gold.

Preview without an API key:

```sh
python3 run_self_instruct.py dry-run
```

With `OPENAI_API_KEY` set in the environment, `python3 run_self_instruct.py all` runs the configured generation/answer/export stages. Student training remains separate.

## Notebook distillation loop

[distillation.ipynb](distillation.ipynb) runs QLoRA SFT with ChatGPT (`gpt-6-luna`) and Qwen3-8B. ChatGPT assesses Qwen's drafts against the evidence and a source-only reference, then minimally corrects drafts with identified errors. Correct drafts and corrected targets must pass the grounding audit. Changes to conclusions require deterministic checks on the affected fields. Each round saves its assessments and quarantined drafts, and recurring error categories from fresh candidates guide the next round.

The bundled synthetic benchmark provides 432 development cases for checkpoint selection and 108 final cases from separate source families. After selection, the loop compares the initial and selected adapters on the holdout. See [the loop guide](DISTILLATION_LOOP.md) and [configuration](config/distillation_loop.json).

[distillation_kaggle_100.ipynb](distillation_kaggle_100.ipynb) runs the
1,000-pair Unsloth experiment on one Kaggle GPU. It collects 100 new training
pairs per round for ten rounds, with 60 broadly generated pairs and 40
student-screened pairs. Recurring training errors steer only the latter group
once feedback is available. It retains 100 fixed development cases and a
108-case holdout. The API budget can pause the run. Preserve the whole run
folder to continue collecting pairs in another session. Enable Internet and GPU access, attach the
matching `outputs/colab_distillation_bundle.zip` as a Kaggle Input or open the
notebook from the updated checkout, and set `OPENAI_API_KEY` in Kaggle Secrets.
Run `python3 tools/build_colab_bundle.py` to regenerate the ZIP after code changes.

A fresh run checks `j6antony/Health-Insurance-SLM` Releases, including adapter
prereleases, and initializes from the newest compatible adapter whose ZIP passes
SHA-256 verification. Add `GITHUB_TOKEN` with Contents read access for a private
repository. The notebook uses the released adapter's LoRA settings so it can
continue training older rank-16 adapters. Existing local run state takes
precedence. Release ZIPs carry weights and adapter configuration, so a
release-based run starts a new baseline, round history, and API budget.
Set `UPLOAD_ADAPTERS_TO_GITHUB = True` and provide Contents write access to
publish completed rounds for the next fresh Kaggle session.
