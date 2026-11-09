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

The active parent is temporarily `gemini-3.5-flash`; the downstream student remains Qwen3-8B. The pipeline uses weighted balanced sampling, structured scenarios, separate teacher answers, automatic validation/audits, resumable storage, and SFT/chat JSONL exports. Human approval is optional; exported data is automatically checked synthetic silver, not independently validated gold.

Preview without an API key:

```sh
python3 run_self_instruct.py dry-run
```

With `GEMINI_API_KEY` set in the environment, `python3 run_self_instruct.py all` runs the configured generation/answer/export stages. Student training remains separate.

## Notebook distillation loop

[distillation.ipynb](distillation.ipynb) connects generation, student inference/cleaning, parent targets, QLoRA SFTTrainer, and Lighteval custom metrics. It prioritizes weak tasks from development scores and stops when all thresholds pass or after the configured maximum rounds (default 5). See [the loop guide](DISTILLATION_LOOP.md) and [loop configuration](config/distillation_loop.json). The notebook uses the existing loaded Qwen3-8B student and requires a CUDA GPU.
