# Installed claims seed pack

The imported repository now uses the project's 72 claims seed examples at `data/seed_tasks.jsonl`. Both `scripts/generate_instructions.sh` and `scripts/prepare_for_finetuning.sh` already point to that location. The original upstream examples are preserved in `data/seed_tasks.upstream.jsonl`.

Installed supporting files:

- `data/seed_tasks.annotated.jsonl`: structured examples with task, skills, and review metadata.
- `data/task_schemas.json`: target-answer schemas for all 18 tasks.
- `data/TASKS.md`: exact task labels and definitions.
- `prompts/GENERATION_PROMPT.md`: custom instruction-and-scenario generation prompt.

The canonical source pack remains in `../../data/self_instruct/`, relative to this repository root. Installed files are copies; after editing the canonical pack, refresh the corresponding copies here.

## Stock scripts versus the custom workflow

Run stock scripts from this repository root (`self-instruct/self-instruct/` in the parent workspace) so their relative data paths resolve. The adapted bootstrapper groups seeds by their explicit `task` field and implements balanced sampling only. It uses seed instructions, not their example inputs/answers. Each request is scoped to one underrepresented task, and accepted records retain that requested task label and their parent seed IDs. This label records the requested task; the legacy filtering does not independently verify semantic task compliance. The preparation stage can read seed instances when `--include_seed_tasks` is enabled.

The custom generation prompt is installed for the planned task-aware pipeline; the stock bootstrapper does not load it. Its numbered-text response parser is incompatible with the JSON scenario array this prompt requests. Do not substitute the prompt without also adapting request construction, parsing, task metadata, schema validation, and the model backend. The stock scripts still specify the legacy `davinci` engine. No generation calls or training were run during installation.

All 72 seeds remain AI-authored and awaiting domain review. The stock fine-tuning preparation script includes them without checking review status; review them before using that export for training. Fictional code catalogs demonstrate grounding only, not real ICD/HCPCS knowledge.

## Balanced sampling

`self_instruct/bootstrap_instructions.py` uses `BalancedSeedSampler` by default; there is no random/targeted mode switch. Accepted counts start at zero, not at the number of seed examples, and are restored from `machine_generated_instructions.jsonl` on resume. Only instructions surviving the existing parser and ROUGE filter increase counts. Legacy generated files without task labels are rejected with an explanation; use a fresh batch directory or annotate them first.

Each request targets a task with the lowest accepted count. Temporary reservations spread requests in a batch without counting them as successful generations. Seed selection includes the target task, then prioritizes other groups with low accepted counts, breaking ties by prior seed exposure and seeded randomness. Full seed records are returned internally; the current numbered prompt consumes their instruction text. Up to two accepted instructions from the requested task are included as machine demonstrations. When there are fewer task groups than seed slots (including classification-only runs), the sampler uses fewer seed examples rather than duplicating a task group. Counts are feedback-driven; variable acceptance means exact equality is not guaranteed after every batch.

Preview one batch from this repository root without API/model dependencies:

```sh
python3 self_instruct/bootstrap_instructions.py \
  --seed_tasks_path data/seed_tasks.jsonl \
  --batch_dir /tmp/claims-balanced-preview \
  --num_prompt_instructions 8 \
  --request_batch_size 5 \
  --dry_run
```

Run the offline sampler checks:

```sh
python3 -m unittest discover -s tests -v
```

The JSON scenario prompt remains a separate next integration step. Balanced sampling does not activate it or replace the legacy inference backend.
