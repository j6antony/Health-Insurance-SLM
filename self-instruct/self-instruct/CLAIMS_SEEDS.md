# Active claims pipeline

The 72 installed seed examples remain at `data/seed_tasks.jsonl`, with annotated seeds, task schemas and definitions alongside them. Original upstream seeds are preserved at `data/seed_tasks.upstream.jsonl`. Canonical copies live at `../../data/self_instruct/`.

The active automated implementation is now `../../self_instruct_pipeline/`, run by `../../run_self_instruct.py`. It uses GPT-6 Luna through the OpenAI Responses API as the parent, weighted balanced task quotas, structured scenarios, separate teacher answers, automated audits and Qwen3-8B SFT exports. See `../../SELF_INSTRUCT.md` for configuration, output contracts, tests, and limitations.

`self_instruct/bootstrap_instructions.py` is a compatibility entry point to the same runner. It maps common legacy path/count flags; `--dry_run` previews the structured request without API calls. The generation/answer/export shell scripts also forward to the unified runner. The legacy GPT-3 classification and fine-tuning shell stages are disabled. The old numbered-list pipeline is no longer active.

No human review is mandatory, but data passing automatic checks remains synthetic silver, not independently validated gold. The raw upstream preparation Python file remains reference-only and should not be used to bypass the active answer validation/export stage.
