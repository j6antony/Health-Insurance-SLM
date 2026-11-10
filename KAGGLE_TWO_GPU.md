# Kaggle T4 x2 execution

Use the updated `distillation.ipynb` (or the updated fixed notebook) with Kaggle's
T4 x2 accelerator. This supersedes the old instruction to hide GPU 1. Restart the
kernel, expose `CUDA_VISIBLE_DEVICES=0,1` before torch imports, and run setup from
the top. The model setup cell must print two visible GPUs. Do not load a separate
`student` model in the notebook: workers need the GPU memory. The notebook's
`student` variable is `None`; the trained model is represented by its saved adapter
path (`result['best_checkpoint']`).

## What changes

- **Training:** `torchrun` starts two processes, each loading a 4-bit Qwen3-8B on its
  own local GPU. SFTTrainer uses DistributedDataParallel to synchronize trainable
  LoRA gradients. This avoids `torch.nn.DataParallel`, whose replication caused
  the earlier CUDA crash. Only the leader writes the adapter/tokenizer/metrics.
- **Same effective training batch:** one example per device × two devices × four
  accumulation steps = eight examples per optimizer update. The existing config's
  `gradient_accumulation_steps=8` is the global single-device equivalent and is
  divided between the workers. Epoch counts, completion-only loss, sequence length,
  learning rate, and optimizer-step cap remain unchanged.
- **Evaluation and diagnostics:** examples are split across both GPUs. Each worker
  sorts its prompts by length and generates in left-padded batches of two. An
  inference batch that runs out of memory retries one example at a time; a failure
  at batch size one is surfaced. EOS and truncation are evaluated per example, not
  from the padded batch length. Returned predictions are restored to original order.
- **Attention:** workers request PyTorch SDPA. The 1,024-token generation budget,
  task coverage, scoring formulas, and stopping thresholds are unchanged.
- **CPU coordinator:** teacher calls, verification, Supabase logging, budget checks,
  and decisions run only once in the notebook. GPU workers are not given teacher
  or Supabase secret keys. No duplicate API billing arises from two GPU ranks.

Models are loaded from the Hugging Face cache for each GPU phase, and workers exit
when that phase finishes. This has a loading cost, but frees their memory reliably
and isolates CUDA faults. It is not a persistent vLLM server or tensor-parallel
training. Each 16-GB GPU must fit its own quantized model and its activations;
the two cards do not act as one shared 32-GB device. Batched inference and DDP are
expected to reduce local compute time, but there is no measured speedup guarantee
until tested on your Kaggle hardware. Teacher API latency and repair overhead are
unchanged and may still dominate total wall-clock time.

## Test before a full run

1. Update both Python folders and the notebook. Preserve existing run output and
   budget files. Restart the kernel; run the model/tokenizer cell with two GPUs.
2. Run the preview/preflight cell. Both ranks must load and generate successfully.
   No teacher API calls occur in this check.
3. Run a smoke experiment (`TEST_RUN=True`) to verify DDP training and evaluation.
   Use a fresh smoke folder if you want an actual training test: a completed saved
   smoke run returns its existing result without exercising training again.
4. Review worker logs and `RUN_DIR/hardware_metrics.jsonl`: it records phase time
   INCLUDING model loading, worker count, batch size, and example count. Trainer's
   `training_metrics.json` separately records its training time and loss.
5. For a new full experiment, restart, set `TEST_RUN=False`, and use the full-run
   output folder. Saved adapters remain compatible for resuming existing experiments
   with the same config/model identity. Floating-point results need not match the
   old single-GPU execution bit for bit. An unfinished round still restarts from
   its last completed adapter checkpoint, as before.

Workers have a two-hour per-phase timeout (`DualGPURuntime(...,
 timeout_seconds=7200)`). Interrupting the notebook terminates the complete worker
process group; do not start another copy concurrently. Kaggle-friendly NCCL defaults
disable peer-to-peer and InfiniBand transports; override environment variables only
if you have confirmed a different topology works better. Memory, driver, NCCL, or
worker failures surface in the notebook; the coordinator does not silently fall
back to unsafe multi-GPU replication or accept a partial training checkpoint.

Implementation references: [PyTorch torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html)
and [PEFT quantized training](https://huggingface.co/docs/peft/en/developer_guides/quantization).
