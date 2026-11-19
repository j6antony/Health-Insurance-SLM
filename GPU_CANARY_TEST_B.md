# Test B: continued-adapter GPU canary

Local package and real-tokenizer preflight passed on November 19, 2026. **GPU training has not run.** No provider, GPU, hourly rate or paid budget has been approved. This Mac is Apple silicon without CUDA; no connected cloud GPU execution tool is available.

Open `notebooks/qwen3_8b_continued_gpu_canary.ipynb` in Colab (File → Upload notebook) or Kaggle (Import Notebook). Run its cells in order. Do not allocate paid hardware until the provider, actual GPU, hourly rate and budget are approved. On free resources, explicitly confirm the allocation has no usage charge. Colab hardware availability is not guaranteed: [official FAQ](https://research.google.com/colaboratory/faq.html).

## Verified immutable inputs

- Original ZIP: `/Volumes/Extreme SSD/Codex Distillation Loop/gpu_jobs/packages/c03c14696b8ab4b394266f83d850641ae3e71f9eaa3cc749a34de12489834fc5/package.zip`.
- ZIP SHA-256: `19ec9e08f6bd85eb7ee585dbb62939db7c3fbcaa588f70f853b89b82a5751e2d`.
- Job ID: `c03c14696b8ab4b394266f83d850641ae3e71f9eaa3cc749a34de12489834fc5`.
- Dataset ID: `51cb0864daa7680acd3afd5d68d31276bac2289cc24bf7921cbc850e81252810`; exactly 64 new + 21 revalidated replay examples.
- Model/tokenizer: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Starting adapter: recovered legacy identity `e8e5edc7a75f82507db3c3fe0afca721728cd5f8abe03a80fa997139970fbc0e`; rank 32, RSLoRA, alpha 32, dropout 0.0.
- Adapter config SHA-256: `1fe1fdbfa6b611c63f9d7dc84d431f2b3685338d91678510f579b24ae002d5df`.
- Adapter weights SHA-256: `154c508775106e8ffe15e224c9f256131852d384e4efaea6af2bdf3d35e99c6f`.

All 67 packaged source files match the current corresponding repository files. Both dataset files and both adapter files pass checksum verification. The original archive, dataset and recovered adapter remain unchanged. The adapter config has `revision: null`: contrary to the earlier readiness report, its historical base revision cannot be proven from that file. The prepared job explicitly pins the revision used for this continuation.

The two-step configuration stays unchanged: 4-bit NF4/double quantization, FP16 compute, learning rate 0.00002, microbatch 1, accumulation 4, context 4,096, inference cap 1,536, seed 43. This engineering canary does not establish model-quality improvement.

## Local checks already executed

The original package was extracted to `/Volumes/Extreme SSD/Codex Distillation Loop/experiments/gpu-canary-20261119/test-b/extracted-original`. The unchanged worker was run from that directory with `--validate-only`, using the existing native CPU environment and cached pinned tokenizer. Transformers 4.57.3 also requested public Hugging Face metadata; an offline-only attempt was blocked by that metadata lookup, and enabling metadata access resolved it. No full model weights were requested.

| Check | Result |
| --- | --- |
| Final-text targets and evidence | 85 validated |
| Prompt / completion / total tokens | 75,462 / 26,716 / 102,178 |
| Sequence p50 / p95 / maximum | 1,077 / 1,866 / 1,978 |
| Overlength at 4,096 | 0 |
| Truncation | None |
| Non-thinking Qwen template | Passed |
| Completion masks and actual TRL padding/collator | Passed |
| Adapter settings, source and dataset integrity | Passed |

These token totals describe the dataset once, not exposure during an unexecuted two-step run. Full receipts and per-example lengths are saved under `experiments/gpu-canary-20261119/test-b/` on the SSD.

## Cloud workflow and authorization

The notebook embeds the runner and requires only the original ZIP upload. It does not assume access to the Mac SSD. Colab uses a file picker; Kaggle accepts the path of an attached private input ZIP.

Use Linux x86-64, Python 3.10–3.12, one visible CUDA GPU, compute capability 7.x–9.x, at least 22 GiB reported capacity (24 GB class), and at least 20 GiB free. The workflow stops on 16 GB hardware or newer architectures unsupported by the pinned CUDA 12.6 stack. Keep at least 40 GiB free disk.

An isolated virtual environment installs Torch 2.8.0, torchvision 0.23.0 and torchaudio 2.8.0 using the [official CUDA 12.6 wheel index](https://pytorch.org/get-started/previous-versions/), then the unchanged packaged requirements with additional compatibility constraints. The [bitsandbytes 0.48.2 CUDA support matrix](https://huggingface.co/docs/bitsandbytes/v0.48.2/installation) covers this CUDA generation. Installation, `pip check`, version capture and hardware checks run before training. This exact Linux/CUDA combination remains unexecuted until a cloud runtime is supplied; it is not represented as GPU tested.

Cloud preflight runs the original command from the extracted directory:

```bash
python -m autonomous_learning.gpu_worker . --validate-only
```

It runs in a separate process so CPU Accelerate state cannot contaminate CUDA training. Any failure stops progress and is logged.

The training cell asks for the actual provider, free/paid status, actual paid rate and budget, then requires a typed approval statement containing the detected GPU and exactly two steps. Paid mode retains the original worker entry point, equivalent to:

```bash
python -m autonomous_learning.gpu_worker . \
  --authorize-training --max-gpu-usd 1 \
  --gpu-hourly-rate-usd "$APPROVED_GPU_HOURLY_RATE"
```

The notebook invokes the worker in process to observe actual steps, both step losses, 4-bit loading, adapter restores and inference text without changing the trainer. It records peak PyTorch allocated/reserved VRAM, hardware, CUDA/library versions, runtime, cost, adapter hashes and exact failure tracebacks. It checks for exactly two steps with finite losses and refuses to rerun a previous attempt.

The original paid CLI rejects a zero hourly rate. For explicitly confirmed free allocations, the runner uses the **same unchanged worker core**, adds a 1,200-second watchdog and records actual usage charge $0. No equivalent commercial rate is invented. Paid mode retains the prepared maximum $1 worker-cost guard and tightens its watchdog to at most 1,200 seconds. Neither mode controls external provider billing or idle charges. Shut down paid hardware yourself. Paid compute-unit pricing without a known USD/hour rate is unsupported.

On success, the worker saves/reloads the adapter and generates one real inference response. The runner checks both restore bindings, stores the actual response and rechecks the initial package. A failed postcondition removes any importable success receipt. Export the ZIP/checksum after success **or failure**; diagnostic-only returns cannot be imported as completed canaries.

## Transfer and import on the Mac

Download `test-b-return.zip` and `test-b-return.zip.sha256` from the notebook into:

`/Volumes/Extreme SSD/Codex Distillation Loop/experiments/gpu-canary-20261119/test-b/returned/`

Verify/extract in Terminal:

```bash
cd '/Volumes/Extreme SSD/Codex Distillation Loop/experiments/gpu-canary-20261119/test-b/returned'
shasum -a 256 -c test-b-return.zip.sha256
```

Only after that checksum passes:

```bash
python3 - <<'PY'
from pathlib import Path
import json, math, zipfile
root = Path.cwd() / 'unpacked'
assert not root.exists(), 'Preserve the previous extraction'
with zipfile.ZipFile('test-b-return.zip') as z:
    assert len(z.namelist()) == len(set(z.namelist()))
    for entry in z.infolist():
        assert (root / entry.filename).resolve().is_relative_to(root.resolve())
        assert (entry.external_attr >> 16) & 0o170000 != 0o120000
    z.extractall(root)
observed = json.loads((root / 'diagnostics/training-observations.json').read_text())
assert observed['status'] == 'completed' and observed['infrastructure_canary_passed']
assert observed['optimizer_steps'] == 2 and observed['four_bit_model_loaded']
losses = [x['loss'] for x in observed['training_log_history'] if 'loss' in x]
assert len(losses) == 2 and all(math.isfinite(x) for x in losses)
print('Completed two-step result verified; ready for the existing importer.')
PY
```

Only after both commands succeed, use the existing importer with actual local paths:

```bash
cd '/Users/johanantony/Documents/ChatGPT/Codex Distillation Loop'
python3 -m autonomous_learning import-results \
  --root '/Volumes/Extreme SSD/Codex Distillation Loop' \
  --input '/Volumes/Extreme SSD/Codex Distillation Loop/experiments/gpu-canary-20261119/test-b/returned/unpacked/result' \
  --job '/Volumes/Extreme SSD/Codex Distillation Loop/gpu_jobs/packages/c03c14696b8ab4b394266f83d850641ae3e71f9eaa3cc749a34de12489834fc5/manifest.json'
```

The importer checks job/dataset association, bounded cost, reload/inference receipts and adapter hashes. Expected imported adapter path:

`/Volumes/Extreme SSD/Codex Distillation Loop/gpu_jobs/imported_results/c03c14696b8ab4b394266f83d850641ae3e71f9eaa3cc749a34de12489834fc5/adapter`

That checkpoint does not exist until a real return is imported. The complete returned directory remains on the SSD, including logs/metrics; the immutable importer keeps the adapter and result receipt. The original recovered adapter is untouched, and no promotion occurs.

## GPU report status

Provider/hardware, CUDA/PyTorch runtime versions, actual VRAM, full-model loading, actual steps/losses, save/reload, inference response, runtime/cost and imported checkpoint are **not yet measured**. The notebook captures these for the final report. Current GPU spending is $0. The training infrastructure is not proven until the live run and import succeed. A successful two-step run proves infrastructure only, not model improvement.
