# One-GPU Colab experiment with 100 evaluation questions

Open `distillation_colab_100.ipynb` in Google Colab. This is a separate notebook
from the Kaggle T4 x2 run; the Kaggle notebooks and their default settings are
unchanged. The Colab code/data ZIP is distributed with the notebook, so the run
does not depend on unpublished GitHub changes or an existing clone.

## Start

1. Upload `distillation_colab_100.ipynb` to Colab.
2. Select a GPU under Runtime → Change runtime type. Use a fresh session.
3. Run cells in order, authorize the Google Drive mount, and select the supplied
   `colab_distillation_bundle.zip` when the code-bundle cell asks for it.
4. Enable the `OPENAI_API_KEY` Colab Secret, or enter it in the hidden prompt.
   GitHub, Kaggle and Google Cloud service-account keys are not required.
5. If installation requests a restart, restart and rerun setup from the top.

The bundle SHA256 is embedded in the notebook and checked before extraction.
It includes required Python source, configuration, seed prompts and the complete
fixed evaluation corpus, but no model weights, environment files, credentials,
training outputs or Git repository. Models are downloaded anonymously into the
Colab VM's `/content/hf-cache`.

## Scale

| Setting | Colab default |
| --- | --- |
| Student | Qwen/Qwen3-8B, 4-bit NF4 with double quantization via Unsloth |
| GPUs | One visible CUDA device; explicit `device_map={"": 0}` |
| Fixed development cases | Exactly 100, 5 or 6 per task across all 18 tasks |
| Final holdout | 108 cases from two reserved source families per task |
| Subset selection | Seed 42; distinct case families preferred; frozen IDs/hash |
| Evaluation-data API generation | None |
| Training | Up to 5 rounds, 24 new pairs per round, cumulative replay |
| Verified targets | Independently generated teacher answers that pass source verification, with at least 25% deterministic-check coverage |
| Maximum distinct training pairs | 120 |
| Inference batch | 1 |
| Training batch | 1, accumulated over 4 steps |
| Sequence/completion limits | 3072 total / 1024 new tokens |
| LoRA | Rank 8, alpha 16 |
| Optimizer cap | 20 steps per round; actual steps can be lower |
| Separate API budget | 800 attempts / $2 estimated USD |

Unsloth accelerates student generation and QLoRA on the single GPU.
The run may end before five rounds if an existing stopping condition or the API budget is reached.

The complete 540-case corpus is excluded from curriculum generation, including the
108 holdout cases and the remaining development-family cases. The teacher
does not receive benchmark cases as demonstrations or correction inputs. The export
`evaluation_100.jsonl` and `evaluation_100_manifest.json` record the exact subset on
Drive. Changing the seed or data does not silently replace an existing run's set.

A five-round run evaluates the 100 questions at baseline and after each round
(up to 600 evaluation generations), plus diagnostic generations on training
scenarios. After selection, the initial and selected adapters each run on
the 108-case holdout. The automatically authored dataset
is synthetic; matching it does not establish clinical accuracy. API budgets may
stop generation before training targets are reached. GPU/Drive execution has not
been tested from this local workspace.

## Persistent saving and resume

Everything important is written under:

`/content/drive/MyDrive/HealthInsuranceSLM/colab_adaptive100_unsloth_v1`

The student-conditioned experiment starts in a new Drive folder. Use the earlier notebook and matching bundle to resume an earlier run.

This includes the initial and trained adapters, best-checkpoint pointers, loop
state, frozen training plans, generated data, evaluation predictions/scores,
API request usage and its persistent budget. Each completed training round also
creates an adapter ZIP under `drive_backups/`; archived bytes are read back and
compared with the per-file SHA256 manifest before a receipt is written. Archives
exclude credentials and do not contain Qwen's base weights.

Google Drive is the persistent destination; no bucket setup or cloud billing
configuration is required by this workflow. Drive access must be authorized in
your Colab session. Colab VMs are temporary, and mounted Drive I/O can fail or be
slow; the notebook checks real journal writes before loading a model or calling
the teacher. See [Colab's official storage/runtime FAQ](https://research.google.com/colaboratory/faq.html).

If Colab disconnects, start the same notebook again, mount the same Drive account,
upload the same bundle if needed, and keep the same `RUN_NAME`, settings and API
limits. The completed training checkpoints and input hashes are reused. An
interrupted optimizer phase may repeat that unfinished round; progress is not
saved after every optimizer step. The budget does not reset on rerun.

Process locks are kept on the Colab VM because mounted Drive may not implement
`flock`. They protect processes on the same VM, not two separate Colab sessions:
use this run folder from only one Colab session at a time. The simultaneous Kaggle
experiment uses a different filesystem/run folder and is unaffected.

If only ZIP export fails, use the last 'retry Drive ZIP export' cell; the saved
training journal prevents completed training from being repeated on resume. If
Drive is out of space or unavailable, restore Drive access before continuing.

## Maintainer rebuild

Run `python3 tools/build_colab_bundle.py` after changing a bundled source or data
file. It produces `outputs/colab_distillation_bundle.zip` and updates the notebook's
embedded checksum. Ship the notebook and ZIP together. Notebook Python-cell,
selection, persistence-lock, Drive archive and backup/resume behavior are tested
locally. Actual CUDA/Colab authorization remains a live-runtime check.

### Resuming after an API limit

Keep the same run folder. In the training cell, increase `API_MAX_CALLS` or
`API_MAX_COST_USD` as indicated by the error, then rerun that cell. These caps
are cumulative. Requests and costs already recorded still count against the new
limits. The notebook resumes from the state and checkpoints already on disk.
