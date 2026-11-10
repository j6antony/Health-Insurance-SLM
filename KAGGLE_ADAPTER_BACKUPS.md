# Automatic private adapter datasets

Enable `BACKUP_ADAPTERS_TO_KAGGLE = True` in the notebook credentials cell (the new
default). Add these Kaggle Secrets and enable access for the notebook:

- `KAGGLE_USERNAME`: your Kaggle account name, used as the dataset owner.
- `KAGGLE_API_TOKEN`: a personal API token from Kaggle account settings.
- Alternatively, use legacy `KAGGLE_USERNAME` + `KAGGLE_KEY` credentials.

The updated requirements include the Kaggle CLI. The setup cell checks authentication
before new teacher calls. The private backup prefix is set to
`your-username/health-insurance-slm-adapters`; you can change it through the
`KAGGLE_ADAPTER_DATASET` environment variable before running the loop. With this
variable unset, Python/CLI users retain local-only checkpoint behavior.

After EVERY successful training round, before evaluation, the CPU coordinator:

1. Saves the adapter locally and writes `round_XX/training_complete.json`.
2. Packages only the adapter weights/config and tokenizer files into `adapter.zip`.
3. Creates a new **private Kaggle dataset**, with a unique run/round/content-hash ID.
4. Confirms that its manifest is readable remotely and records the URL under
   `RUN_DIR/adapter_backups/round_XX.json`.
5. Continues with evaluation and the next round.

Each round has a separate dataset; earlier adapters are never replaced or deleted.
The new dataset uses original-authors copyright metadata; the adapter's and base
model's applicable terms still apply. The base model itself is NOT uploaded.
Training data, secrets, the entire working directory, and API-budget state are
NOT in the adapter archive. Publishing remains private; the code never passes
Kaggle's `--public` option. Do not manually make backup datasets public if you want
subsequent recovery checks to remain private.

A Kaggle dataset is persistent remote storage. `/kaggle/input` is read-only inside
a running notebook, so the adapter must first be written in `/kaggle/working` and
then uploaded using the Kaggle API. This cannot bypass local staging. Attach the
saved dataset in a future notebook and unzip `adapter.zip`, then load its contents
with the original base model using PEFT.

Uploads require Internet access and sufficient Kaggle dataset/storage quota. If
an upload fails or remains processing past the bounded verification wait, execution
stops with the adapter retained locally. Rerun using the same folder/settings:
the new training-complete marker restores the finished adapter and retries upload
and evaluation WITHOUT training the round again. Remote manifests and file hashes
avoid creating another dataset if an earlier upload succeeded but its acknowledgment
was lost. The dataset receipt is an acknowledgment of its readable manifest, not
an independent download-and-hash verification of all adapter bytes.

Only successful new training rounds invoke this upload hook. For a previously
completed round, you can explicitly upload an existing adapter without retraining:

```python
from distillation_loop.kaggle_backup import save_adapter_dataset
print(save_adapter_dataset(
    RUN_DIR / 'round_01/sft/adapter', RUN_DIR, round_id=1,
))
```

Keep full-run outputs separately if you want the datasets, evaluation results,
optimizer-independent round state, and budget journals preserved too. Supabase
still only stores API usage. Restarted training before a `training_complete.json`
marker was saved still starts from the previous completed round's adapter.

Reference: [Official Kaggle dataset commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).
