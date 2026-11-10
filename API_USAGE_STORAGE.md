# Persistent API cost history with Supabase

Kaggle's working folder is temporary across sessions. A CSV in that folder is not
an independent backup. Supabase now stores one immutable usage row per completed
HTTP attempt. The CSV remains an Excel-readable export of that cloud history.
No prompts, generated answers, source documents, or API keys are uploaded.

## Setup once

1. Create a Supabase project dedicated to this experiment's usage ledger.
2. In its SQL Editor, paste and run `supabase/api_usage.sql`. This creates the
   private `api_usage_events` table and `api_usage_run_totals` cost-summary view.
3. Enable the Data API for the `public` schema/table if your project requires
   explicit exposure. The table has RLS enabled and no public/client policies.
4. In Supabase project settings, find the project URL and a **secret key**
   (`sb_secret_...`). A legacy service-role key is also supported. These are
   privileged server credentials: keep them in your private Kaggle Secrets,
   never a notebook cell, workbook, Git repository, or public notebook output.
5. In Kaggle Add-ons → Secrets, add both `SUPABASE_URL` and
   `SUPABASE_SECRET_KEY`, and enable access for this notebook. Keep
   `OPENAI_API_KEY` as before.
6. Upload the updated Python folders, `export_api_usage.py`, and notebook.
   Run the notebook from the beginning. Before a paid call, it should print
   `Supabase cost ledger connected`. The settings cell also uploads earlier local
   usage journals still available in the selected run directory.

If neither Supabase secret is set, the existing local-only mode still works and
prints that history is local only. Setting only one secret raises an error;
cloud mode never silently falls back when the connection fails.

## What is saved

The table includes run/event IDs, UTC timestamp, model, operation, status, token
counts, estimated USD cost, and a `details` JSON object containing cached/reasoning
counts, durations, attempt numbers, and the pricing rates/source/date. Null cost
means unknown, not free. `api_usage_run_totals` groups calls and known estimated
cost by run ID and shows the number of unknown-cost calls.

Each event's cost and rates are snapshotted when logged. Changing the pricing
code later does not silently reprice historical events. Earlier local rows that
lack snapshots are labeled `legacy_reconstructed` when imported; their estimates
use the rates available during import. Deleted historical logs cannot be recovered
by this integration, and these estimates are not the provider's billing statement.

## Reliability and failure behavior

Each event is first saved/fsynced to local JSONL, then uploaded. A stable event UUID
is the database primary key. Retries use conflict-ignore inserts: losing an upload
acknowledgment or repeating a sync does not double-count the call. The two standard
pipeline folders share one run UUID saved in `RUN_DIR/api_usage_run.json`.

If the upload fails, execution stops after saving the pending event locally; it
does not issue another teacher request to retry logging. Restore connectivity and
rerun. Pending uploads are retried automatically before more paid requests. Preserve
the run folder until sync succeeds. A session killed during an in-flight teacher
request can still incur an unreported charge; no post-response logger can recover
usage the provider never returned.

Cloud history survives losing the Kaggle working folder, provided the rows were
successfully uploaded and you retain your Supabase project. This does not back up
model checkpoints, generated datasets, or local API-budget state. Budgets remain
per-run safeguards, not an account-wide cloud spending cap.

## Excel/CSV export

The notebook's last cell exports **all runs** to
`RUN_DIR/api_usage_all_runs.csv`. Download it from Kaggle and open it in Excel.
It queries Supabase, so it also works in a fresh Kaggle session without the old
working folders. Unknown-cost rows remain blank and are counted in the summary.
Export after generation has stopped for a stable history report.

Without loading a model, with the same credentials in environment variables:

```bash
python export_api_usage.py --output outputs/api_usage_all_runs.csv
```

To sync an existing run's local history before exporting:

```bash
python export_api_usage.py --sync-dir /kaggle/working/distillation_smoke_openai --output /kaggle/working/api_usage_all_runs.csv
```

To export one run, pass `--run-id` using the UUID in `api_usage_run.json`.
In the Supabase SQL Editor you can also run:

```sql
select * from public.api_usage_run_totals order by first_call desc;
```

The integration uses the standard-library HTTP client; no Supabase SDK is required.
Implementation references: [Supabase REST API](https://supabase.com/docs/guides/api),
[API keys](https://supabase.com/docs/guides/getting-started/api-keys), and
[upsert semantics](https://supabase.com/docs/reference/javascript/upsert).
