# Health Insurance Document Verification SLM — Consolidated Dataset

This archive consolidates the five uploaded Phase 2 archives and retains their dataset separation. It is an **OCR/model-development starting point**, not a trained model or a verified production claims dataset.

## Structure

- `data/raw/structured/`: original CMS synthetic claims, MIMIC demo ZIP, and NLM XML report archive.
- `data/raw/reference_pdfs/`: original Cigna, CMS RADV, and NAIC reference PDFs.
- `data/raw/mtsamples/`, `data/raw/harvard/`, `data/raw/noisy/`, `data/raw/benchmark/`: original additional datasets (including JPG and PDF assets).
- `data/processed/`: source-specific normalized JSONL and inventory outputs.
- `data/labeled/`: classification, extraction, synthetic verification labels and grouping data.
- `data/splits/`: train/validation/test JSONL partitions; preserve these as frozen test assignments.
- `src/ocr/`, `src/preprocessing/`, `src/labeling/`, `src/splitting/`, `src/extraction/`, `src/verification/`: workspace placeholders for Phase 3 source code.
- `tests/validate_dataset.py`: checks file references, counts, and known patient-group leakage.

## Portable file references

Generated JSONL records were rewritten from absolute `/Users/.../Downloads/new_clinical_collection/downloaded_datasets/...` paths to relative `data/raw/...` paths where applicable. Raw files remain unchanged. Always resolve image or PDF paths relative to the **project root**.

```python
from pathlib import Path
project_root = Path.cwd()  # run scripts from the project root
image = project_root / record['image_path']
```

## Quick checks

From the extracted `health-insurance-slm/` directory:

```bash
python3 tests/validate_dataset.py
```

## OCR starting point

Read `data/splits/noisy_images_train.jsonl`, then resolve each record's `image_path` relative to the project root. Keep validation/test files out of model training. Original images in `data/raw/noisy/bills/` and `data/raw/noisy/discharge_summaries/` are essential: JSONL alone is not sufficient for OCR.

## Important limitations

- The noisy document images are synthetic, with published labels not independently audited for clinical correctness.
- The linked PDF benchmark contains only five underlying simulated patients; a held-out patient is a smoke test, not a robust generalization estimate.
- The 24 insurance verification scenarios are rules-generated and not independent insurer-verified examples.
- MTSamples specialties are not confirmed document types.
- The Harvard summaries and NLM radiology reports have reuse restrictions; verify rights before model training or redistribution. The Cigna forms are proprietary reference materials.
- No PHI-sensitive client data, trained model, OCR inference results, or end-to-end claims adjudication system is provided.
- Any existing provenance or metadata may still refer to original local source locations in free-text descriptions; standardized path-valued fields were updated.

## Sectioning regression checks

Run from the project root (no Ollama required):

```bash
.venv/bin/python -m unittest discover -s tests/regression -v
```

Digital and OCR lines are ordered using whitespace gutters and horizontal bands before heading classification. Original extraction IDs and bounding boxes are retained; prediction IDs refer to the resulting page line order. The original digital `raw_text` remains unchanged.

Section boundaries validate model predictions against formatting, continuation text, form fields, and neighboring logical separators. Explicit numbered `SECTION` headings are retained even when the model misses them. Paragraphs and sections containing standalone `OR` alternatives stay together during chunking. These units may exceed `max_words`; `oversized` makes this explicit, and `word_count` includes the repeated heading. Heading-only sections also produce chunks.

To regenerate existing outputs with fresh model predictions, start Ollama with the configured Qwen model available, then run:

```bash
.venv/bin/python -m tests.test_batch_preprocessing
.venv/bin/python -m tests.inspect_preprocessing
```

The regressions cover the reported Cigna failures and pipeline output using deliberately incorrect heading predictions. They do not establish general accuracy on unseen layouts. Whitespace ordering is heuristic, section boundaries are conservative, and conditional groups spanning pages still require document-level handling. The extraction `quality.status` does not measure semantic sectioning accuracy.

## React + Vite frontend and FastAPI backend

Install backend dependencies from the project root:

```bash
.venv/bin/pip install -r requirements.txt
```

Start the FastAPI API in one terminal:

```bash
.venv/bin/python -m uvicorn src.web.server:app --host 127.0.0.1 --port 8000
```

Start React with Vite in another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://127.0.0.1:5173`). Vite proxies `/api` requests to FastAPI on port 8000. Ollama must be running with `qwen3:4b-instruct` available. Upload PDFs for one claim package and click **Verify package** to see the current hospitalization rule, extracted field evidence, review flags, and cross-document checks. JSON results can be downloaded.

The API accepts up to 10 PDFs totaling 30 MB, with up to 50 pages per PDF. One package processes at a time. Upload files are temporary; the latest ten job results remain in backend memory. The current classifier does not support every document type named in the requirement rule; the UI exposes this limitation. Documentation results are not claim approval decisions.

Build the frontend with `cd frontend && npm run build`. Run API tests with `.venv/bin/python -m unittest discover -s tests/regression -p test_web.py -v` from the project root.
