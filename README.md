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
