
# Health Insurance Claims Document Intelligence POC

A local, explainable proof of concept for processing health-insurance claim documents using a Small Language Model (SLM).

The system extracts information from PDFs, reconstructs document structure, classifies documents, extracts relevant claim fields, and verifies consistency and documentation requirements across multiple documents.

The goal is to assist insurance claims handlers by identifying missing information, conflicting details, and supporting evidence.

**Scope:** This system supports claims-document review. It does not automatically approve, deny, or adjudicate insurance claims.

## 1. Current Capabilities

| Component | Status |
|---|---|
| Digital PDF extraction | Implemented |
| Tesseract OCR fallback | Implemented |
| Text normalization | Implemented |
| Layout and paragraph extraction | Implemented |
| Extraction quality validation | Implemented |
| SLM heading classification | Implemented |
| Section segmentation | Implemented |
| Section-aware chunking | Implemented |
| Document classification | Implemented |
| Structured field extraction | Implemented |
| Evidence recovery | Implemented |
| Deterministic field validation | Implemented |
| Cross-document verification | Implemented |
| AND/OR requirement evaluation | Implemented |
| Document-scoped requirement verification | Implemented |
| Complete claim-package pipeline | Implemented |
| Requirement-level evidence tracking | In progress |
| Policy-specific rule extraction | Planned |

## 2. System Architecture

```text
INPUT: Claim Package (Multiple PDFs)
                 |
                 v
        DOCUMENT PREPROCESSING
        ----------------------
        Digital PDF Extraction
        OCR Fallback (Tesseract)
        Text Normalization
        Layout Reconstruction
        Quality Validation
                 |
                 v
        DOCUMENT SEGMENTATION
        ---------------------
        SLM Heading Detection
        Section Identification
        Section-Aware Chunking
                 |
                 v
        SLM DOCUMENT INTELLIGENCE
        -------------------------
        Document Classification
        Structured Field Extraction
        Evidence Recovery
        Field Validation
                 |
                 v
        CLAIM PACKAGE VERIFICATION
        --------------------------
        Cross-Document Comparisons
        Document Requirements
        Field Requirements
        AND/OR Rule Evaluation
                 |
                 v
        STRUCTURED VERIFICATION REPORT
        ------------------------------
        Extracted Fields
        Source Evidence
        Consistent Information
        Conflicting Information
        Missing Documentation
        Review Requirements
```

The SLM handles tasks requiring document understanding, while deterministic Python logic handles validation, comparisons, and rule evaluation.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core processing and verification |
| PyMuPDF | PDF text and layout extraction |
| Tesseract / pytesseract | OCR |
| Ollama | Local model inference |
| Qwen3 4B Instruct | Document understanding and extraction |
| Pydantic | Structured SLM responses |
| ReportLab | Generating synthetic test PDFs |
| JSON | Structured document and verification outputs |

The current local model is:

`qwen3:4b-instruct`

## 4. Project Structure

```text
health-insurance-slm/
├── data/
│   ├── raw/
│   │   ├── reference_pdfs/
│   │   └── synthetic_claims/
│   └── processed/
│       ├── batch/
│       ├── claim_package_test/
│       └── synthetic_claim_package/
│
├── src/
│   ├── extraction/
│   │   ├── pdf_extractor.py
│   │   └── digital_layout.py
│   │
│   ├── ocr/
│   │   └── tesseract_ocr.py
│   │
│   ├── preprocessing/
│   │   ├── coordinates.py
│   │   ├── document_builder.py
│   │   ├── text_normalizer.py
│   │   ├── section_detector.py
│   │   ├── section_chunker.py
│   │   └── pipeline.py
│   │
│   ├── slm/
│   │   ├── heading_classifier.py
│   │   ├── document_classifier.py
│   │   └── field_extractor.py
│   │
│   ├── verification/
│   │   ├── extraction_quality.py
│   │   ├── evidence_matcher.py
│   │   ├── field_validator.py
│   │   ├── cross_document.py
│   │   ├── requirements.py
│   │   └── requirement_rules.py
│   │
│   └── pipeline/
│       └── claim_pipeline.py
│
└── tests/
    ├── test_preprocessing_pipeline.py
    ├── test_batch_preprocessing.py
    ├── test_document_classifier.py
    ├── test_field_extractor.py
    ├── test_field_extractor_positive.py
    ├── test_field_extractor_robustness.py
    ├── test_cross_document.py
    ├── test_claim_pipeline.py
    ├── generate_synthetic_claims.py
    ├── test_synthetic_claim_packages.py
    ├── test_requirements.py
    ├── test_requirement_fields.py
    ├── test_requirement_documents.py
    └── test_document_scoped_requirements.py
```

## 5. Setup

### Prerequisites

- Python 3
- Tesseract OCR
- Ollama
- Qwen3 4B Instruct

Download the model:

```bash
ollama pull qwen3:4b-instruct
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the core dependencies:

```bash
pip install pymupdf pytesseract pillow ollama pydantic reportlab
```

Ensure Ollama is running locally at:

`http://localhost:11434`

Use the repository's dependency manifest, if available, to reproduce the complete development environment.

## 6. Document Preprocessing

The preprocessing pipeline converts raw PDFs into structured JSON.

### Processing Steps

1. Detect whether the PDF contains extractable digital text.
2. Apply Tesseract OCR when necessary.
3. Normalize extracted text.
4. Extract layout information, paragraphs, and bounding boxes.
5. Evaluate extraction quality.
6. Identify headings using the SLM.
7. Segment the document into sections.
8. Generate section-aware chunks.

### Run Preprocessing

```bash
python -m tests.test_preprocessing_pipeline
```

### Run Batch Processing

```bash
python -m tests.test_batch_preprocessing
```

### Inspect Segmentation Results

```bash
python -m tests.inspect_preprocessing
```

The pipeline preserves document metadata, page references, source line information, and extracted content.

## 7. Document Classification

The SLM classifies documents based on their primary purpose.

### Supported Categories

- `claim_form`
- `hospital_bill`
- `physician_statement`
- `discharge_summary`
- `investigation_report`
- `prescription`
- `claims_checklist`
- `policy_document`
- `other`
- `unknown`

The classifier uses structured JSON responses with supporting evidence and a review flag.

Document titles are considered alongside the document's content to avoid confusing claim forms containing charges with hospital bills.

### Run Classification Test

```bash
python -m tests.test_document_classifier
```

## 8. Structured Field Extraction

The SLM extracts populated claim information from documents.

### Supported Fields

| Field | Description |
|---|---|
| patient_name | Name of the patient |
| policy_number | Insurance policy identifier |
| hospital_name | Treating hospital or facility |
| admission_date | Hospital admission date |
| discharge_date | Hospital discharge date |
| diagnosis | Documented medical diagnosis |
| total_amount | Total monetary amount |
| currency | Explicitly identified currency |

Field extraction is currently performed page by page.

Each accepted field contains:

- Field name
- Extracted value
- Supporting evidence
- Page number
- Review flag

### Evidence Recovery

If the SLM identifies a value but produces incorrectly formatted evidence, the evidence matcher attempts to recover an exact source excerpt.

Recovery only occurs when the value can be uniquely located in the source text.

### Field Validation

The deterministic validator rejects unsupported candidates, including:

- Empty values and placeholders
- Empty form labels mistaken for populated values
- Invalid date formats
- Invalid monetary amounts
- Multiline collections of form labels mistaken for names
- Policy identifiers without sufficient supporting context

Rejected fields are currently logged to the terminal.

### Run Extraction Tests

```bash
python -m tests.test_field_extractor
python -m tests.test_field_extractor_positive
python -m tests.test_field_extractor_robustness
```

## 9. Cross-Document Verification

The system compares extracted fields from different documents belonging to the same claim.

### Compared Fields

- Patient name
- Policy number
- Hospital name
- Admission date
- Discharge date

Date values are normalized before comparison.

### Verification Statuses

| Status | Meaning |
|---|---|
| `consistent` | At least two distinct documents contain matching normalized values |
| `conflict` | At least two distinct documents contain differing values |
| `insufficient_evidence` | Fewer than two distinct documents provide the field |

For example:

```text
Claim Form:
Discharge Date = 10/07/2026

Physician Statement:
Discharge Date = 10/08/2026

Result:
conflict
```

A conflict indicates that the information requires review. It does not automatically mean that a claim should be rejected.

### Run Verification Test

```bash
python -m tests.test_cross_document
```

## 10. Documentation Requirement Verification

The system also checks whether a claim package contains required supporting documentation.

This is separate from cross-document consistency verification.

### Supported Requirement Operations

| Operation | Purpose |
|---|---|
| `document` | Verify a document type is present |
| `field` | Verify a field is present |
| `document_with_fields` | Verify one document contains specific fields |
| `AND` | Require all conditions |
| `OR` | Require at least one condition |

### Document-Scoped Requirements

Fields must belong to the appropriate document when a requirement specifies `document_with_fields`.

For example, a diagnosis found in a hospital bill cannot satisfy a requirement for a physician statement containing a diagnosis.

### Current Demonstration Rule

The initial hospitalization rule supports these alternatives:

```text
UB-04 Form

OR

Physician Statement
    AND
Diagnosis Field

OR

Room-and-Board Documentation
    AND
Medical Codes Documentation
```

This is a simplified demonstration rule, not a complete or authoritative representation of Cigna's coverage or documentation requirements.

Rules are currently defined in:

`src/verification/requirement_rules.py`

The evaluation engine is located in:

`src/verification/requirements.py`

### Run Requirement Tests

```bash
python -m tests.test_requirements
python -m tests.test_requirement_fields
python -m tests.test_document_scoped_requirements
```

## 11. Complete Claim Package Pipeline

The claim-package pipeline combines all the implemented components.

### Processing Flow

1. Receive multiple PDF documents.
2. Preprocess each PDF.
3. Classify each document.
4. Extract and validate structured fields.
5. Compare shared fields across documents.
6. Evaluate documentation requirements.
7. Return a structured verification report.

### Example Usage

```python
from src.pipeline.claim_pipeline import process_claim_package

result = process_claim_package(
    file_paths=[
        "data/raw/synthetic_claims/claim_form.pdf",
        "data/raw/synthetic_claims/physician_statement.pdf"
    ],
    output_dir="data/processed/synthetic_claim_package"
)

print(result["verification"])
print(result["requirements"])
```

**Assumption:** All supplied documents belong to the same claim. Automated claim-package association has not yet been implemented.

### Run Full Integration Test

```bash
python -m tests.test_claim_pipeline
```

### Run Synthetic Claim Package Test

```bash
python -m tests.generate_synthetic_claims
python -m tests.test_synthetic_claim_packages
```

## 12. Testing Results

### Reference Document Preprocessing

| Document | Pages | Chunks |
|---|---:|---:|
| Cigna Hospital Care Checklist | 1 | 5 |
| Cigna Hospital Care Claim Form | 7 | 32 |
| Cigna Physician Statement | 4 | 20 |
| NAIC Claims Guide | 2 | 3 |
| **Total** | **14** | **60** |

All four reference PDFs completed batch preprocessing successfully.

Chunk counts may vary after changes to segmentation, chunking parameters, or SLM predictions.

### Field Extraction Tests

**Simple synthetic claim**

- 8/8 expected fields extracted correctly.

**Robustness test**

- 7/7 expected fields extracted correctly.
- No forbidden values accepted.
- Unsupported currency values rejected.

**Blank reference forms**

- Cigna claim form: 0 accepted populated fields.
- Cigna physician statement: 0 accepted populated fields.

### Synthetic Claim Package

Two synthetic PDF documents were generated:

- Hospital claim form
- Physician statement

Both contained the same patient and hospital information, but intentionally conflicting discharge dates.

### Verification Results

| Field | Result |
|---|---|
| Patient name | Consistent |
| Hospital name | Consistent |
| Admission date | Consistent |
| Discharge date | Conflict |
| Policy number | Insufficient evidence |

The classifier correctly identified both synthetic document types after prompt refinement.

### Requirement Verification

The demonstration requirement was satisfied because the synthetic physician statement contained a populated diagnosis field.

All document-scoped requirement tests passed.

These are functional POC test results, not statistically representative accuracy measurements.

## 13. Known Limitations

### Document Processing

- Multi-column reading order is not fully solved.
- Section boundaries may occasionally be incorrect.
- Cross-page section continuity is not yet supported.
- Complex conditional documentation requirements may be split during chunking.

### Field Extraction

- Evidence matching confirms textual presence, not necessarily semantic correctness.
- Unconventional field formats may be rejected.
- Page-by-page extraction may miss cross-page context.
- Rejected candidates are not yet stored in a structured audit report.

### Classification

- Some document subtypes are not independently recognized.
- A generic hospital bill is not necessarily a UB-04 form.
- More labeled examples are required to measure classification reliability.

### Requirement Verification

- Rules are currently manually defined.
- Insurer-specific policy versions and effective dates are not integrated.
- Requirement applicability conditions are not fully represented.
- Requirement-level evidence tracking is being developed.
- Requirement satisfaction does not establish medical necessity or insurance coverage.

### Production Readiness

- Real claim-package membership validation is not implemented.
- Production authentication and authorization are not implemented.
- Privacy and security controls require further development.
- Human review workflows are not yet implemented.
- Broader evaluation on representative authorized datasets is required.

## 14. Next Steps

1. Complete requirement-level evidence tracking with document IDs, page numbers, and supporting excerpts.
2. Improve document-subtype detection, including UB-04 identification.
3. Extract documentation requirements from insurer checklists and policy documents.
4. Store rules with insurer identifiers, policy versions, effective dates, and source evidence.
5. Implement claim-specific applicability conditions.
6. Expand structured extraction for hospitalization, surgery, and medicine coverage verification.
7. Create a labeled evaluation dataset and measure classification, extraction, and verification performance.
8. Add structured audit logging and human-review workflows.

## 15. Project Principles

**Evidence First**

Every extracted field and verification result should be traceable to supporting document evidence.

**SLM for Understanding, Rules for Verification**

Use the SLM to interpret unstructured documents and extract information. Use deterministic logic to evaluate explicit requirements and detect conflicts.

**Insurer-Independent Core**

Keep generic preprocessing and validation logic separate from insurer-specific policies and exceptions.

**Human-in-the-Loop**

The system assists claims handlers by surfacing evidence, inconsistencies, and missing requirements. It does not make autonomous claim approval or denial decisions.
