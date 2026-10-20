
from pathlib import Path
from docx import Document

from src.preprocessing.pipeline import preprocess_document


SAMPLE = """HOSPITAL BILL
Patient Name: Alex Morgan
Hospital Name: General Hospital
Policy Number: POL123456
Admission Date: 10/03/2026
Discharge Date: 10/07/2026
Total Hospital Charges: $8,700.00
"""


def main():
    folder = Path("data/raw/synthetic_claims")
    folder.mkdir(parents=True, exist_ok=True)

    txt_path = folder / "test_hospital_bill.txt"
    txt_path.write_text(SAMPLE, encoding="utf-8")

    docx_path = folder / "test_hospital_bill.docx"
    word = Document()
    for line in SAMPLE.splitlines():
        if line.strip():
            word.add_paragraph(line)
    word.save(docx_path)

    for file_path in [txt_path, docx_path]:
        print(f"\nProcessing {file_path.name}")

        result = preprocess_document(
            file_path=file_path,
            output_path=(
                Path("data/processed")
                / f"{file_path.stem}_{file_path.suffix.lstrip('.')}_slm_result.json"
            )
        )

        classification = result["classification"]
        fields = result["extraction"]["fields"]

        print("Status:", result["processing_status"])
        print("Document type:", classification["document_type"])
        print("Review required:", classification["requires_review"])
        print("Evidence:", classification["evidence"])

        print("\nExtracted fields:")
        for field in fields:
            print(
                f"  {field['field_name']}: {field['value']}"
            )
            print("    Evidence:", field["evidence"])

        assert result["processing_status"] == "processed"
        assert classification["document_type"] == "hospital_bill"

        extracted = {
            field["field_name"]: field["value"]
            for field in fields
        }

        assert extracted.get("hospital_name") == "General Hospital"
        assert extracted.get("policy_number") == "POL123456"
        assert extracted.get("total_amount") == "$8,700.00"

    print("\nBoth formats passed the real SLM test!")


if __name__ == "__main__":
    main()
