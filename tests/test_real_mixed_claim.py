
import json
import csv
from pathlib import Path

from docx import Document
from reportlab.pdfgen import canvas

from src.pipeline.mixed_claim_pipeline import (
    process_mixed_claim_package
)


INPUT_DIR = Path("data/raw/mixed_claim_test")
OUTPUT_DIR = Path("data/processed/mixed_claim_test")


def create_pdf(path):
    pdf = canvas.Canvas(str(path))

    lines = [
        "HOSPITAL BILL",
        "Patient Name: Alex Morgan",
        "Hospital Name: General Hospital",
        "Policy Number: POL123456",
        "Admission Date: 10/03/2026",
        "Discharge Date: 10/07/2026",
        "Total Hospital Charges: $8,700.00"
    ]

    y = 750

    for line in lines:
        pdf.drawString(60, y, line)
        y -= 30

    pdf.save()


def create_docx(path):
    document = Document()

    lines = [
        "DISCHARGE SUMMARY",
        "Patient Name: Alex Morgan",
        "Hospital Name: General Hospital",
        "Policy Number: POL123456",
        "Admission Date: 10/04/2026",
        "Discharge Date: 10/07/2026",
        "Diagnosis: Appendicitis"
    ]

    for line in lines:
        document.add_paragraph(line)

    document.save(path)


def create_csv(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "claim_id",
            "patient_name",
            "hospital_name",
            "policy_number",
            "admission_date",
            "discharge_date"
        ])

        writer.writerow([
            "CLM001",
            "Alex Morgan",
            "General Hospital",
            "POL123456",
            "10/03/2026",
            "10/07/2026"
        ])

        # A second claim to test isolation
        writer.writerow([
            "CLM999",
            "Jordan Lee",
            "City Hospital",
            "POL999999",
            "10/01/2026",
            "10/05/2026"
        ])


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = INPUT_DIR / "hospital_bill.pdf"
    docx_path = INPUT_DIR / "discharge_summary.docx"
    csv_path = INPUT_DIR / "claim_records.csv"

    create_pdf(pdf_path)
    create_docx(docx_path)
    create_csv(csv_path)

    print("\nProcessing mixed claim package...")

    result = process_mixed_claim_package(
        claim_id="CLM001",
        file_paths=[
            pdf_path,
            docx_path,
            csv_path
        ],
        output_dir=OUTPUT_DIR
    )

    verification = result["verification"]

    print("\n--- VERIFICATION RESULTS ---")

    for item in verification["results"]:
        print(
            f"{item['field_name']}: {item['status']}"
        )

        for observation in item["observations"]:
            print(
                f"  {observation['value']}"
                f" | Source: {observation['source_id']}"
            )

    print("\n--- REVIEW ISSUES ---")
    print(json.dumps(
        verification["review_issues"],
        indent=2
    ))

    print("\n--- REQUIREMENTS ---")
    print(json.dumps(
        result["requirements"],
        indent=2
    ))

    report_path = OUTPUT_DIR / "verification_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nReport saved: {report_path}")

    statuses = {
        item["field_name"]: item["status"]
        for item in verification["results"]
    }

    print("\n--- EXPECTED RESULTS ---")
    print("Patient name:", statuses["patient_name"])
    print("Admission date:", statuses["admission_date"])

    assert statuses["patient_name"] == "consistent"
    assert statuses["admission_date"] == "conflict"

    # Confirm that records from CLM999 were excluded.
    for item in verification["results"]:
        for observation in item["observations"]:
            assert observation["value"] not in {
                "Jordan Lee",
                "City Hospital",
                "POL999999"
            }

    print("\nReal mixed-format test passed!")


if __name__ == "__main__":
    main()

