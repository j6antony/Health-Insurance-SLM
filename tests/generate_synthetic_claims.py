
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


OUTPUT_DIR = Path("data/raw/synthetic_claims")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_pdf(filename, title, lines):
    path = OUTPUT_DIR / filename

    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, 750, title)

    pdf.setFont("Helvetica", 11)

    y = 710

    for line in lines:
        pdf.drawString(50, y, line)
        y -= 24

    pdf.save()

    print("Created:", path)


def main():
    create_pdf(
        "claim_form.pdf",
        "HOSPITAL CARE CLAIM FORM",
        [
            "Patient Name: Sarah Johnson",
            "Policy Number: HPL-78234",
            "Hospital Name: Lakeside Medical Center",
            "Admission Date: 10/03/2026",
            "Discharge Date: 10/07/2026",
            "Diagnosis: Acute Appendicitis",
            "Total Hospital Charges: $8,700.00",
        ]
    )

    create_pdf(
        "physician_statement.pdf",
        "PHYSICIAN STATEMENT",
        [
            "Patient Name: Sarah Johnson",
            "Hospital Name: Lakeside Medical Center",
            "Admission Date: 2026-10-03",
            "Discharge Date: 10/08/2026",
            "Diagnosis: Acute Appendicitis",
            "Attending Physician: Dr. Emily Carter",
        ]
    )


if __name__ == "__main__":
    main()
