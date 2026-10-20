
from src.slm.field_extractor import extract_document_fields


TEXT = """
HOSPITAL CLAIM FORM

Patient Name: Jane Smith
Policy Number: HPL-48291
Hospital Name: General Hospital
Admission Date: 10/01/2026
Discharge Date: 10/05/2026
Diagnosis: Acute Appendicitis
Total Amount: $4,500.00
Currency: USD
"""


document = {
    "document_id": "synthetic_test_001",
    "classification": {
        "document_type": "claim_form"
    },
    "pages": [
        {
            "page_number": 1,
            "normalized_text": TEXT
        }
    ]
}


def test_positive_extraction():
    result = extract_document_fields(document)

    expected = {
        "patient_name": "Jane Smith",
        "policy_number": "HPL-48291",
        "hospital_name": "General Hospital",
        "admission_date": "10/01/2026",
        "discharge_date": "10/05/2026",
        "diagnosis": "Acute Appendicitis",
        "total_amount": "$4,500.00",
        "currency": "USD"
    }

    extracted = {
        field["field_name"]: field["value"]
        for field in result["fields"]
    }

    print("\n--- Extracted Fields ---")

    for name, value in extracted.items():
        print(f"{name}: {value}")

    print("\n--- Validation ---")

    for name, expected_value in expected.items():
        actual = extracted.get(name)

        print(
            f"{name}: "
            f"{'PASS' if actual == expected_value else 'FAIL'}"
        )

        assert actual == expected_value, (
            f"{name}: expected {expected_value!r}, got {actual!r}"
        )

    print("\nPositive extraction test passed!")


if __name__ == "__main__":
    test_positive_extraction()
