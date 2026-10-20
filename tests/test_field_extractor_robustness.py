
from src.slm.field_extractor import extract_document_fields


TEXT = """
HOSPITALIZATION CLAIM RECORD

Document Reference: 975710
Form Revision: 08/2024
Date Prepared: 10/12/2026

Hospital: Lakeside Medical Center
Patient: Sarah Johnson

Diagnosis:
Acute Appendicitis

The patient was admitted on 10/03/2026.
The patient was discharged on 10/07/2026.

Billing Summary:
Room Charges: $3,000.00
Medication Charges: $450.00
Surgical Charges: $5,250.00
Total Hospital Charges: $8,700.00

Policy Number: HPL-78234

Physician Signature:
Not provided

Additional Notes:
The patient previously visited the hospital
on 09/15/2026 for a routine consultation.
"""


DOCUMENT = {
    "document_id": "synthetic_robustness_001",
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


EXPECTED = {
    "patient_name": "Sarah Johnson",
    "policy_number": "HPL-78234",
    "hospital_name": "Lakeside Medical Center",
    "admission_date": "10/03/2026",
    "discharge_date": "10/07/2026",
    "diagnosis": "Acute Appendicitis",
    "total_amount": "$8,700.00"
}

# We deliberately don't require currency because the document
# contains "$" but never explicitly states USD.
ALLOWED_FIELDS = set(EXPECTED) | {"currency"}


def test_robustness():
    result = extract_document_fields(DOCUMENT)

    print("\n--- Extracted Fields ---")

    for field in result["fields"]:
        print(
            f"{field['field_name']}: {field['value']} "
            f"(Review: {field['requires_review']})"
        )

    extracted = {}

    for field in result["fields"]:
        extracted.setdefault(
            field["field_name"], []
        ).append(field["value"])

    print("\n--- Expected Fields ---")

    failures = []

    for name, expected_value in EXPECTED.items():
        values = extracted.get(name, [])

        passed = values == [expected_value]

        print(
            f"{name}: "
            f"{'PASS' if passed else 'FAIL'} "
            f"| Expected: {expected_value!r} "
            f"| Actual: {values}"
        )

        if not passed:
            failures.append(name)

    print("\n--- False Positive Checks ---")

    forbidden_values = {
        "975710",
        "08/2024",
        "10/12/2026",
        "09/15/2026",
        "$3,000.00",
        "$450.00",
        "$5,250.00",
        "Not provided"
    }

    actual_values = [
        field["value"]
        for field in result["fields"]
    ]

    unexpected_fields = [
        field for field in result["fields"]
        if field["field_name"] not in ALLOWED_FIELDS
    ]

    forbidden_found = [
        value for value in actual_values
        if value in forbidden_values
    ]

    print("Forbidden values:", forbidden_found)
    print("Unexpected fields:", unexpected_fields)

    if forbidden_found:
        failures.append("forbidden_values")

    if unexpected_fields:
        failures.append("unexpected_fields")

    # Do not infer USD just because a dollar sign appears.
    currency_values = extracted.get("currency", [])

    if currency_values:
        print("Currency extracted:", currency_values)
        failures.append("unsupported_currency")

    assert not failures, (
        f"Robustness test failed: {failures}"
    )

    print("\nRobustness test passed!")


if __name__ == "__main__":
    test_robustness()
