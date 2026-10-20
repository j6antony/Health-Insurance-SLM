
from src.verification.cross_document import verify_claim_package


def make_document(doc_id, doc_type, fields):
    return {
        "document_id": doc_id,
        "classification": {"document_type": doc_type},
        "extraction": {
            "fields": [
                {
                    "field_name": name,
                    "value": value,
                    "page_number": 1,
                    "evidence": f"{name}: {value}",
                }
                for name, value in fields.items()
            ]
        }
    }


def test_cross_document():
    claim_form = make_document(
        "doc_001",
        "claim_form",
        {
            "patient_name": "Sarah Johnson",
            "policy_number": "HPL-78234",
            "hospital_name": "Lakeside Medical Center",
            "admission_date": "10/03/2026",
            "discharge_date": "10/07/2026",
        }
    )

    physician_statement = make_document(
        "doc_002",
        "physician_statement",
        {
            "patient_name": "Sarah Johnson",
            "hospital_name": "Lakeside Medical Center",
            "admission_date": "2026-10-03",
            "discharge_date": "10/08/2026",
        }
    )

    result = verify_claim_package([
        claim_form,
        physician_statement
    ])

    print("\n--- Cross-Document Verification ---")

    for item in result["results"]:
        print(
            f"{item['field_name']}: {item['status']}"
        )

        for observation in item["observations"]:
            print(
                f"  {observation['document_type']}: "
                f"{observation['value']}"
            )

    statuses = {
        item["field_name"]: item["status"]
        for item in result["results"]
    }

    assert statuses["patient_name"] == "consistent"
    assert statuses["hospital_name"] == "consistent"
    assert statuses["admission_date"] == "consistent"

    assert statuses["discharge_date"] == "conflict"

    assert statuses["policy_number"] == "insufficient_evidence"
    assert result["has_conflicts"] is True

    print("\nCross-document verification test passed!")


if __name__ == "__main__":
    test_cross_document()
