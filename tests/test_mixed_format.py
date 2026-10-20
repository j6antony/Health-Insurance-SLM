
from src.verification.mixed_format import verify_mixed_claim


def make_document():
    return {
        "document_id": "pdf_001",
        "classification": {
            "document_type": "hospital_bill",
            "requires_review": False
        },
        "extraction": {
            "fields": [
                {
                    "field_name": "patient_name",
                    "value": "Alex Morgan",
                    "page_number": 1,
                    "evidence": "Patient Name: Alex Morgan",
                    "requires_review": False
                },
                {
                    "field_name": "admission_date",
                    "value": "10/03/2026",
                    "page_number": 1,
                    "evidence": "Admission Date: 10/03/2026",
                    "requires_review": False
                }
            ]
        }
    }


def make_structured_document(claim_id="CLM001"):
    return {
        "document_id": "csv_001",
        "quality": {"status": "passed"},
        "structured_mapping": {
            "records": [{
                "record_id": "csv_row_2",
                "identifiers": {
                    "claim_id": {"value": claim_id}
                },
                "requires_review": False,
                "fields": [
                    {
                        "field_name": "patient_name",
                        "value": "Alex Morgan",
                        "validation_status": "valid",
                        "requires_review": False,
                        "source": {
                            "type": "csv",
                            "row": 2,
                            "column": "patient_name"
                        }
                    },
                    {
                        "field_name": "admission_date",
                        "value": "2026-10-04",
                        "validation_status": "valid",
                        "requires_review": False,
                        "source": {
                            "type": "csv",
                            "row": 2,
                            "column": "admission_date"
                        }
                    }
                ]
            }]
        }
    }


def test_mixed_format_detects_conflict():
    result = verify_mixed_claim(
        claim_id="CLM001",
        documents=[make_document()],
        structured_documents=[make_structured_document()]
    )

    statuses = {
        item["field_name"]: item["status"]
        for item in result["results"]
    }

    assert statuses["patient_name"] == "consistent"
    assert statuses["admission_date"] == "conflict"
    assert result["requires_review"] is True


def test_different_claim_not_mixed():
    result = verify_mixed_claim(
        claim_id="CLM001",
        documents=[make_document()],
        structured_documents=[
            make_structured_document("CLM999")
        ]
    )

    statuses = {
        item["field_name"]: item["status"]
        for item in result["results"]
    }

    assert statuses["patient_name"] == "insufficient_evidence"
    assert statuses["admission_date"] == "insufficient_evidence"


def test_source_provenance_preserved():
    result = verify_mixed_claim(
        claim_id="CLM001",
        documents=[make_document()],
        structured_documents=[make_structured_document()]
    )

    admission = next(
        item for item in result["results"]
        if item["field_name"] == "admission_date"
    )

    csv_observation = next(
        item for item in admission["observations"]
        if item["source"]["type"] == "csv"
    )

    assert csv_observation["source"]["row"] == 2
    assert csv_observation["source"]["column"] == "admission_date"


def test_invalid_field_not_used():
    structured = make_structured_document()

    field = structured["structured_mapping"]["records"][0]["fields"][0]
    field["validation_status"] = "invalid"

    result = verify_mixed_claim(
        claim_id="CLM001",
        documents=[make_document()],
        structured_documents=[structured]
    )

    patient = next(
        item for item in result["results"]
        if item["field_name"] == "patient_name"
    )

    assert patient["status"] == "insufficient_evidence"
    assert result["requires_review"] is True
