
from unittest.mock import patch

import pytest

from src.pipeline.mixed_claim_pipeline import (
    process_mixed_claim_package,
)


def fake_preprocess(file_path, output_path=None):
    name = file_path.name

    if name.endswith(".csv"):
        return {
            "document_id": "csv001",
            "filename": name,
            "processing_status": "mapped",
            "quality": {"status": "passed"},
            "structured_mapping": {
                "records": [
                    {
                        "record_id": "row_2",
                        "identifiers": {
                            "claim_id": {"value": "CLM001"}
                        },
                        "fields": [{
                            "field_name": "patient_name",
                            "value": "Alex Morgan",
                            "validation_status": "valid",
                            "requires_review": False,
                            "source": {
                                "type": "csv",
                                "row": 2,
                                "column": "patient_name",
                            },
                        }],
                        "requires_review": False,
                    }
                ]
            },
        }

    return {
        "document_id": "pdf001",
        "filename": name,
        "classification": {
            "document_type": "hospital_bill",
            "requires_review": False,
        },
        "extraction": {
            "fields": [{
                "field_name": "patient_name",
                "value": "Alex Morgan",
                "page_number": 1,
                "evidence": "Patient Name: Alex Morgan",
                "requires_review": False,
            }]
        },
    }


@patch(
    "src.pipeline.mixed_claim_pipeline.evaluate_requirement",
    return_value={"status": "demonstration"},
)
@patch(
    "src.pipeline.mixed_claim_pipeline.preprocess_document",
    side_effect=fake_preprocess,
)
def test_mixed_pipeline(mock_preprocess, mock_requirements):
    result = process_mixed_claim_package(
        claim_id="CLM001",
        file_paths=[
            "hospital_bill.pdf",
            "claim_records.csv",
        ],
    )

    patient = next(
        item for item in result["verification"]["results"]
        if item["field_name"] == "patient_name"
    )

    assert patient["status"] == "consistent"
    assert result["verification"]["has_conflicts"] is False
    assert len(result["documents"]) == 1
    assert len(result["structured_documents"]) == 1
    assert result["requirements_status"] == "demonstration_only"


@patch(
    "src.pipeline.mixed_claim_pipeline.preprocess_document",
    side_effect=fake_preprocess,
)
def test_wrong_claim_id_rejected(mock_preprocess):
    with pytest.raises(ValueError, match="No records"):
        process_mixed_claim_package(
            claim_id="CLM999",
            file_paths=["claim_records.csv"],
        )

