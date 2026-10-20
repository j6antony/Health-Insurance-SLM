
from src.verification.structured_quality import (
    validate_csv,
    validate_excel,
    validate_json,
    validate_text,
    validate_structured_document
)

from src.preprocessing.pipeline import preprocess_document


def test_csv_missing_value():
    data = {
        "columns": ["claim_id", "hospital"],
        "rows": [
            {
                "row_number": 2,
                "values": {
                    "claim_id": "CLM001",
                    "hospital": ""
                }
            }
        ]
    }

    issues = validate_csv(data)

    assert any(
        issue["type"] == "missing_value"
        for issue in issues
    )


def test_csv_duplicate_record():
    data = {
        "columns": ["claim_id"],
        "rows": [
            {"row_number": 2, "values": {"claim_id": "A"}},
            {"row_number": 3, "values": {"claim_id": "A"}}
        ]
    }

    issues = validate_csv(data)

    assert any(
        issue["type"] == "duplicate_row"
        for issue in issues
    )


def test_excel_missing_header():
    data = {
        "sheets": [{
            "sheet_name": "Claims",
            "rows": [
                {"row_number": 1, "values": ["claim_id", None]},
                {"row_number": 2, "values": ["CLM001", 4500]}
            ]
        }]
    }

    issues = validate_excel(data)

    assert any(
        issue["type"] == "missing_header"
        for issue in issues
    )


def test_empty_json():
    issues = validate_json({})

    assert len(issues) == 1
    assert issues[0]["type"] == "empty_json"


def test_empty_text():
    issues = validate_text("   ")

    assert issues[0]["type"] == "empty_document"


def test_valid_csv():
    data = {
        "columns": ["claim_id", "amount"],
        "rows": [{
            "row_number": 2,
            "values": {"claim_id": "CLM001", "amount": "4500"}
        }]
    }

    assert validate_csv(data) == []


def test_pipeline_quality_integration(tmp_path):
    file_path = tmp_path / "invalid_claims.csv"

    file_path.write_text(
        "claim_id,hospital,amount\n"
        "CLM001,Hospital A,4500\n"
        "CLM002,,3200\n"
        "CLM001,Hospital A,4500\n",
        encoding="utf-8"
    )

    result = preprocess_document(file_path)

    assert result["processing_status"] == "ingested"
    assert result["structured_quality"]["status"] == "review_required"
    assert result["quality"]["status"] == "review_required"

    issue_types = {
        issue["type"]
        for issue in result["structured_quality"]["issues"]
    }

    assert "missing_value" in issue_types
    assert "duplicate_row" in issue_types


def test_valid_json_document():
    document = {
        "file_type": "json",
        "structured_data": {"claim_id": "CLM001"},
        "pages": [{"text": '{"claim_id": "CLM001"}'}]
    }

    result = validate_structured_document(document)

    assert result["status"] == "passed"
    assert result["issue_count"] == 0
