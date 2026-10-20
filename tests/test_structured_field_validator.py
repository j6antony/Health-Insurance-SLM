
from src.verification.structured_field_validator import (
    validate_mapped_field,
    validate_structured_mapping
)


def make_field(name, value):
    return {
        "field_name": name,
        "value": value,
        "source": {"type": "csv", "row": 2},
        "requires_review": False
    }


def test_valid_date():
    valid, reason = validate_mapped_field(
        make_field("admission_date", "2026-10-20")
    )
    assert valid is True


def test_invalid_date():
    valid, reason = validate_mapped_field(
        make_field("admission_date", "2026-15-45")
    )
    assert valid is False
    assert reason == "invalid_date"


def test_invalid_amount():
    valid, reason = validate_mapped_field(
        make_field("total_amount", "abc")
    )
    assert valid is False
    assert reason == "invalid_amount"


def test_valid_amount():
    valid, reason = validate_mapped_field(
        make_field("total_amount", "$8,700.00")
    )
    assert valid is True


def test_duplicate_claim_id():
    def record(record_id):
        return {
            "record_id": record_id,
            "identifiers": {
                "claim_id": {"value": "CLM001"}
            },
            "fields": [],
            "issues": [],
            "requires_review": False
        }

    mapping = {
        "records": [record("row_2"), record("row_3")]
    }

    result = validate_structured_mapping(mapping)

    assert result["requires_review"] is True
    assert any(
        issue["type"] == "duplicate_claim_identifier"
        for issue in result["records"][1]["issues"]
    )


def test_invalid_field_requires_review():
    mapping = {
        "records": [{
            "record_id": "row_2",
            "identifiers": {},
            "fields": [
                make_field("total_amount", "invalid")
            ],
            "issues": [],
            "requires_review": False
        }]
    }

    result = validate_structured_mapping(mapping)

    assert result["requires_review"] is True
    assert (
        result["records"][0]["fields"][0]["validation_status"]
        == "invalid"
    )
