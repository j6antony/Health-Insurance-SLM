
import json
import csv

from openpyxl import Workbook

from src.preprocessing.pipeline import preprocess_document


def test_csv_mapping(tmp_path):
    path = tmp_path / "claims.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "claim_id", "patient_name",
            "policy_number", "total_amount"
        ])
        writer.writerow([
            "CLM001", "Alex Morgan", "POL123", "8700"
        ])
        writer.writerow([
            "CLM002", "Jordan Lee", "POL456", "3200"
        ])

    result = preprocess_document(path)
    mapping = result["structured_mapping"]

    assert result["processing_status"] == "mapped"
    assert mapping["record_count"] == 2

    first = mapping["records"][0]
    second = mapping["records"][1]

    assert first["identifiers"]["claim_id"]["value"] == "CLM001"
    assert second["identifiers"]["claim_id"]["value"] == "CLM002"

    fields = {
        f["field_name"]: f
        for f in first["fields"]
    }

    assert fields["patient_name"]["value"] == "Alex Morgan"
    assert fields["total_amount"]["value"] == "8700"
    assert fields["patient_name"]["source"]["row"] == 2
    assert fields["patient_name"]["source"]["column"] == "patient_name"


def test_excel_mapping(tmp_path):
    path = tmp_path / "claims.xlsx"

    wb = Workbook()
    ws = wb.active
    ws.title = "Claims"

    ws.append(["claim_id", "hospital_name", "total_amount"])
    ws.append(["CLM001", "General Hospital", 8700])

    wb.save(path)
    wb.close()

    result = preprocess_document(path)
    mapping = result["structured_mapping"]

    assert mapping["record_count"] == 1

    fields = {
        f["field_name"]: f
        for f in mapping["records"][0]["fields"]
    }

    assert fields["hospital_name"]["value"] == "General Hospital"
    assert fields["hospital_name"]["source"]["cell"] == "B2"


def test_json_mapping(tmp_path):
    path = tmp_path / "claims.json"

    data = {
        "claims": [
            {
                "claim_id": "CLM001",
                "patient_name": "Alex Morgan",
                "total_amount": 8700
            },
            {
                "claim_id": "CLM002",
                "patient_name": "Jordan Lee",
                "total_amount": 3200
            }
        ]
    }

    path.write_text(json.dumps(data), encoding="utf-8")

    result = preprocess_document(path)
    mapping = result["structured_mapping"]

    assert mapping["record_count"] == 2

    second = mapping["records"][1]
    fields = {
        f["field_name"]: f
        for f in second["fields"]
    }

    assert fields["patient_name"]["value"] == "Jordan Lee"
    assert (
        fields["patient_name"]["source"]["path"]
        == "$.claims[1]['patient_name']"
    )


def test_unknown_column(tmp_path):
    path = tmp_path / "claims.csv"

    path.write_text(
        "claim_id,custom_code\n"
        "CLM001,ABC123\n",
        encoding="utf-8"
    )

    result = preprocess_document(path)
    record = result["structured_mapping"]["records"][0]

    assert record["unmapped_fields"][0]["source_field"] == "custom_code"
    assert result["requires_review"] is True


def test_ambiguous_mapping(tmp_path):
    path = tmp_path / "claims.csv"

    path.write_text(
        "patient_name,claimant_name\n"
        "Alex Morgan,Jordan Lee\n",
        encoding="utf-8"
    )

    result = preprocess_document(path)
    record = result["structured_mapping"]["records"][0]

    assert record["requires_review"] is True
    assert any(
        issue["type"] == "ambiguous_field_mapping"
        for issue in record["issues"]
    )


def test_nested_json_requires_review(tmp_path):
    path = tmp_path / "claim.json"

    path.write_text(json.dumps({
        "claim_id": "CLM001",
        "patient": {"name": "Alex Morgan"}
    }), encoding="utf-8")

    result = preprocess_document(path)
    record = result["structured_mapping"]["records"][0]

    assert record["requires_review"] is True
    assert any(
        issue["type"] == "nested_fields_not_mapped"
        for issue in record["issues"]
    )
