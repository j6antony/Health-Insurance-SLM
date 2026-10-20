"""Tests for evidence-backed mixed-format verification reports."""
import json
from unittest.mock import patch

from src.reporting.report_builder import build_verification_report
from src.pipeline.mixed_claim_pipeline import process_mixed_claim_package


def example_verification():
    return {
        "sources_checked": 3,
        "results": [
            {
                "field_name": "admission_date",
                "status": "conflict",
                "observations": [
                    {
                        "source_id": "pdf-1",
                        "value": "10/03/2026",
                        "normalized_value": "2026-10-03",
                        "source": {
                            "type": "document",
                            "page_number": 1,
                            "evidence": "Admission Date: 10/03/2026",
                        },
                    },
                    {
                        "source_id": "csv-1",
                        "record_id": "csv_row_2",
                        "value": "10/04/2026",
                        "normalized_value": "2026-10-04",
                        "source": {
                            "type": "csv", "row": 2,
                            "column": "admission_date",
                        },
                    },
                ],
            },
            {
                "field_name": "patient_name",
                "status": "consistent",
                "observations": [
                    {"source_id": "pdf-1", "value": "Alex Morgan",
                     "source": {"type": "document", "page_number": 1}},
                    {"source_id": "csv-1", "value": "Alex Morgan",
                     "source": {"type": "csv", "row": 2, "column": "patient_name"}},
                ],
            },
            {"field_name": "policy_number", "status": "insufficient_evidence",
             "observations": []},
        ],
        "review_issues": [],
    }


def test_conflict_and_provenance():
    report = build_verification_report(
        "CLM001", example_verification(), {"satisfied": True}
    )
    conflict = next(f for f in report["findings"]
                    if f["field_name"] == "admission_date")
    assert conflict["status"] == "issue_detected"
    assert conflict["review_required"] is True
    assert conflict["evidence"][0]["location"]["page_number"] == 1
    assert conflict["evidence"][1]["record_id"] == "csv_row_2"
    assert conflict["evidence"][1]["location"]["row"] == 2
    assert report["overall_status"] == "requires_review"
    assert report["adjudication_status"] == "not_evaluated"
    json.dumps(report)


def test_unsatisfied_requirement_always_flags_review():
    report = build_verification_report(
        "CLM001",
        {"sources_checked": 1, "results": [], "review_issues": []},
        {"satisfied": False, "operation": "OR", "children": []},
    )
    requirement = next(f for f in report["findings"]
                       if f["check_type"] == "documentation_requirements")
    assert requirement["status"] == "requires_review"
    assert requirement["rule"]["source"] == "demonstration_only"
    assert requirement["evidence"][0]["evaluation"]["satisfied"] is False
    assert report["requires_review"] is True


def test_review_issue_and_unavailable_rules():
    report = build_verification_report(
        "CLM001",
        {"results": [], "review_issues": [
            {"type": "unassociated_record",
             "source_id": "csv-1", "record_id": "row_3"},
        ]},
        None,
    )
    assert any(f["check_type"] == "verification_review_issue"
               for f in report["findings"])
    assert any(f["check_type"] == "documentation_requirements"
               and f["status"] == "not_evaluated" for f in report["findings"])
    assert report["requires_review"] is True


@patch("src.pipeline.mixed_claim_pipeline.evaluate_requirement",
       return_value={"satisfied": False, "operation": "OR", "children": []})
@patch("src.pipeline.mixed_claim_pipeline.preprocess_document")
def test_pipeline_reports_unsatisfied_requirements(
    preprocess_mock, requirement_mock, tmp_path
):
    preprocess_mock.return_value = {
        "document_id": "pdf-1",
        "filename": "bill.pdf",
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
    result = process_mixed_claim_package(
        "CLM001", ["bill.pdf"], output_dir=tmp_path
    )
    assert result["requires_review"] is True
    assert result["report"]["overall_status"] == "requires_review"
    saved = json.loads((tmp_path / "verification_report.json").read_text())
    assert saved["report_id"] == result["report"]["report_id"]
