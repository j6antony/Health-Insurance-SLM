
from unittest.mock import patch

import pytest
from docx import Document

from src.preprocessing.pipeline import preprocess_document


SAMPLE_TEXT = """HOSPITAL BILL
Patient Name: Alex Morgan
Hospital Name: General Hospital
Policy Number: POL123456
Admission Date: 10/03/2026
Discharge Date: 10/07/2026
Total Hospital Charges: $8,700.00
"""


@pytest.fixture(params=["txt", "docx"])
def sample_document(tmp_path, request):
    extension = request.param
    file_path = tmp_path / f"hospital_bill.{extension}"

    if extension == "txt":
        file_path.write_text(
            SAMPLE_TEXT, encoding="utf-8"
        )

    else:
        document = Document()

        for line in SAMPLE_TEXT.splitlines():
            if line.strip():
                document.add_paragraph(line)

        document.save(file_path)

    return file_path


def test_text_document_pipeline(sample_document):
    expected_classification = {
        "document_type": "hospital_bill",
        "evidence": ["HOSPITAL BILL"],
        "requires_review": False,
        "rejected_evidence": []
    }

    expected_extraction = {
        "document_id": "temporary",
        "document_type": "hospital_bill",
        "fields": [{
            "field_name": "hospital_name",
            "value": "General Hospital",
            "evidence": "Hospital Name: General Hospital",
            "page_number": 1,
            "requires_review": False
        }],
        "field_count": 1
    }

    # Mock the model calls to test pipeline integration
    # without depending on Ollama responses.
    with (
        patch(
            "src.preprocessing.pipeline.classify_document",
            return_value=expected_classification
        ) as classify_mock,
        patch(
            "src.preprocessing.pipeline.extract_document_fields",
            return_value=expected_extraction
        ) as extract_mock,
        patch(
            "src.preprocessing.pipeline.classify_document_lines",
            side_effect=lambda lines: {"predictions": [
                {"line_id": i, "label": "not_heading"}
                for i in range(1, len(lines) + 1)
            ]}
        ) as headings_mock
    ):
        result = preprocess_document(sample_document)

    assert result["processing_status"] == "processed"
    assert result["classification"]["document_type"] == "hospital_bill"
    assert result["extraction"]["field_count"] == 1

    assert classify_mock.call_count == 1
    assert extract_mock.call_count == 1

    assert "General Hospital" in result["pages"][0]["text"]
    assert result["chunk_count"] >= 1
    assert result["pages"][0]["sections"]
    assert result["chunks"][0]["source_locations"]
    assert headings_mock.call_count == 1

    print(f"{sample_document.suffix} pipeline passed!")
