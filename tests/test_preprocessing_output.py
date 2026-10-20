
import json
from pathlib import Path


FILE_PATH = Path(
    "data/processed/cigna_preprocessed.json"
)


def test_preprocessing_output():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        document = json.load(f)

    # Validate document metadata
    assert document["document_id"]
    assert document["filename"]
    assert document["page_count"] == len(document["pages"])

    # Validate chunk count
    assert document["chunk_count"] == len(document["chunks"])

    for page in document["pages"]:
        assert "raw_text" in page
        assert "normalized_text" in page
        assert "quality" in page
        assert "sections" in page
        assert "chunks" in page

        lines = page["lines"]
        predictions = page.get("heading_predictions", [])

        # Validate all extracted lines are classified
        assert len(predictions) == len(lines)

        expected_ids = set(range(1, len(lines) + 1))
        returned_ids = [
            p["line_id"] for p in predictions
        ]

        assert len(returned_ids) == len(expected_ids)
        assert set(returned_ids) == expected_ids

        # Validate that segmentation preserved all lines
        content_ids = [
            item["line_id"]
            for section in page["sections"]
            for item in section["source_lines"]
        ]

        heading_ids = [
            section["heading_line_id"]
            for section in page["sections"]
            if section["heading_line_id"] is not None
        ]

        all_ids = content_ids + heading_ids

        assert len(all_ids) == len(lines)
        assert set(all_ids) == expected_ids

        # Validate chunk references
        for chunk in page["chunks"]:
            assert chunk["document_id"] == document["document_id"]
            assert chunk["page_number"] == page["page_number"]
            assert chunk["text"].strip()

            assert all(
                line_id in expected_ids
                for line_id in chunk["source_lines"]
            )

    print("Document schema validated!")
    print("All source lines preserved!")
    print("All chunk references valid!")
    print("Total chunks:", document["chunk_count"])


if __name__ == "__main__":
    test_preprocessing_output()
