
import json
from pathlib import Path

from src.slm.field_extractor import extract_document_fields


INPUT_DIR = Path("data/processed/batch")


def test_field_extractor():
    json_files = sorted(INPUT_DIR.glob("*.json"))

    assert json_files, "No preprocessed documents found"

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            document = json.load(f)

        # Older batch JSON may not contain classification yet
        if "classification" not in document:
            document["classification"] = {
                "document_type": "unknown"
            }

        result = extract_document_fields(document)

        print("\n" + "=" * 50)
        print("Document:", document["filename"])
        print("Fields extracted:", result["field_count"])

        for field in result["fields"]:
            print(
                f"  {field['field_name']}: {field['value']} "
                f"(Page {field['page_number']})"
            )
            print(f"    Evidence: {field['evidence']}")
            print(f"    Review: {field['requires_review']}")

        assert result["field_count"] == len(result["fields"])

        for field in result["fields"]:
            assert field["value"].strip()
            assert field["evidence"].strip()

    print("\nField extraction test completed!")


if __name__ == "__main__":
    test_field_extractor()
