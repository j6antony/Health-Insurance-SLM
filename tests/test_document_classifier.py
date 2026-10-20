
from pathlib import Path
import json

from src.slm.document_classifier import classify_document


INPUT_DIR = Path("data/processed/batch")


def test_document_classifier():
    pdf_results = sorted(INPUT_DIR.glob("*.json"))

    assert pdf_results, "No preprocessed JSON files found"

    for file_path in pdf_results:
        with open(file_path, "r", encoding="utf-8") as f:
            document = json.load(f)

        result = classify_document(document)

        print("\n" + "=" * 50)
        print("Document:", document["filename"])
        print("Predicted type:", result["document_type"])
        print("Review required:", result["requires_review"])
        print("Evidence:")

        for excerpt in result["evidence"]:
            print(f"  - {excerpt}")


if __name__ == "__main__":
    test_document_classifier()
