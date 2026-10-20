
from src.preprocessing.pipeline import preprocess_document


FILE_PATH = (
    "./data/raw/reference_pdfs/"
    "cigna_hospital_care_checklist.pdf"
)

OUTPUT_PATH = "data/processed/cigna_preprocessed.json"


def test_pipeline():
    result = preprocess_document(
        file_path=FILE_PATH,
        output_path=OUTPUT_PATH,
        max_words=100
    )

    print("\n--- Preprocessing Results ---")
    print("Document:", result["filename"])
    print("Pages:", result["page_count"])
    print("Quality:", result["quality"]["status"])
    print("Total chunks:", result["chunk_count"])

    for page in result["pages"]:
        print(
            f"Page {page['page_number']} | "
            f"Method: {page['extraction_method']} | "
            f"Sections: {len(page['sections'])} | "
            f"Chunks: {len(page['chunks'])}"
        )

    assert result["chunk_count"] == len(result["chunks"])
    assert result["chunk_count"] > 0

    print("\nEnd-to-end preprocessing test passed!")

    classification = result["classification"]

    print("\n--- Document Classification ---")
    print("Type:", classification["document_type"])
    print("Review Required:", classification["requires_review"])
    print("Evidence:", classification["evidence"])

    assert classification["document_type"] in {
        "claim_form",
        "hospital_bill",
        "physician_statement",
        "discharge_summary",
        "investigation_report",
        "prescription",
        "claims_checklist",
        "policy_document",
        "other",
        "unknown"
    }

    print("\n--- Field Extraction ---")

    extraction = result["extraction"]

    print("Fields extracted:", extraction["field_count"])

    for field in extraction["fields"]:
        print(
            f"{field['field_name']}: {field['value']} "
            f"(Page {field['page_number']})"
        )

    assert extraction["field_count"] == len(extraction["fields"])

    for field in extraction["fields"]:
        assert field["value"].strip()
        assert field["evidence"].strip()
        assert 1 <= field["page_number"] <= result["page_count"]

    print("\nField extraction integration passed!")

if __name__ == "__main__":
    test_pipeline()
