
from src.pipeline.claim_pipeline import process_claim_package


FILES = [
    "data/raw/reference_pdfs/cigna_hospital_care_claim_form.pdf",
    "data/raw/reference_pdfs/cigna_physician_statement.pdf",
]


def test_claim_pipeline():
    result = process_claim_package(
        file_paths=FILES,
        output_dir="data/processed/claim_package_test"
    )

    print("\n--- Claim Package Results ---")

    for document in result["documents"]:
        print(f"\nDocument: {document['filename']}")
        print(
            "Type:",
            document["classification"]["document_type"]
        )
        print(
            "Fields:",
            document["extraction"]["field_count"]
        )
        for field in document["extraction"]["fields"]:
            print(
                f"  {field['field_name']}: {field['value']}"
            )
            print(f"    Page: {field['page_number']}")
            print(f"    Evidence: {field['evidence']}")
            print(f"    Requires review: {field['requires_review']}")

    print("\n--- Cross-Document Verification ---")

    verification = result["verification"]

    for item in verification["results"]:
        print(
            f"{item['field_name']}: {item['status']}"
        )

    assert len(result["documents"]) == 2
    assert verification["documents_checked"] == 2

    print("\nClaim package pipeline test passed!")


if __name__ == "__main__":
    test_claim_pipeline()
