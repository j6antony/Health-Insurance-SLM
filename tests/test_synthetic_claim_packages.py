
from src.pipeline.claim_pipeline import process_claim_package


FILES = [
    "data/raw/synthetic_claims/claim_form.pdf",
    "data/raw/synthetic_claims/physician_statement.pdf",
]


def test_synthetic_package():
    result = process_claim_package(
        file_paths=FILES,
        output_dir="data/processed/synthetic_claim_package"
    )

    print("\n--- Extracted Fields ---")

    for document in result["documents"]:
        print(f"\nDocument: {document['filename']}")
        print("Type:", document["classification"]["document_type"])

        for field in document["extraction"]["fields"]:
            print(
                f"  {field['field_name']}: {field['value']}"
            )

    print("\n--- Verification Results ---")

    statuses = {}

    for item in result["verification"]["results"]:
        statuses[item["field_name"]] = item["status"]

        print(f"{item['field_name']}: {item['status']}")

    assert statuses["patient_name"] == "consistent"
    assert statuses["hospital_name"] == "consistent"
    assert statuses["admission_date"] == "consistent"
    assert statuses["discharge_date"] == "conflict"
    assert statuses["policy_number"] == "insufficient_evidence"

    assert result["verification"]["has_conflicts"]

    print("\nSynthetic claim package test passed!")

    print("\n--- Requirement Verification ---")

    requirements = result["requirements"]

    print("Requirements satisfied:", requirements["satisfied"])

    for child in requirements["children"]:
        print(child)

    assert requirements["satisfied"] is True

    print("\nRequirement verification integration passed!")

if __name__ == "__main__":
    test_synthetic_package()

