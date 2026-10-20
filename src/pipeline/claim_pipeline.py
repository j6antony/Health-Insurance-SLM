from pathlib import Path

from src.preprocessing.pipeline import preprocess_document
from src.verification.cross_document import verify_claim_package

from src.verification.requirements import evaluate_requirement
from src.verification.requirement_rules import HOSPITALIZATION_REQUIREMENT

def process_claim_package(file_paths, output_dir = None):
    if not file_paths:
        raise ValueError("Claim package cannot be empty")
    documents = []

    for file_path in file_paths:
        file_path = Path(file_path)

        output_path = None

        if output_dir is not None:
            output_path = (
                Path(output_dir) / f"{file_path.stem}.json"
            )
        document = preprocess_document(
            file_path=str(file_path),
            output_path=output_path
        )
        documents.append(document)
    verification = verify_claim_package(documents)
    requirement_documents = []

    for document in documents:
        requirement_documents.append({
            "document_id": document["document_id"],
            "document_type": document["classification"]["document_type"],
            "fields": {
                field["field_name"]
                for field in document["extraction"]["fields"]
            }
        })

    # Evaluate documentation requirements
    requirements_result = evaluate_requirement(
        HOSPITALIZATION_REQUIREMENT,
        requirement_documents
    )

    return {
        "documents": documents,
        "verification": verification,
        "requirements": requirements_result
    }
