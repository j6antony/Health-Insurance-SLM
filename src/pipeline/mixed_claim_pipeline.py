
from pathlib import Path

from src.preprocessing.pipeline import preprocess_document
from src.verification.mixed_format import (
    verify_mixed_claim,
    select_claim_records,
)
from src.verification.requirements import evaluate_requirement
from src.verification.requirement_rules import (
    HOSPITALIZATION_REQUIREMENT,
)


STRUCTURED_FORMATS = {".csv", ".xlsx", ".json"}
DOCUMENT_FORMATS = {".pdf", ".docx", ".txt"}


def process_mixed_claim_package(
    claim_id,
    file_paths,
    output_dir=None,
):
    if not str(claim_id).strip():
        raise ValueError("claim_id is required")

    if not file_paths:
        raise ValueError("Claim package cannot be empty")

    claim_id = str(claim_id).strip()

    documents = []
    structured_documents = []

    for file_path in file_paths:
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension not in (
            STRUCTURED_FORMATS | DOCUMENT_FORMATS
        ):
            raise ValueError(
                f"Unsupported file format: {extension}"
            )

        output_path = None

        if output_dir is not None:
            output_path = (
                Path(output_dir)
                / f"{path.stem}_{extension.lstrip('.')}.json"
            )

        document = preprocess_document(
            file_path=path,
            output_path=output_path,
        )

        if extension in STRUCTURED_FORMATS:
            if document.get("processing_status") != "mapped":
                raise ValueError(
                    f"Structured file not mapped: {path.name}"
                )

            matching = select_claim_records(
                document["structured_mapping"],
                claim_id,
            )

            if not matching:
                raise ValueError(
                    f"No records for claim {claim_id} "
                    f"in {path.name}"
                )

            structured_documents.append(document)

        else:
            if document.get("processing_status") in {
                "ingested",
                "mapped",
                "requires_review",
            }:
                raise ValueError(
                    f"Document not fully processed: {path.name}"
                )

            documents.append(document)

    # Compare fields across all eligible sources.
    verification = verify_mixed_claim(
        claim_id=claim_id,
        documents=documents,
        structured_documents=structured_documents,
    )

    # Preserve the existing demonstration
    # documentation-requirement engine.
    #
    # Only classified document sources participate.
    # Structured records must not masquerade as
    # required documents.
    requirement_documents = []

    for document in documents:
        classification = document["classification"]

        if classification.get("requires_review"):
            continue

        fields = {
            field["field_name"]
            for field in document["extraction"]["fields"]
            if not field.get("requires_review")
        }

        requirement_documents.append({
            "document_id": document["document_id"],
            "document_type": classification["document_type"],
            "fields": fields,
        })

    # Existing rules are demonstration rules.
    # Report the result separately from mixed verification.
    requirements = evaluate_requirement(
        HOSPITALIZATION_REQUIREMENT,
        requirement_documents,
    )

    return {
        "claim_id": claim_id,
        "documents": documents,
        "structured_documents": structured_documents,
        "verification": verification,
        "requirements": requirements,
        "requirements_status": "demonstration_only",
        "requires_review": verification["requires_review"],
        "processing_status": "verification_completed",
    }
