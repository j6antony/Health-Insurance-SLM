
from collections import defaultdict

from src.verification.cross_document import (
    COMPARABLE_FIELDS,
    normalize_value,
)


CLAIM_ID_KEYS = {"claim_id", "claim_number", "claim_no"}


def select_claim_records(mapping, claim_id):
    """Select only records explicitly belonging to this claim."""
    selected = []

    for record in mapping["records"]:
        ids = {
            str(item["value"]).strip()
            for key, item in record["identifiers"].items()
            if key in CLAIM_ID_KEYS
            and item.get("value") is not None
            and str(item["value"]).strip()
        }

        if len(ids) > 1:
            raise ValueError(
                f"Conflicting claim IDs in {record['record_id']}"
            )

        if claim_id in ids:
            selected.append(record)

    return selected


def build_mixed_observations(
    claim_id,
    documents,
    structured_documents,
):
    observations = []
    review_issues = []

    # --------------------------------------------------
    # 1. Process PDF, DOCX and TXT documents
    # --------------------------------------------------
    # These documents must be explicitly associated
    # with the requested claim by the caller.
    for document in documents:

        if document.get("processing_status") in {
            "ingested", "mapped", "requires_review"
        }:
            raise ValueError(
                f"Document {document['document_id']} "
                "is not fully processed"
            )

        classification = document["classification"]

        if classification.get("requires_review"):
            review_issues.append({
                "type": "classification_requires_review",
                "source_id": document["document_id"]
            })
            continue

        for field in document["extraction"]["fields"]:
            field_name = field["field_name"]

            if field_name not in COMPARABLE_FIELDS:
                continue

            if field.get("requires_review"):
                review_issues.append({
                    "type": "field_requires_review",
                    "source_id": document["document_id"],
                    "field_name": field_name
                })
                continue

            observations.append({
                "field_name": field_name,
                "value": field["value"],
                "source_id": document["document_id"],
                "source": {
                    "type": "document",
                    "document_id": document["document_id"],
                    "page_number": field.get("page_number"),
                    "evidence": field.get("evidence")
                }
            })

    # --------------------------------------------------
    # 2. Process CSV, Excel and JSON documents
    # --------------------------------------------------
    for document in structured_documents:
        mapping = document["structured_mapping"]

        selected_records = select_claim_records(
            mapping,
            claim_id
        )

        if document.get("quality", {}).get("status") == "review_required":
            review_issues.append({
                "type": "source_quality_requires_review",
                "source_id": document["document_id"]
            })

        # Records without a usable claim ID must not
        # be automatically associated with this claim.
        unassociated_records = [
            record
            for record in mapping["records"]
            if not any(
                key in CLAIM_ID_KEYS
                and item.get("value") is not None
                and str(item["value"]).strip()
                for key, item in record["identifiers"].items()
            )
        ]

        for record in unassociated_records:
            review_issues.append({
                "type": "unassociated_record",
                "source_id": document["document_id"],
                "record_id": record["record_id"]
            })

        for record in selected_records:
            # Use document_id for independent-source counting.
            # Multiple rows in one spreadsheet are not
            # automatically independent documents.
            source_id = document["document_id"]

            if record.get("requires_review"):
                review_issues.append({
                    "type": "record_requires_review",
                    "source_id": source_id,
                    "record_id": record["record_id"]
                })

            for field in record["fields"]:
                field_name = field["field_name"]

                if field_name not in COMPARABLE_FIELDS:
                    continue

                if (
                    field.get("validation_status") != "valid"
                    or field.get("requires_review")
                ):
                    review_issues.append({
                        "type": "field_not_validated",
                        "source_id": source_id,
                        "record_id": record["record_id"],
                        "field_name": field_name
                    })
                    continue

                observations.append({
                    "field_name": field_name,
                    "value": field["value"],
                    "source_id": source_id,
                    "record_id": record["record_id"],
                    "source": field["source"]
                })

    return observations, review_issues


def verify_mixed_claim(
    claim_id,
    documents=None,
    structured_documents=None,
):
    """
    Compare validated fields belonging to one claim
    across PDF, TXT, DOCX, CSV, Excel and JSON sources.

    This checks consistency only. It does not approve
    or reject insurance claims.
    """

    if not claim_id or not str(claim_id).strip():
        raise ValueError("Explicit claim_id is required")

    claim_id = str(claim_id).strip()

    documents = documents or []
    structured_documents = structured_documents or []

    observations, review_issues = build_mixed_observations(
        claim_id,
        documents,
        structured_documents
    )

    # --------------------------------------------------
    # 3. Group observations by canonical field
    # --------------------------------------------------
    grouped = defaultdict(list)

    for observation in observations:
        field_name = observation["field_name"]
        value = str(observation["value"])

        grouped[field_name].append({
            **observation,
            "normalized_value": normalize_value(
                field_name,
                value
            )
        })

    # --------------------------------------------------
    # 4. Compare values across sources
    # --------------------------------------------------
    results = []

    for field_name in sorted(COMPARABLE_FIELDS):
        entries = grouped[field_name]

        source_ids = {
            entry["source_id"]
            for entry in entries
        }

        normalized_values = {
            entry["normalized_value"]
            for entry in entries
        }

        # Detect contradictions inside a single source,
        # such as two rows with different admission dates
        # for the same claim.
        values_by_source = defaultdict(set)

        for entry in entries:
            values_by_source[entry["source_id"]].add(
                entry["normalized_value"]
            )

        conflicting_sources = [
            source_id
            for source_id, values in values_by_source.items()
            if len(values) > 1
        ]

        if conflicting_sources:
            review_issues.append({
                "type": "internal_source_conflict",
                "field_name": field_name,
                "source_ids": conflicting_sources
            })

        if len(source_ids) < 2:
            # One document is insufficient to establish
            # cross-document consistency.
            status = "insufficient_evidence"

        elif len(normalized_values) == 1:
            status = "consistent"

        else:
            status = "conflict"

        results.append({
            "field_name": field_name,
            "status": status,
            "observations": entries
        })

    # --------------------------------------------------
    # 5. Generate verification result
    # --------------------------------------------------
    has_conflicts = any(
        item["status"] == "conflict"
        for item in results
    )

    return {
        "claim_id": claim_id,
        "sources_checked": (
            len(documents) + len(structured_documents)
        ),
        "results": results,
        "has_conflicts": has_conflicts,
        "review_issues": review_issues,
        "requires_review": (
            bool(review_issues) or has_conflicts
        )
    }
