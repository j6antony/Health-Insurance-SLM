from collections import defaultdict
from datetime import datetime

COMPARABLE_FIELDS = {
    "patient_name",
    "policy_number",
    "hospital_name",
    "admission_date",
    "discharge_date"
}

def normalize_value(field_name, value):
    value = value.strip()

    if field_name in {"admission_date", "discharge_date"}:
        for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).date().isoformat()
            except ValueError:
                pass
    if field_name in {
        "patient_name",
        "hospital_name"
    }:
        return " ".join(value.casefold().split())
    return value.casefold()

def verify_claim_package(documents):
    grouped = defaultdict(list)

    for document in documents:
        extraction = document.get("extraction", {})

        for field in extraction.get("fields", []):
            name = field["field_name"]

            if name not in COMPARABLE_FIELDS:
                continue

            grouped[name].append({
                "document_id": document["document_id"],
                "document_type": document["classification"]["document_type"],
                "page_number": field["page_number"],
                "value": field["value"],
                "normalized_value": normalize_value(
                    name, field["value"]
                ),
                "evidence": field["evidence"],
            })
    results = []

    for field_name in sorted(COMPARABLE_FIELDS):
        entries = grouped.get(field_name, [])

        document_ids = {
            entry["document_id"] for entry in entries
        }

        values = {
            entry["normalized_value"] for entry in entries
        }

        if len(document_ids) < 2:
            status = "insufficient_evidence"
        elif len(values) == 1:
            status = "consistent"
        else:
            status = "conflict"

        results.append({
            "field_name": field_name,
            "status": status,
            "observations": entries,
        })
    return {
        "documents_checked": len(documents),
        "results": results,
        "has_conflicts": any(
            item["status"] == "conflict"
            for item in results
        )
    }