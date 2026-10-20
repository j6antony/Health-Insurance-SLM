"""Conservative schema-driven classification of structured claim records.

Classification is advisory, not equivalent to an authenticated document type.
Unknown/ambiguous layouts must be reviewed, never used to satisfy document rules.
"""
from src.extraction.structured_field_mapper import normalize_key

TYPE_MARKERS = {
    "hospital_bill": {"bill_number", "invoice_number", "hospital_charges", "line_item", "item_code"},
    "claim_form": {"claim_form_number", "claimant_signature", "claim_submission_date"},
    "prescription": {"prescription_number", "prescriber_name", "dosage", "medication_name"},
    "investigation_report": {"test_name", "test_result", "lab_result"},
    "discharge_summary": {"discharge_diagnosis", "discharge_instructions"},
    "physician_statement": {"physician_statement", "physician_signature"},
}


def classify_structured_record(record):
    """Recognize only distinctive mapped/unmapped column names.

    Shared columns such as patient_name or total_amount do not establish
    a document type. Record classes are *not* documentation evidence.
    """
    columns = {
        normalize_key(f.get("source_field", f["field_name"]))
        for f in record.get("fields", [])
    }
    columns.update(
        normalize_key(f["source_field"])
        for f in record.get("unmapped_fields", [])
    )
    candidates = {
        kind: sorted(columns & markers)
        for kind, markers in TYPE_MARKERS.items()
        if columns & markers
    }
    if len(candidates) == 1:
        kind, evidence = next(iter(candidates.items()))
        return {
            "document_type": kind,
            "classification_method": "schema_heuristic",
            "evidence_columns": evidence,
            "requires_review": True,
            "documentary_evidence": False,
        }
    return {
        "document_type": "unknown",
        "classification_method": "schema_heuristic",
        "evidence_columns": sorted(
            col for matches in candidates.values() for col in matches
        ),
        "requires_review": True,
        "documentary_evidence": False,
    }


def classify_structured_mapping(mapping):
    for record in mapping["records"]:
        record["classification"] = classify_structured_record(record)
    mapping["record_classification_method"] = "schema_heuristic"
    return mapping
