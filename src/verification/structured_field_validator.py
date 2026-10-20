
import re
from datetime import datetime

from src.verification.field_validator import valid_date


def validate_mapped_field(field):
    name = field["field_name"]
    value = str(field["value"]).strip()

    if not value or value.lower() in {
        "null", "none", "n/a", "unknown"
    }:
        return False, "empty_value"

    if name in {"admission_date", "discharge_date"}:
        if not valid_date(value):
            return False, "invalid_date"

    elif name == "total_amount":
        amount = value.replace(",", "").replace("$", "").strip()

        if not re.fullmatch(r"\d+(\.\d{1,2})?", amount):
            return False, "invalid_amount"

    elif name == "currency":
        if not re.fullmatch(r"[A-Za-z]{3}", value):
            return False, "invalid_currency"

    return True, "valid"


def validate_record(record):
    issues = list(record.get("issues", []))

    for field in record["fields"]:
        is_valid, reason = validate_mapped_field(field)

        field["validation_status"] = (
            "valid" if is_valid else "invalid"
        )

        if not is_valid:
            field["requires_review"] = True

            issues.append({
                "type": reason,
                "field_name": field["field_name"],
                "value": field["value"],
                "source": field["source"]
            })

    record["issues"] = issues
    record["requires_review"] = (
        bool(issues)
        or any(
            field["requires_review"]
            for field in record["fields"]
        )
    )

    return record


def validate_structured_mapping(mapping):
    seen_claim_ids = {}

    for record in mapping["records"]:
        validate_record(record)

        identifiers = record["identifiers"]

        # Detect duplicate claim IDs between records.
        claim_ids = [
            str(item["value"]).strip()
            for key, item in identifiers.items()
            if key in {"claim_id", "claim_number", "claim_no"}
        ]

        for claim_id in set(claim_ids):
            if claim_id in seen_claim_ids:
                record["issues"].append({
                    "type": "duplicate_claim_identifier",
                    "claim_id": claim_id,
                    "first_record_id": seen_claim_ids[claim_id]
                })
                record["requires_review"] = True
            else:
                seen_claim_ids[claim_id] = record["record_id"]

    mapping["requires_review"] = any(
        record["requires_review"]
        for record in mapping["records"]
    )

    return mapping
