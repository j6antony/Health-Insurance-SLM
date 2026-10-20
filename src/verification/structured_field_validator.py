
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

    # Validate basic cross-field chronology without guessing missing dates.
    date_formats = (
        "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y",
        "%m-%d-%Y", "%B %d, %Y",
    )
    def parse_date(value):
        for fmt in date_formats:
            try:
                return datetime.strptime(str(value).strip(), fmt).date()
            except ValueError:
                pass
        return None

    valid_dates = {
        field["field_name"]: parse_date(field["value"])
        for field in record["fields"]
        if field["field_name"] in {"admission_date", "discharge_date"}
        and field.get("validation_status") == "valid"
    }
    admission = valid_dates.get("admission_date")
    discharge = valid_dates.get("discharge_date")
    if admission and discharge and discharge < admission:
        issues.append({
            "type": "discharge_before_admission",
            "admission_date": admission.isoformat(),
            "discharge_date": discharge.isoformat(),
        })
        for field in record["fields"]:
            if field["field_name"] in {"admission_date", "discharge_date"}:
                field["requires_review"] = True

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
