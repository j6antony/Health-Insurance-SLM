
import re
from collections import defaultdict


# Existing canonical fields supported by your project.
FIELD_ALIASES = {
    "patient_name": {
        "patient_name", "patient name",
        "claimant_name", "claimant name"
    },
    "policy_number": {
        "policy_number", "policy number",
        "policy_no", "group_policy_number"
    },
    "hospital_name": {
        "hospital_name", "hospital name",
        "facility_name", "facility name"
    },
    "admission_date": {
        "admission_date", "admission date",
        "date_of_admission"
    },
    "discharge_date": {
        "discharge_date", "discharge date",
        "date_of_discharge"
    },
    "diagnosis": {
        "diagnosis", "diagnosis_description"
    },
    "total_amount": {
        "total_amount", "total amount",
        "total_charges", "total_bill",
        "total_hospital_charges"
    },
    "currency": {
        "currency", "currency_code"
    }
}

# Record identity is preserved separately from
# the fields extracted by the existing SLM.
IDENTIFIER_ALIASES = {
    "claim_id", "claim_number", "claim_no"
}


def normalize_key(key):
    key = str(key).strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


ALIAS_LOOKUP = {}

for canonical, aliases in FIELD_ALIASES.items():
    for alias in aliases | {canonical}:
        normalized = normalize_key(alias)

        if (
            normalized in ALIAS_LOOKUP
            and ALIAS_LOOKUP[normalized] != canonical
        ):
            raise ValueError(
                f"Conflicting alias: {normalized}"
            )

        ALIAS_LOOKUP[normalized] = canonical


def is_empty(value):
    return (
        value is None
        or isinstance(value, str)
        and (
            not value.strip()
            or value.strip().lower() in {
                "n/a", "null", "none", "unknown"
            }
        )
    )


def make_record(record_id, values, locations):
    fields = []
    identifiers = {}
    unmapped = []
    issues = []

    # Multiple source columns may map to the same field.
    mapped_candidates = defaultdict(list)

    for key, value in values.items():
        normalized_key = normalize_key(key)
        source = locations[key]

        if normalized_key in IDENTIFIER_ALIASES:
            if not is_empty(value):
                identifiers[normalized_key] = {
                    "value": str(value),
                    "source": source
                }
            continue

        canonical = ALIAS_LOOKUP.get(normalized_key)

        if canonical is None:
            unmapped.append({
                "source_field": str(key),
                "value": value,
                "source": source
            })
            continue

        if is_empty(value):
            # Absence is preserved, not turned into a
            # fabricated field value.
            continue

        mapped_candidates[canonical].append({
            "field_name": canonical,
            "value": str(value),
            "raw_value": value,
            "source_field": str(key),
            "source": source,
            "extraction_method": "deterministic",
            "requires_review": False
        })

    for canonical, candidates in mapped_candidates.items():
        # Do not silently select between multiple columns
        # mapping to the same canonical field.
        if len(candidates) > 1:
            issues.append({
                "type": "ambiguous_field_mapping",
                "field_name": canonical,
                "sources": [
                    candidate["source"]
                    for candidate in candidates
                ]
            })

            for candidate in candidates:
                candidate["requires_review"] = True

        fields.extend(candidates)

    return {
        "record_id": record_id,
        "identifiers": identifiers,
        "fields": fields,
        "field_count": len(fields),
        "unmapped_fields": unmapped,
        "issues": issues,
        "requires_review": bool(issues)
    }


def map_csv(data):
    records = []

    for row in data["rows"]:
        row_number = row["row_number"]
        values = row["values"]

        locations = {
            key: {
                "type": "csv",
                "row": row_number,
                "column": key
            }
            for key in values
        }

        records.append(make_record(
            f"csv_row_{row_number}",
            values,
            locations
        ))

    return records


def column_letter(index):
    result = ""

    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result

    return result


def map_excel(data):
    records = []

    for sheet in data["sheets"]:
        sheet_name = sheet["sheet_name"]
        rows = sheet["rows"]

        if len(rows) < 2:
            continue

        # First row is treated as the header.
        headers = rows[0]["values"]

        # Ambiguous headers cannot safely be mapped.
        normalized = [
            normalize_key(h) if h is not None else ""
            for h in headers
        ]

        if (
            not all(normalized)
            or len(normalized) != len(set(normalized))
        ):
            raise ValueError(
                f"Invalid or duplicate headers: {sheet_name}"
            )

        for row in rows[1:]:
            row_number = row["row_number"]
            values = {}
            locations = {}

            for index, header in enumerate(headers):
                cell_value = (
                    row["values"][index]
                    if index < len(row["values"])
                    else None
                )

                values[str(header)] = cell_value
                locations[str(header)] = {
                    "type": "excel",
                    "sheet": sheet_name,
                    "row": row_number,
                    "cell": f"{column_letter(index + 1)}{row_number}"
                }

            records.append(make_record(
                f"excel_{sheet_name}_row_{row_number}",
                values,
                locations
            ))

    return records


def map_json(data):
    # Supports a flat record, a list of flat records,
    # or a top-level {"claims": [...]} object.
    if isinstance(data, list):
        items = data
        prefix = "$"

    elif isinstance(data, dict) and isinstance(
        data.get("claims"), list
    ):
        items = data["claims"]
        prefix = "$.claims"

    elif isinstance(data, dict):
        items = [data]
        prefix = "$"

    else:
        raise ValueError(
            "JSON must contain an object or list of objects"
        )

    records = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected JSON object at index {index}"
            )

        # Nested objects are not mapped automatically.
        values = {
            key: value
            for key, value in item.items()
            if not isinstance(value, (dict, list))
        }

        if isinstance(data, dict) and prefix == "$":
            record_path = "$"
        else:
            record_path = f"{prefix}[{index}]"

        locations = {
            key: {
                "type": "json",
                "path": f"{record_path}[{key!r}]"
            }
            for key in values
        }

        record = make_record(
            f"json_record_{index + 1}",
            values,
            locations
        )

        nested_keys = [
            key for key, value in item.items()
            if isinstance(value, (dict, list))
        ]

        if nested_keys:
            record["issues"].append({
                "type": "nested_fields_not_mapped",
                "fields": nested_keys
            })
            record["requires_review"] = True

        records.append(record)

    return records


def map_structured_fields(document):
    file_type = document["file_type"]
    data = document["structured_data"]

    if file_type == "csv":
        records = map_csv(data)
    elif file_type == "xlsx":
        records = map_excel(data)
    elif file_type == "json":
        records = map_json(data)
    else:
        raise ValueError(
            f"Unsupported mapping format: {file_type}"
        )

    return {
        "document_id": document["document_id"],
        "mapping_method": "deterministic",
        "record_count": len(records),
        "records": records,
        "requires_review": any(
            record["requires_review"]
            for record in records
        )
    }
