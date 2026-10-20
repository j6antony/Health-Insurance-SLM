import re
from datetime import datetime

EMPTY_VALUE = {
    "",
    "n/a",
    "none",
    "null",
    "unknown",
    "requires_review=true",
    "not provided"
}

FIELD_LABELS = {
    "admission_date": {
        "admission date",
        "admission date and time"
    },
    "discharge_date": {
        "discharge date",
        "discharge date and time"
    },
    "diagnosis": {
        "diagnosis",
        "present condition",
    },
    "patient_name": {
        "patient name",
        "claimant name",
        "name first last",
    },
    "hospital_name": {
        "hospital name",
        "facility name"
    },
    "policy_number": {
        "group policy number",
        "policy number"
    },
    "total_amount": {
        "total amount",
        "total charges",
        "total bill"
    },
    "currency": {
        "currency"
    }
}

def normalize_label(value):
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return " ".join(value.split())

def valid_date(value):
    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%m-%d-%Y",
        "%B %d, %Y"
    ]
    for fmt in formats:
        try:
            datetime.strptime(value.strip(), fmt)
            return True
        except ValueError:
            continue
    return False

def validate_field(field, page_text):
    name = field["field_name"]
    value = field["value"].strip()
    evidence = field["evidence"].strip()

    if value.lower() in EMPTY_VALUE:
        return False, "empty_or_placeholder"
    if not evidence or evidence not in page_text:
        return False, "evidence_not_found"
    if value not in evidence:
        return False, "value_not_in_evidence"

    normalized = normalize_label(value)

    if normalized in FIELD_LABELS.get(name, set()):
        return False, "field_label_not_value"

    if name in {"admission_date", "discharge_date"}:
        if not valid_date(value):
            return False, "invalid_date"

    if name == "total_amount":
        amount = value.replace(",", "").replace("$", "").strip()
    # this is policy specifc and will avoid
    #if name == "policy_number":
     #   if re.fullmatch(r"975")

    # Validate monetary amounts
    if name == "total_amount":
        amount = (
            value.replace(",", "")
            .replace("$", "")
            .strip()
        )

        if not re.fullmatch(r"\d+(?:\.\d{1,2})?", amount):
            return False, "invalid_amount"

    # Reject multiple form labels masquerading as a name
    if name in {"hospital_name", "patient_name"}:
        lines = [
            line.strip()
            for line in value.splitlines()
            if line.strip()
        ]

        if len(lines) > 1:
            return False, "multiline_name_candidate"

        if re.search(
            r"(?:name|address|phone|fax|treatment period)"
            r"\s*:\s*\*?",
            value,
            re.IGNORECASE
        ):
            return False, "form_label_not_value"

    # Require explicit policy-number context
    if name == "policy_number":
        if not re.search(
            r"\b(?:group\s+)?policy\s*(?:number|no\.?|#)\s*:",
            evidence,
            re.IGNORECASE
        ):
            return False, "missing_policy_number_context"

    return True, "valid"
