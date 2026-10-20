from typing import Literal
from ollama import Client
from pydantic import BaseModel, Field
from src.verification.field_validator import validate_field
from src.verification.evidence_matcher import recover_evidence

MODEL_NAME = "qwen3:4b-instruct"

client = Client(host="http://localhost:11434")

FieldName = Literal[
    "patient_name",
    "policy_number",
    "hospital_name",
    "admission_date",
    "discharge_date",
    "diagnosis",
    "total_amount",
    "currency"
]

class ExtractField(BaseModel):
    field_name: FieldName
    value: str
    evidence: str
    page_number: int
    requires_review: bool = False

class FieldExtractionResult(BaseModel):
    fields: list[ExtractField] = Field(default_factory=list)

def extract_page_fields(page):
    text = page.get("normalized_text", "")

    if not text.strip():
        return []

    prompt = f"""
    You extract structured fields from health insurance documents.

    Extract ONLY these fields:
    - patient_name
    - policy_number
    - hospital_name
    - admission_date
    - discharge_date
    - diagnosis
    - total_amount
    - currency

    Rules:
    1. Extract only actual populated values.
    2. Do not extract empty form labels or placeholders.
    3. Do not invent missing values.
    4. Do not infer values from general instructions.
    5. A company name in a logo or footer is not automatically
    the hospital name.
    6. Do not extract example patient names or example amounts.
    7. Values MUST be copied exactly as written in the document.
        Do not normalize dates, amounts, or identifiers.
        For monetary amounts, preserve currency symbols,
        commas, and decimal places.
        Example:
        Source: Total Hospital Charges: $8,700.00
        Correct value: $8,700.00
        Incorrect value: 8700.00
    8. Evidence must be an exact substring of the document text.
    9. Use an empty fields list when no populated fields exist.
    10. Mark ambiguous values as requires_review=true.
    11. A currency symbol such as "$" does not establish
        a specific ISO currency code.
        Only extract currency if explicitly stated.

    Return JSON following the supplied schema.

    DOCUMENT TEXT:
    {text[:10000]}
    """
    response = client.chat(
        model=MODEL_NAME,
        messages= [
            {"role": "user", "content": prompt}
        ],
        format=FieldExtractionResult.model_json_schema(),
        think = False,
        options = {
            "temperature": 0,
            "num_predict": 1024
        }
    )

    if not response.message.content:
        raise ValueError("Empty field extraction response")

    parsed = FieldExtractionResult.model_validate_json(
        response.message.content
    )

    
    validated = []
    seen = set()

    for field in parsed.fields:
        result = field.model_dump()
        # Recover exact evidence if the model's evidence
        # does not match the extracted document text.
        if result["evidence"] not in text[:10000]:
            recovered = recover_evidence(
                result["value"],
                text[:10000]
            )

            if recovered is not None:
                result["evidence"] = recovered

        is_valid, reason = validate_field(
            result,
            text[:10000]
        )

        if not is_valid:
            print(
                f"[REJECTED] Page {page['page_number']}: "
                f"{field.field_name} = {field.value!r} "
                f"({reason})"
            )
            continue

        # Prevent exact duplicates on the same page
        key = (
            result["field_name"],
            result["value"]
        )

        if key in seen:
            continue

        seen.add(key)

        result["page_number"] = page["page_number"]
        validated.append(result)

    return validated

def extract_document_fields(document):
    all_fields = []

    for page in document["pages"]:
        fields = extract_page_fields(page)
        all_fields.extend(fields)

    return {
        "document_id": document["document_id"],
        "document_type": document["classification"]["document_type"],
        "fields": all_fields,
        "field_count": len(all_fields)
    }
