from typing import Literal

from ollama import Client
from pydantic import BaseModel, Field

MODEL_NAME = "qwen3:4b-instruct"

client = Client(host="http://localhost:11434")

class DocumentClassification(BaseModel):
    document_type: Literal[
        "claim_form",
        "hospital_bill",
        "physician_statement",
        "discharge_summary",
        "investigation_report",
        "prescription",
        "claims_checklist",
        "policy_document",
        "other",
        "unknown"
    ]

    evidence: list[str] = Field(
        description="Short exact excerpts supporting the classification"
    )

    requires_review: bool

def classify_document(document):
    text = "\n".join(
        page["normalized_text"] for page in document["pages"]
    )

    text = text[:6000]

    prompt = f"""
        You classify health insurance documents.

        Choose exactly one document type:

        claim_form:
        A form used to submit an insurance claim.

        hospital_bill:
        An itemized bill listing hospital charges.

        physician_statement:
        A medical statement completed by a physician.

        discharge_summary:
        A summary of a completed hospital stay.

        investigation_report:
        A diagnostic or medical investigation report.

        prescription:
        A document prescribing medications.

        claims_checklist:
        A checklist describing documents required for a claim.

        policy_document:
        Insurance policy terms, limits, and exclusions.

        other:
        A recognizable document outside these categories.

        unknown:
        Insufficient evidence or unclear document type.

        Instructions:
        - Classify the whole document, not individual sections.
        - Use the document's purpose and content.
        - Do not classify based only on the insurer's name.
        - Return short evidence excerpts copied exactly
        from the supplied document.
        - Do not invent evidence.
        - Return at most three short evidence excerpts.
        - Each excerpt must be a continuous span of the supplied text.
        - Preserve bullet markers; do not join text across omitted words or symbols.
        - Set requires_review to true if the classification
        is uncertain, conflicting, or unsupported.
        - Use unknown when evidence is insufficient.

        DOCUMENT:
        {text}
DOCUMENT CLASSIFICATION RULES:

1. Determine the primary purpose of the entire document.

2. Give strong consideration to explicit document titles
   such as "CLAIM FORM", "PHYSICIAN STATEMENT",
   "HOSPITAL BILL", and "DISCHARGE SUMMARY".

3. A claim form may contain hospital charges, diagnoses,
   and treatment dates. These fields alone do not make
   it a hospital bill.

4. Classify a document as hospital_bill when its primary
   purpose is to itemize and request payment for
   healthcare services.

5. When the explicit document title and content conflict,
   evaluate both and set requires_review=true if the
   document type remains ambiguous.
    """
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        format=DocumentClassification.model_json_schema(),
        think = False,
        options={
            "temperature": 0,
            "num_predict": 512
        }
    )

    if not response.message.content:
        raise ValueError(
            "SLM return an empty document classification"
        )

    result = DocumentClassification.model_validate_json(
        response.message.content
    )

    # PDF line wrapping can differ from the model's quoted whitespace.
    evidence_text = " ".join(text.split())
    valid_evidence = []
    rejected_evidence = []
    for excerpt in result.evidence:
        normalized_excerpt = " ".join(excerpt.split())
        if not normalized_excerpt or normalized_excerpt not in evidence_text:
            rejected_evidence.append(excerpt)
        else:
            valid_evidence.append(excerpt)
    result.evidence = valid_evidence
    if rejected_evidence or not valid_evidence or result.document_type == "unknown":
        result.requires_review = True

    output = result.model_dump()
    output["rejected_evidence"] = rejected_evidence
    return output

