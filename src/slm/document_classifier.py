import os
from typing import Literal

from ollama import Client
from pydantic import BaseModel, Field

MODEL_NAME = os.environ.get("SLM_MODEL_NAME", "qwen2.5:3b-instruct")

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
        - Determine the primary purpose of the document.
        - Give strong consideration to explicit document titles such as "CLAIM FORM", "PHYSICIAN STATEMENT", "HOSPITAL BILL", and "DISCHARGE SUMMARY".
        - A document titled "CLAIM FORM" or "HOSPITAL CARE CLAIM FORM" is a claim_form.
        - A document titled "PHYSICIAN STATEMENT" is a physician_statement.
        - Even brief or single-page documents must be classified by their title and content; do NOT mark them as unknown if a clear title is present.
        - Only use unknown if the text is unreadable, garbled, or has no discernible health insurance document type.
        - Return 1-3 short exact evidence excerpts copied verbatim from the document (under 100 characters each).
        - Set requires_review to true ONLY if the classification is genuinely conflicting or ambiguous.

        DOCUMENT TEXT:
        {text}
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
            "num_predict": 1024
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

