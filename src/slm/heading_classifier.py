import json
import os
from ollama import Client
from typing import Literal
from pydantic import BaseModel

MODEL_NAME = os.environ.get("SLM_MODEL_NAME", "qwen3:4b-instruct")

LABELS = {
    "document_header",
    "document_title",
    "section_heading",
    "subsection_heading",
    "field_label",
    "body_text",
    "list_item",
    "separator"
}


class SingleLinePrediction(BaseModel):
    label: Literal[
        "document_header",
        "document_title",
        "section_heading",
        "subsection_heading",
        "not_heading"
    ]


class DocumentLinePrediction(BaseModel):
    line_id: int
    label: Literal[
        "document_header",
        "document_title",
        "section_heading",
        "subsection_heading",
        "field_label",
        "body_text",
        "list_item",
        "separator"
    ]


class ClassificationResponse(BaseModel):
    predictions: list[DocumentLinePrediction]


client = Client(host="http://localhost:11434")

def classify_single_line(lines, index):
    target = lines[index]

    previous_text = (
        lines[index - 1]["text"]
        if index > 0 else ""
    )

    next_text = (
        lines[index + 1]["text"]
        if index < len(lines) - 1 else ""
    )

    prompt = f"""
You are a heading detection system for health
insurance documents.

Your ONLY task is to determine whether the target
line is a heading.

Allowed labels:

document_header:
Company names, logos, or branding at the beginning
of a document.

document_title:
The main title describing the document.

section_heading:
A heading introducing a major section containing
related information.

subsection_heading:
A smaller heading introducing a subsection or note.

not_heading:
All ordinary text, field labels, bullet points,
continuation lines, separators, and footer text.

IMPORTANT:
- Classify ONLY the target line.
- Use surrounding lines for context.
- Do not classify ordinary sentences as headings.
- A bold field label is not necessarily a heading.
- "OR" and "AND" are not headings.
- Footer copyright and legal text are not headings.
- Return only a JSON object with exactly one key, "label".
- Example output: {{"label": "section_heading"}}
- Do not return explanations or plain text outside the JSON object.

Previous line: {previous_text}

Target line: {target["text"]}
Font size: {target.get("font_size", "unknown")}
Bold: {target.get("is_bold", "unknown")}

Next line: {next_text}
"""

    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        format=SingleLinePrediction.model_json_schema(),
        think=False,
        options={
            "temperature": 0,
            "num_predict": 256
        }
    )

    content = response.message.content
    if not content or not content.strip():
        raise ValueError(
            f"Ollama returned empty content for line {index + 1}. "
            f"Model: {MODEL_NAME}; "
            f"done_reason: {getattr(response, 'done_reason', None)}; "
            f"eval_count: {getattr(response, 'eval_count', None)}. "
            f"thinking_characters: {len(getattr(response.message, 'thinking', None) or '')}. "
            "No classification was produced."
        )
    result = SingleLinePrediction.model_validate_json(content)

    return result.label

def classify_document_lines(lines):
    predictions = []

    for index, line in enumerate(lines):
        label = classify_single_line(lines, index)

        predictions.append({
            "line_id": index + 1,
            "label": label
        })

    return {
        "predictions": predictions
    }


def classify_document_structure(text, expected_ids=None):
    expected_ids = expected_ids or []

    prompt = f"""
You classify the structure of health insurance documents.

Classify every numbered line independently, using its
surrounding lines for context.

Labels:
- document_header: Company branding
- document_title: Main document title
- section_heading: Major section heading
- subsection_heading: Minor heading
- field_label: Field name or requested information
- body_text: Normal explanatory content
- list_item: List entry
- separator: OR, AND, or similar separators

Required line IDs: {expected_ids}

Return exactly {len(expected_ids)} predictions.
Include every required ID exactly once.
Never merge or omit lines.

DOCUMENT:
{text}
"""

    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        format=ClassificationResponse.model_json_schema(),
        options={
            "temperature": 0,
            "num_predict": 2048
        }
    )

    parsed = ClassificationResponse.model_validate_json(
        response.message.content
    )

    returned_ids = [
        prediction.line_id
        for prediction in parsed.predictions
    ]

    if (
        len(returned_ids) != len(expected_ids)
        or set(returned_ids) != set(expected_ids)
    ):
        raise ValueError(
            f"Expected IDs {expected_ids}, "
            f"received {returned_ids}"
        )

    return parsed.model_dump()


def prepare_lines_for_slm(lines):
    formatted_lines = []

    for index, line in enumerate(lines, start = 1):
        text = line["text"].strip()

        formatted_lines.append(
            f"{index}. {text} "
            f"[font_size={line.get('font_size', 'unknown')}, "
            f"bold={line.get('is_bold', 'unknown')}]"
        )
    return "\n".join(formatted_lines)
def classify_document_in_batches(lines, batch_size = 10):
    all_predictions = []

    for start in range(0, len(lines), batch_size):
        batch = lines[start: start + batch_size]

        formatted_lines = []

        for index, line in enumerate(batch, start = start + 1):
            formatted_lines.append(
                f"{index}. {line['text']} "
                f"[font_size={line.get('font_size', 'unknown')}, "
                f"bold={line.get('is_bold', 'unknown')}]"
            )
            prompt_input = "\n".join(formatted_lines)

            result = classify_document_structure(
                prompt_input,
                expected_ids=list(range(
                    start + 1,
                    start + len(batch) + 1
                ))
            )

            print(f"\n--- Batch {start + 1}-{start + len(batch)} ---")
            print(json.dumps(result, indent=2))

            predictions = result["predictions"]

            expected_ids = set(
                range(start + 1, start + len(batch) + 1)
            )

            returned_ids = [p["line_id"] for p in predictions]

            if (
                len(returned_ids) != len(expected_ids)
                or set(returned_ids) != expected_ids
                or any(p["label"] not in LABELS for p in predictions)
            ):
                raise ValueError(
                f"Invalid predictions for batch "
                f"{start + 1}-{start + len(batch)}. "
                f"Returned IDs: {returned_ids}"
                )
            all_predictions.extend(predictions)
        return {
            "predictions": all_predictions
        }
