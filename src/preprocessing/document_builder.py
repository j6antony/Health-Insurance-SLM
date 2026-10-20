from src.extraction.reading_order import order_lines
import uuid
from datetime import datetime, timezone
from src.extraction.pdf_extractor import needs_ocr
from src.ocr.tesseract_ocr import (
    extract_ocr_blocks,
    group_words_into_lines,
    group_lines_into_paragraphs
)
from src.preprocessing.text_normalizer import normalize_text
from src.extraction.digital_layout import extract_digital_layout
from src.verification.extraction_quality import (
    evaluate_poor_quality,
    evaluate_document_quality
)
from src.verification.structured_quality import validate_structured_document
from src.extraction.csv_extractor import extract_csv
from src.extraction.structured_extractors import extract_non_pdf
import pymupdf
from pathlib import Path


def process_non_pdf_document(file_path):
    file_path = Path(file_path)
    document_id = str(uuid.uuid4())

    extracted = extract_non_pdf(file_path)
    text = extracted["text"]

    lines = []

    for index, line_text in enumerate(text.splitlines(), start=1):
        if not line_text.strip():
            continue
        lines.append({
            "text": line_text,
            "bbox": None,
            "confidence": None,
            "line_id": [index, 0, 0],
            "font_size": None,
            "is_bold": False
        })
    page = {
        "document_id": document_id,
        "filename": file_path.name,
        "page_number": 1,
        "extraction_method": file_path.suffix.lower().lstrip("."),
        "text": text,
        "raw_text": text,
        "normalized_text": normalize_text(text),
        "lines": lines,
        "paragraphs": [],
        "word_count": len(text.split())
    }

    page["quality"] = evaluate_poor_quality(page)

    document = {"document_id": document_id,
        "filename": file_path.name,
        "file_type": file_path.suffix.lower().lstrip("."),
        "page_count": 1,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "quality": evaluate_document_quality([page]),
        "pages": [page],
        "structured_data": extracted["structured_data"]}
    document["structured_quality"] = (
        validate_structured_document(document)
    )
    if document["structured_quality"]["status"] == "review_required":
        document["quality"]["status"] = "review_required"
        document["quality"]["issues"] = sorted(set(
            document["quality"]["issues"] + [issue["type"] for issue in document["structured_quality"]["issues"]]
        ))

    return document

def process_document(file_path):
    file_path = Path(file_path)
    if file_path.suffix.lower() in {
        ".csv", ".xlsx", ".json", ".txt", ".docx"
    }:
        return process_non_pdf_document(file_path)

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}"
        )
    document_id = str(uuid.uuid4())
    processed_pages = []

    with pymupdf.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start = 1):
            result = build_document(
                page = page,
                filename= file_path.name,
                page_number = page_number
            )
            result["document_id"] = document_id
            result["quality"] = evaluate_poor_quality(result)
            processed_pages.append(result)
    return {
        "document_id": document_id,
        "filename": file_path.name,
        "page_count": len(processed_pages),
        "processed_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "quality": evaluate_document_quality(processed_pages),
        "pages": processed_pages
    }
def build_document(page, filename, page_number):
    digital_text = page.get_text("text")
    requires_ocr = needs_ocr(digital_text)

    if requires_ocr:
        words = extract_ocr_blocks(page)
        lines = order_lines(group_words_into_lines(words))
        paragraphs = group_lines_into_paragraphs(lines)

        return {
            "document_id": str(uuid.uuid4()),
            "filename": filename,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "page_number": page_number,
            "extraction_method": "ocr",
            "text": "\n".join(line["text"] for line in lines),
            "raw_text": "\n".join(line["text"] for line in lines),
            "normalized_text": normalize_text("\n".join(line["text"] for line in lines)),
            "lines": lines,
            "paragraphs": paragraphs,
            "word_count": len(words)
        }
    lines, paragraphs = extract_digital_layout(page)
    words = []

    return {
        "document_id": str(uuid.uuid4()),
        "filename": filename,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "page_number": page_number,
        "extraction_method": "digital",

        # Preserve original extraction
        "raw_text": digital_text,
        "text": digital_text,

        # Cleaned text for downstream processing
        "normalized_text": normalize_text(digital_text),

        # Preserve layout and provenance
        "lines": lines,
        "paragraphs": paragraphs,
        "word_count": len(digital_text.split())
    }
