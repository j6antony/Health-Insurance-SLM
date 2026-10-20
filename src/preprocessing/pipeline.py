import json
from pathlib import Path

from src.preprocessing.document_builder import process_document
from src.slm.heading_classifier import classify_document_lines
from src.preprocessing.section_detector import segment_with_slm
from src.preprocessing.section_chunker import chunk_sections
from src.slm.document_classifier import classify_document
from src.slm.field_extractor import extract_document_fields
def preprocess_document(file_path, output_path= None, max_words = 300):
    document = process_document(file_path)

    all_chunks = []

    for page in document["pages"]:
        lines = page["lines"]

        if not lines:
            page["sections"] = []
            page["chunks"] = []
            continue
        predictions = classify_document_lines(lines)

        sections = segment_with_slm(
            lines,
            predictions["predictions"]
        )

        chunks = chunk_sections (
            sections,
            max_words = max_words
        )
        for chunk in chunks:
            chunk["document_id"] = document["document_id"]
            chunk["page_number"] = page["page_number"]
            chunk["chunk_id"] = (
                f"page_{page["page_number"]}_{chunk["chunk_id"]}"
            )

        page["heading_predictions"] = predictions["predictions"]
        page["sections"] = sections
        page["chunks"] = chunks

        all_chunks.extend(chunks)

    document["chunks"] = all_chunks
    document["chunk_count"] = len(all_chunks)

    classification = classify_document(document)
    document["classification"] = classification

    extraction = extract_document_fields(document=document)

    document["extraction"] = extraction

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                document,
                f,
                indent=2,
                ensure_ascii=False
            )
    return document
    
