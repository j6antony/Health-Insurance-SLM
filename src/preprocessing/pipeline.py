import json
from pathlib import Path

from src.preprocessing.document_builder import process_document
from src.slm.heading_classifier import classify_document_lines
from src.preprocessing.section_detector import segment_with_slm
from src.preprocessing.section_chunker import chunk_sections
from src.slm.document_classifier import classify_document
from src.verification.structured_field_validator import validate_structured_mapping
from src.slm.field_extractor import extract_document_fields
from src.verification.extraction_quality import evaluate_document_quality
from src.extraction.structured_field_mapper import map_structured_fields
def preprocess_document(file_path, output_path= None, max_words = 300):
   
    document = process_document(file_path)
    file_type = Path(file_path).suffix.lower()

    # Structured files remain ingestion-only for now.
    
    if file_type in {".csv", ".xlsx", ".json"}:
        document["chunks"] = []
        document["chunk_count"] = 0

        for page in document["pages"]:
            page["sections"] = []
            page["chunks"] = []

        mapping = map_structured_fields(document)
        mapping = validate_structured_mapping(mapping)

        document["structured_mapping"] = mapping

        # Record-level fields must NOT be flattened into
        # the document-level extraction schema.
        document["classification"] = {
            "document_type": "unknown",
            "evidence": [],
            "requires_review": True,
            "rejected_evidence": []
        }

        document["extraction"] = {
            "document_id": document["document_id"],
            "document_type": "unknown",
            "fields": [],
            "field_count": 0
        }

        document["processing_status"] = (
            "mapped"
            if any(
                record["fields"]
                for record in mapping["records"]
            )
            else "ingested"
        )

        # Quality and mapping findings still need review.
        document["requires_review"] = (
            document["quality"]["status"] == "review_required"
            or mapping["requires_review"]
            or any(
                record["unmapped_fields"]
                for record in mapping["records"]
            )
        )

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(
                parents=True, exist_ok=True
            )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    document,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

        return document

    # Word and text documents use the existing SLM.
    if file_type in {".txt", ".docx"}:
        # Empty or unreadable documents must not
        # continue to semantic processing.
        if not any(
            page["normalized_text"].strip()
            for page in document["pages"]
        ):
            document["processing_status"] = "requires_review"
            document["classification"] = {
                "document_type": "unknown",
                "evidence": [],
                "requires_review": True,
                "rejected_evidence": []
            }
            document["extraction"] = {
                "document_id": document["document_id"],
                "document_type": "unknown",
                "fields": [],
                "field_count": 0
            }
        else:
            document["classification"] = classify_document(
                document
            )

            document["extraction"] = extract_document_fields(
                document=document
            )

            document["processing_status"] = "processed"

        # Heading segmentation is intentionally skipped
        # until DOCX/TXT structure handling is improved.
        document["chunks"] = []
        document["chunk_count"] = 0

        for page in document["pages"]:
            page["sections"] = []
            page["chunks"] = []

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(
                parents=True, exist_ok=True
            )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    document, f, indent=2,
                    ensure_ascii=False
                )

        return document

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
                f"page_{page['page_number']}_{chunk['chunk_id']}"
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
    
