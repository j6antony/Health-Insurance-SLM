
from src.preprocessing.document_builder import process_document
from src.slm.heading_classifier import classify_document_lines
from src.preprocessing.section_detector import segment_with_slm
from src.preprocessing.section_chunker import chunk_sections


FILE_PATH = (
    "./data/raw/reference_pdfs/"
    "cigna_hospital_care_checklist.pdf"
)


def test_section_chunking():
    document = process_document(FILE_PATH)

    for page in document["pages"]:
        lines = page["lines"]

        if not lines:
            continue

        predictions = classify_document_lines(lines)["predictions"]
        sections = segment_with_slm(lines, predictions)

        chunks = chunk_sections(sections, max_words=100)

        print(f"\nPage {page['page_number']}")
        print(f"Sections: {len(sections)}")
        print(f"Chunks: {len(chunks)}")

        for chunk in chunks:
            print("\n----------------------")
            print("Chunk ID:", chunk["chunk_id"])
            print("Heading:", chunk["heading"])
            print("Words:", chunk["word_count"])
            print("Oversized:", chunk["oversized"])
            print("Text:")
            print(chunk["text"][:500])

        # Check that chunking did not lose content lines
        original_content_ids = [
            line["line_id"]
            for section in sections
            for line in section["source_lines"]
        ]

        chunk_content_ids = [
            line_id
            for chunk in chunks
            for line_id in chunk["source_lines"]
        ]

        assert sorted(original_content_ids) == sorted(chunk_content_ids)

        print("\nChunking validation passed!")


if __name__ == "__main__":
    test_section_chunking()
