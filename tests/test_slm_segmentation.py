
from src.preprocessing.document_builder import process_document
from src.slm.heading_classifier import classify_document_lines
from src.preprocessing.section_detector import segment_with_slm


FILE_PATH = (
    "./data/raw/reference_pdfs/"
    "cigna_hospital_care_checklist.pdf"
)


def test_slm_segmentation():
    document = process_document(FILE_PATH)

    for page in document["pages"]:
        lines = page["lines"]

        print(f"\nProcessing Page {page['page_number']}")
        print(f"Total lines: {len(lines)}")

        if not lines:
            continue

        # Step 1: SLM heading classification
        result = classify_document_lines(lines)

        # Step 2: Deterministic segmentation
        sections = segment_with_slm(
            lines,
            result["predictions"]
        )

        print(f"Detected sections: {len(sections)}")

        for section in sections:
            print("\n-----------------------------")
            print("Heading:", section["heading"])
            print("Type:", section["heading_type"])
            print("Heading Line ID:", section["heading_line_id"])

            print("Content:")
            print(" ".join(section["content"])[:400])

        # Ensure no original lines were lost
        heading_ids = {
            s["heading_line_id"]
            for s in sections
            if s["heading_line_id"] is not None
        }

        content_ids = {
            item["line_id"]
            for section in sections
            for item in section["source_lines"]
        }

        assert heading_ids | content_ids == set(
            range(1, len(lines) + 1)
        ), "Some source lines were lost"

        assert not heading_ids & content_ids, (
            "Lines duplicated between headings and content"
        )

        print("\nSegmentation validation passed!")


if __name__ == "__main__":
    test_slm_segmentation()
