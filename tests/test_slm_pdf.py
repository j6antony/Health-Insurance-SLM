
from src.preprocessing.document_builder import process_document
from src.slm.heading_classifier import classify_document_lines


# PDF file path
FILE_PATH = "./data/raw/reference_pdfs/cigna_hospital_care_checklist.pdf"

# Number of lines to test
# Set to None to process the entire document
TEST_LIMIT = 5

VALID_LABELS = {
    "document_header",
    "document_title",
    "section_heading",
    "subsection_heading",
    "not_heading"
}


def test_slm_pdf():
    # Step 1: Process PDF
    document = process_document(FILE_PATH)

    for page in document["pages"]:
        lines = page["lines"]

        # Step 2: Select lines to classify
        if TEST_LIMIT is None:
            test_lines = lines
        else:
            test_lines = lines[:TEST_LIMIT]

        print(f"\nProcessing Page {page['page_number']}")
        print(f"Total lines: {len(lines)}")
        print(f"Lines being tested: {len(test_lines)}")

        if not test_lines:
            print("No lines found. Skipping page.")
            continue

        # Step 3: Run SLM classification
        result = classify_document_lines(test_lines)

        predictions = result["predictions"]

        # Step 4: Validate predictions
        expected_ids = set(range(1, len(test_lines) + 1))
        returned_ids = [
            prediction["line_id"]
            for prediction in predictions
        ]

        assert len(returned_ids) == len(expected_ids), (
            "SLM returned an incorrect number of predictions"
        )

        assert set(returned_ids) == expected_ids, (
            "SLM returned missing or invalid line IDs"
        )

        assert all(
            prediction["label"] in VALID_LABELS
            for prediction in predictions
        ), "SLM returned an invalid label"

        print("\nAll SLM predictions validated!")

        # Step 5: Display results
        print("\n--- SLM Heading Classification ---")

        for prediction in predictions:
            line_id = prediction["line_id"]
            label = prediction["label"]

            original_text = test_lines[line_id - 1]["text"]

            print(f"{line_id}. [{label}] {original_text}")

        # Step 6: Summary
        print("\n--- Classification Summary ---")

        for label in sorted(VALID_LABELS):
            count = sum(
                prediction["label"] == label
                for prediction in predictions
            )

            print(f"{label}: {count}")


if __name__ == "__main__":
    test_slm_pdf()
