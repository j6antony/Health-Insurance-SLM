
import json
from pathlib import Path
from src.preprocessing.text_normalizer import normalize_text
from src.preprocessing.document_builder import process_document


file_path = "./data/raw/reference_pdfs/cigna_hospital_care_checklist.pdf"

result = process_document(file_path)


for page in result["pages"]:
    print(f"\n--- Page {page['page_number']} ---")

    print("Extraction method:", page["extraction_method"])

    print("\nRaw text preview:")
    print(page["raw_text"][:300])

    print("\nNormalized text preview:")
    print(page["normalized_text"][:300])

    print("\nQuality:", page["quality"]["status"])

    assert page["normalized_text"] == normalize_text(
        page["raw_text"]
    )


print("Document ID:", result["document_id"])
print("Filename:", result["filename"])
print("Total pages:", result["page_count"])

for page in result["pages"]:
    print(
        f"Page {page['page_number']} | "
        f"Method: {page['extraction_method']} | "
        f"Lines: {len(page['lines'])} | "
        f"Paragraphs: {len(page['paragraphs'])}"
    )

# Save document output as JSON
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "test_document.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\nSaved JSON to: {output_path}")

print(
    f"Quality: {page['quality']['status']} | "
    f"Issues: {page['quality']['issues']} | "
    f"OCR Confidence: {page['quality']['average_ocr_confidence']}"
)

print("\n--- Document Quality ---")
print("Status:", result["quality"]["status"])
print("Total pages:", result["quality"]["total_pages"])
print("Passed pages:", result["quality"]["passed_pages"])
print("Flagged pages:", result["quality"]["flagged_pages"])
print("Issues:", result["quality"]["issues"])