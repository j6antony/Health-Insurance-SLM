
from src.preprocessing.document_builder import process_document
from src.preprocessing.section_detector import detect_sections


result = process_document(
    "data/raw/reference_pdfs/cigna_hospital_care_checklist.pdf"
)

for page in result["pages"]:
    sections = detect_sections(page["lines"])

    print(f"\nPage {page['page_number']}")
    print(f"Detected sections: {len(sections)}")

    for section in sections:
        print(f"\nSECTION: {section['heading']}")
        print("Content:")
        print(" ".join(section["content"])[:300])

