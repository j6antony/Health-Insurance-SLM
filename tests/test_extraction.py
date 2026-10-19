from src.extraction.pdf_extractor import extract_pdf_text

file_path = "./data/raw/reference_pdfs/cigna_hospital_care_checklist.pdf"

pages = extract_pdf_text(file_path)


print(f"Total pages: {len(pages)}")
for page in pages:
    print(f"\n--- Page {page['page_number']} ---")
    print(f"Characters: {page['character_count']}")
    print(f"Extractions Method: {page['extraction_method']}")
    print(f"Text preview: {page['text'][:500]}")


