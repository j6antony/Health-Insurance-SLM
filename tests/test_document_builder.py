import pymupdf

from src.preprocessing.document_builder import build_document

file_path = "data/raw/reference_pdfs/cigna_hospital_care_checklist.pdf"


with pymupdf.open(file_path) as doc:
    result = build_document(
        page=doc[0],
        filename="test_document.pdf",
        page_number=1
    )

print("Document ID:", result["document_id"])
print("Extraction method:", result["extraction_method"])
print("Word count:", result["word_count"])
print("Lines:", len(result["lines"]))
print("Paragraphs:", len(result["paragraphs"]))

print("\nText preview:")
print(result["text"][:500])
