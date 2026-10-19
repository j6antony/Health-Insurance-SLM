
import pymupdf

from src.ocr.tesseract_ocr import (
    extract_ocr_blocks,
    group_words_into_lines,
    group_lines_into_paragraphs
)

with pymupdf.open("/Users/johanantony/Desktop/health-insurance-slm/data/raw/reference_pdfs/cigna_hospital_care_checklist.pdf") as doc:
    words = extract_ocr_blocks(doc[0])

lines = group_words_into_lines(words)
paragraphs = group_lines_into_paragraphs(lines)

print(f"Total words: {len(words)}")
print(f"Total lines: {len(lines)}")
print(f"Total paragraphs: {len(paragraphs)}")

for paragraph in paragraphs[:15]:
    print(f"\nParagraph {paragraph['paragraph_id']}")
    print(paragraph["text"])

print("\nFirst 5 normalized bounding boxes:")

for word in words[:5]:
    print(word["text"], word["bbox"])


with pymupdf.open("/Users/johanantony/Desktop/health-insurance-slm/data/raw/reference_pdfs/cigna_hospital_care_checklist.pdf") as doc:
    page = doc[0]

    for word in words:
        x0, y0, x1, y1 = word["bbox"]

        assert 0 <= x0 <= x1 <= page.rect.width
        assert 0 <= y0 <= y1 <= page.rect.height

print("All bounding boxes are within page boundaries!")