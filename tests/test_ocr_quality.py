
import pymupdf
from pathlib import Path
from jiwer import wer, cer
import difflib
from collections import Counter
import re

from src.extraction.pdf_extractor import extract_pdf_text


# Original PDF
original_path = Path("./data/raw/reference_pdfs/cigna_hospital_care_checklist.pdf")
scanned_path = Path("data/scanned_test.pdf")


# STEP 1: Create a scanned version automatically
source = pymupdf.open(original_path)
scanned = pymupdf.open()

for page in source:
    pix = page.get_pixmap(dpi=200)

    new_page = scanned.new_page(
        width=page.rect.width,
        height=page.rect.height
    )

    new_page.insert_image(new_page.rect, pixmap=pix)

scanned.save(scanned_path)
scanned.close()
source.close()

print("Scanned PDF generated successfully!")


# STEP 2: Extract text from both PDFs
# Read embedded text directly from the original PDF
with pymupdf.open(original_path) as doc:
    digital_text = "\n".join(
        page.get_text("text") for page in doc
    )

# Force OCR on the scanned PDF through our pipeline
ocr_pages = extract_pdf_text(scanned_path)

ocr_text = "\n".join(
    page["text"] for page in ocr_pages
)


# STEP 3: Normalize whitespace
reference = " ".join(digital_text.split())
prediction = " ".join(ocr_text.split())


# STEP 4: Evaluate OCR accuracy
print("\n--- OCR Quality Evaluation ---")
print(f"Word Error Rate (WER): {wer(reference, prediction):.4f}")
print(f"Character Error Rate (CER): {cer(reference, prediction):.4f}")

print(f"\nReference characters: {len(reference)}")
print(f"OCR characters: {len(prediction)}")
print(f"Extraction method: {ocr_pages[0]['extraction_method']}")


# STEP 5: Delete temporary scanned PDF
scanned_path.unlink()

print("\nEvaluation completed!")



# Compare the reference and OCR output
reference_words = reference.split()
ocr_words = prediction.split()

matcher = difflib.SequenceMatcher(
    None,
    reference_words,
    ocr_words,
    autojunk=False
)

print("\n--- OCR Differences ---")

shown = 0

for operation, i1, i2, j1, j2 in matcher.get_opcodes():
    if operation == "equal":
        continue

    print(f"\nDifference type: {operation}")
    print("Reference:", " ".join(reference_words[i1:i2])[:250])
    print("OCR:      ", " ".join(ocr_words[j1:j2])[:250])

    shown += 1

    if shown >= 10:
        break



def normalize_words(text):
    return re.findall(r"\b\w+\b", text.lower())


reference_words = Counter(normalize_words(reference))
ocr_words = Counter(normalize_words(prediction))

matched = sum(
    (reference_words & ocr_words).values()
)

total_reference = sum(reference_words.values())
total_ocr = sum(ocr_words.values())

precision = matched / total_ocr if total_ocr else 0
recall = matched / total_reference if total_reference else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

print("\n--- Order-Independent OCR Evaluation ---")
print(f"Word Precision: {precision:.4f}")
print(f"Word Recall:    {recall:.4f}")
print(f"Word F1:        {f1:.4f}")
