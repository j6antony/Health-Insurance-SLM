import pymupdf
from src.ocr.tesseract_ocr import extract_ocr_text

def extract_pdf_text(file_path):
    document = pymupdf.open(file_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")
        extraction_method = "digital"    
        requires_ocr = needs_ocr(text, 50)
        if requires_ocr:
            text = extract_ocr_text(page)
            extraction_method = "ocr"

        pages.append({
            "page_number": page_number,
            "text": text,
            "character_count": len(text),
            "extraction_method": extraction_method,
            "needs_OCR":requires_ocr
        })
    document.close()
    return pages
# This is used to decide if OCR is required keep in mind that this is very basic and a better system will need to be implemented later
def needs_ocr(text, min_characters=50):
    cleaned_text = text.strip()
    if len(cleaned_text) <= min_characters:
        return True
    return False

