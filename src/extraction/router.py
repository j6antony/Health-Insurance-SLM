from pathlib import Path

from .csv_extractor import extract_csv
from .pdf_extractor import extract_pdf_text

def extract_document(file_path):
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return extract_pdf_text(file_path)
    elif extension == ".csv":
        return extract_csv(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )