import csv
import uuid
from pathlib import Path
from datetime import datetime, timezone

from src.preprocessing.text_normalizer import normalize_text
from src.verification.extraction_quality import (
    evaluate_poor_quality,
    evaluate_document_quality
)

def extract_csv (file_path):
    file_path = Path(file_path)
    document_id = str(uuid.uuid4())

    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            delimeter = dialect.delimiter
        except csv.Error:
            delimeter = ","

        reader = csv.DictReader(f, delimiter=delimeter)

        if not reader.fieldnames:
            raise ValueError("csv file has no headers")
        headers = reader.fieldnames
        if any(header is None or not header.strip() for header in headers):
            raise ValueError("CSV contains empty column headers")

        rows = []
        lines = []

        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"CSV row {row_number} has extra columns"
                )
            record = {
                key: value
                for key, value in row.items()
            }

            rows.append({
                "row_number": row_number,
                "values": record
            })

            text = " | ".join(
                f"{key}: {value}" for key, value in record.item()
            )

            lines.append({
                "text": text,
                "bbox": None,
                "confidence": None,
                "line_id": [row_number, 0, 0],
                "row_number": row_number,
                "font_size": None,
                "is_bold": False
            })

        text = "\n".join(line["text"] for line in lines)

        page = {
            "document_id": document_id,
            "filename": file_path.name,
            "page_number": 1,
            "extraction_method": "csv",
            "text": text,
            "raw_taxt": text,
            "normalized_text": normalize_text(text),
            "lines": lines,
            "paragraphs": [],
            "word_count": len(text.split()),
        }

        page["quality"] = evaluate_poor_quality(page)

        return {
            "document_id": document_id,
            "filename": file_path.name,
            "file_type": "csv",
            "page_count": 1,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "quality": evaluate_document_quality([page]),
            "pages": [page],
            "table": {
                "columns": headers,
                "row_count": len(rows),
                "rows": rows
            }
        }