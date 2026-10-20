
import csv
import json
from datetime import date, datetime
from pathlib import Path

from docx import Document
from openpyxl import load_workbook


def to_json_value(value):
    """Convert spreadsheet values into JSON-compatible values."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def extract_csv(file_path):
    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(
                sample, delimiters=",;\t"
            )
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)

        if not reader.fieldnames:
            raise ValueError("CSV is missing column headers")

        if any(not h or not h.strip() for h in reader.fieldnames):
            raise ValueError("CSV contains empty column headers")

        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV contains duplicate headers")

        rows = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Extra columns in row {row_number}")

            rows.append({
                "row_number": row_number,
                "values": row
            })

    columns = reader.fieldnames

    lines = [
        (f"Row {row['row_number']}: " +
         " | ".join(f"{k}: {v}" for k, v in row["values"].items()))
        for row in rows
    ]

    return {
        "text": "\n".join(lines),
        "structured_data": {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows)
        }
    }


def extract_excel(file_path):
    workbook = load_workbook(
        file_path, read_only=True, data_only=True
    )

    sheets = []
    text_lines = []

    try:
        for sheet in workbook.worksheets:
            sheet_rows = []

            for row_number, row in enumerate(
                sheet.iter_rows(values_only=True), start=1
            ):
                values = [to_json_value(v) for v in row]

                sheet_rows.append({
                    "row_number": row_number,
                    "values": values
                })

                text_lines.append(
                    f"Sheet {sheet.title}, Row {row_number}: " +
                    " | ".join(
                        "" if v is None else str(v)
                        for v in values
                    )
                )

            sheets.append({
                "sheet_name": sheet.title,
                "rows": sheet_rows,
                "row_count": len(sheet_rows)
            })
    finally:
        workbook.close()

    return {
        "text": "\n".join(text_lines),
        "structured_data": {
            "sheets": sheets,
            "sheet_count": len(sheets)
        }
    }


def extract_json(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    return {
        "text": json.dumps(
            data, indent=2, ensure_ascii=False
        ),
        "structured_data": data
    }


def extract_txt(file_path):
    text = Path(file_path).read_text(encoding="utf-8-sig")

    return {
        "text": text,
        "structured_data": None
    }


def extract_word(file_path):
    document = Document(file_path)
    content = []
    tables = []

    # Extract paragraphs
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            content.append(paragraph.text)

    # Extract tables
    for table_number, table in enumerate(document.tables, start=1):
        table_rows = []

        content.append(f"Table {table_number}")

        for row_number, row in enumerate(table.rows, start=1):
            values = [cell.text for cell in row.cells]
            table_rows.append(values)

            content.append(
                f"Row {row_number}: " + " | ".join(values)
            )

        tables.append({
            "table_number": table_number,
            "rows": table_rows
        })

    return {
        "text": "\n".join(content),
        "structured_data": {
            "tables": tables
        }
    }


EXTRACTORS = {
    ".csv": extract_csv,
    ".xlsx": extract_excel,
    ".json": extract_json,
    ".txt": extract_txt,
    ".docx": extract_word,
}


def extract_non_pdf(file_path):
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension not in EXTRACTORS:
        raise ValueError(f"Unsupported file type: {extension}")

    return EXTRACTORS[extension](file_path)
