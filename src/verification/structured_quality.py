import json
from collections import Counter

def make_issue(issue_type, location, message, field=None):
     return {
        "type": issue_type,
        "location": location,
        "field": field,
        "message": message
    }

def validate_csv(data):
    issues = []
    columns = data["columns"]
    rows = data["rows"]

    if not rows:
        issues.append(make_issue(
            "empty_dataset",
            "file",
            "CSV contains no data rows"
        ))

    duplicates = [
        name
        for name, count in Counter(columns).items()
        if count > 1
    ]

    for column in duplicates:
        issues.append(make_issue(
            "duplicate_column",
            "header",
            f"Duplicate column: {column}",
            column
        ))

    seen = set()

    for row in rows:
        row_number = row["row_number"]
        values = row["values"]

        # Check missing values
        for column, value in values.items():
            if value is None or str(value).strip() == "":
                issues.append(make_issue(
                    "missing_value",
                    f"row_{row_number}",
                    f"Empty value in column {column}",
                    column
                ))

        # Create a unique signature for the row
        signature = json.dumps(
            values,
            sort_keys=True,
            default=str
        )

        # Check if the signature was encountered before
        if signature in seen:
            issues.append(make_issue(
                "duplicate_row",
                f"row_{row_number}",
                "Duplicate record detected"
            ))

        # Add the current row AFTER checking
        seen.add(signature)

    return issues

def validate_excel(data):
    issues = []

    for sheet in data["sheets"]:
        sheet_name = sheet["sheet_name"]
        rows = sheet["rows"]

        if not rows:
            issues.append(make_issue(
                "empty_sheet", sheet_name, "Excel sheet is empty"
            ))
            continue
        headers = rows[0]["values"]

        for index, header in enumerate(headers, start = 1):
            if header is None or str(header).strip() == "":
                issues.append(make_issue(
                    "missing_header", f"{sheet_name}!row_1", f"Missing header at column {index}"
                ))
        for row in rows[1:]:
            values = row["values"]

            for index, value in enumerate(values, start = 1):
                if value is None or str(value).strip() == "":
                    issues.append(make_issue(
                        "missing_value",
                        f"{sheet_name}!row_{row["row_number"]}",
                        f"Empty value in column {index}"
                    ))

    return issues
def validate_json(data):
    issues = []

    if data is None or data == {} or data == []:
        issues.append(make_issue(
            "empty_json", "root",
            "JSON contains no data"
        ))

    return issues


def validate_text(text):
    if not text.strip():
        return [make_issue(
            "empty_document", "file",
            "No readable text extracted"
        )]

    return []


def validate_structured_document(document):
    file_type = document["file_type"]
    data = document.get("structured_data")

    if file_type == "csv":
        issues = validate_csv(data)

    elif file_type == "xlsx":
        issues = validate_excel(data)

    elif file_type == "json":
        issues = validate_json(data)

    elif file_type in {"txt", "docx"}:
        text = "\n".join(
            page["text"] for page in document["pages"]
        )
        issues = validate_text(text)

    else:
        raise ValueError(
            f"Unsupported structured file type: {file_type}"
        )

    return {
        "status": "review_required" if issues else "passed",
        "issue_count": len(issues),
        "issues": issues
    }
