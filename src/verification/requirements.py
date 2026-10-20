
def evaluate_requirement(rule, documents):
    """
    Evaluate documentation requirements.

    Supported types:
    - document
    - field
    - document_with_fields
    - AND
    - OR
    """

    operation = rule["type"]

    if operation == "document":
        required_type = rule["document_type"]

        matched = [
            doc for doc in documents
            if doc["document_type"] == required_type
        ]

        return {
            "satisfied": bool(matched),
            "required": required_type
        }

    if operation == "field":
        required_field = rule["field_name"]

        matched = [
            doc for doc in documents
            if required_field in doc["fields"]
        ]

        return {
            "satisfied": bool(matched),
            "required": required_field
        }

    if operation == "document_with_fields":
        required_type = rule["document_type"]
        required_fields = set(rule["fields"])

        # Both document type and required fields
        # must match within the SAME document.
        matched = [
            doc for doc in documents
            if (
                doc["document_type"] == required_type
                and required_fields.issubset(doc["fields"])
            )
        ]

        return {
            "satisfied": bool(matched),
            "required_document": required_type,
            "required_fields": sorted(required_fields)
        }

    if operation in {"AND", "OR"}:
        children = [
            evaluate_requirement(child, documents)
            for child in rule["requirements"]
        ]

        satisfied = (
            all(child["satisfied"] for child in children)
            if operation == "AND"
            else any(child["satisfied"] for child in children)
        )

        return {
            "satisfied": satisfied,
            "operation": operation,
            "children": children
        }

    raise ValueError(f"Unknown requirement type: {operation}")
