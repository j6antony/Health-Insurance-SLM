def evaluate_poor_quality(page, min_characters = 50, min_confidence = 70):
    text = page["text"].strip()
    method = page["extraction_method"]

    issues = []

    if len(text) == 0:
        issues.append("empty_text")
    elif len(text) < min_characters:
        issues.append("insufficient_text")

    if method.lower() == "ocr":
        confidence = [
            line["confidence"] for line in page["lines"]
            if line["confidence"] is not None
        ]
        if confidence:
            avg_confidence = sum(confidence) / len(confidence)

            if avg_confidence < min_confidence:
                issues.append("low_ocr_confidence")
        else:
            avg_confidence = None
            issues.append("missing_ocr_confidence")
    else:
        avg_confidence = None
    return {
        "status": "review_required" if issues else "passed",
        "issues": issues,
        "average_ocr_confidence": avg_confidence,
        "character_count": len(text)
    }
def evaluate_document_quality(pages):
    total_pages = len(pages)

    passed_pages = 0
    flagged_pages = []
    all_issues = set()

    for page in pages:
        quality = page["quality"]

        if quality["status"] == "passed":
            passed_pages += 1
        else:
            flagged_pages.append(page["page_number"])
        all_issues.update(quality["issues"])
    if total_pages == 0:
        status = "review_required"
        all_issues.add("no_pages")
    elif flagged_pages:
        status = "review_required"
    else:
        status = "passed"
    
    return {
        "status": status,
        "total_pages": total_pages,
        "passed_pages": passed_pages,
        "flagged_pages": flagged_pages,
        "issues": sorted(all_issues)
    }
