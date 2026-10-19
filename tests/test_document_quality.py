
from src.verification.extraction_quality import (
    evaluate_document_quality
)

sample_pages = [
    {
        "page_number": 1,
        "quality": {
            "status": "passed",
            "issues": []
        }
    },
    {
        "page_number": 2,
        "quality": {
            "status": "review_required",
            "issues": ["low_ocr_confidence"]
        }
    },
    {
        "page_number": 3,
        "quality": {
            "status": "passed",
            "issues": []
        }
    }
]

result = evaluate_document_quality(sample_pages)

print(result)

assert result["status"] == "review_required"
assert result["passed_pages"] == 2
assert result["flagged_pages"] == [2]

print("Document quality test passed!")
