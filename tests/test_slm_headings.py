
from src.slm.heading_classifier import classify_document_structure


sample_text = """
1. Cigna Healthcare
2. Hospital Care claims checklist
3. Employee Basic Information
4. Name
5. Hospital information
6. OR
7. • Documentation outlining room and board charges
"""

result = classify_document_structure(sample_text)

print("\n--- SLM Classification ---")

for prediction in result["predictions"]:
    print(
        f"Line {prediction['line_id']}: "
        f"{prediction['label']}"
    )


# Validate output
assert len(result["predictions"]) == 7

expected_ids = set(range(1, 8))
returned_ids = {
    item["line_id"]
    for item in result["predictions"]
}

assert returned_ids == expected_ids

print("\nJSON output test passed!")
