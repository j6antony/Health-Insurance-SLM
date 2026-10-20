
import os

import pytest

from src.slm.heading_classifier import classify_document_structure


@pytest.mark.skipif(
    os.environ.get("RUN_OLLAMA_TESTS") != "1",
    reason="Set RUN_OLLAMA_TESTS=1 to run Ollama integration tests."
)
def test_classify_document_structure():
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

    assert len(result["predictions"]) == 7
    assert {
        item["line_id"]
        for item in result["predictions"]
    } == set(range(1, 8))
