
import re
from statistics import median


# ============================================================
# 1. RULE-BASED HEADING DETECTION (BASELINE)
# ============================================================

def is_section_heading(line, body_font_size=10):
    """
    Identify potential headings using formatting heuristics.
    Used as a baseline for comparison with the SLM.
    """

    text = line["text"].strip()
    words = text.split()

    if not text or len(words) > 8:
        return False

    # Exclude logical separators
    if text.upper() in {"OR", "AND", "N/A", "YES", "NO"}:
        return False

    # Exclude bullet points
    if text.startswith(("•", "-", "*", "◦")):
        return False

    # Exclude parenthetical descriptions
    if text.startswith("("):
        return False

    # Exclude numeric or monetary values
    if re.fullmatch(r"[\d\s.,:$%-]+", text):
        return False

    # Exclude complete sentences
    if text.endswith((".", "?", "!")):
        return False

    font_size = line.get("font_size")
    if font_size is None:
        font_size = body_font_size

    is_bold = line.get("is_bold", False)

    # Compare font size with typical body font
    larger_font = font_size >= body_font_size * 1.2

    if larger_font and is_bold:
        return True

    if is_bold and len(words) <= 4:
        return True

    return False


# ============================================================
# 2. RULE-BASED SECTION SEGMENTATION (BASELINE)
# ============================================================

def detect_sections(lines):
    """
    Group lines into sections using rule-based heading detection.
    """

    if not lines:
        return []

    font_sizes = [
        line["font_size"]
        for line in lines
        if line.get("font_size") is not None
    ]

    body_font_size = median(font_sizes) if font_sizes else 10

    sections = []

    current_section = {
        "heading": "Document Start",
        "content": [],
        "bbox": None
    }

    for line in lines:
        text = line["text"].strip()

        if not text:
            continue

        if is_section_heading(line, body_font_size):

            if (
                current_section["content"]
                or current_section["heading"] != "Document Start"
            ):
                sections.append(current_section)

            current_section = {
                "heading": text,
                "content": [],
                "bbox": line.get("bbox")
            }

        else:
            current_section["content"].append(text)

    if (
        current_section["content"]
        or current_section["heading"] != "Document Start"
    ):
        sections.append(current_section)

    return sections


# ============================================================
# 3. SLM-BASED SECTION SEGMENTATION (MAIN PIPELINE)
# ============================================================

VALID_LABELS = {
    "document_header",
    "document_title",
    "section_heading",
    "subsection_heading",
    "not_heading"
}

HEADING_LABELS = {
    "section_heading",
    "subsection_heading"
}

def is_valid_section_boundary(line, label):
    """
    Validate whether an SLM-predicted heading
    is structurally suitable as a section boundary.
    """

    if label not in HEADING_LABELS:
        return False

    text = line["text"].strip()

    if not text:
        return False

    # Bullet points should remain list items
    if text.startswith(("•", "◦", "*")):
        return False

    # Exclude logical separators
    if text.upper() in {"OR", "AND", "YES", "NO"}:
        return False

    # Exclude ordinary sentences
    if text.endswith((".", "?", "!")):
        return False

    return True


def segment_with_slm(lines, predictions):
    """
    Segment document lines using SLM heading predictions.

    The SLM identifies headings.
    Python groups the content into sections.

    Args:
        lines:
            List of extracted document lines.

        predictions:
            List of dictionaries containing:
            - line_id (1-based)
            - label

    Returns:
        List of structured sections with source references.
    """
        

    if not lines:
        if predictions:
            raise ValueError(
                "Predictions provided for an empty document"
            )
        return []

    # --------------------------------------------------------
    # Step 1: Validate SLM predictions
    # --------------------------------------------------------

    prediction_map = {}

    for prediction in predictions:
        line_id = prediction["line_id"]
        label = prediction["label"]

        if type(line_id) is not int:
            raise ValueError(
                f"Invalid line ID type: {line_id}"
            )

        if line_id in prediction_map:
            raise ValueError(
                f"Duplicate prediction ID: {line_id}"
            )

        if label not in VALID_LABELS:
            raise ValueError(
                f"Invalid SLM label: {label}"
            )

        prediction_map[line_id] = label

    expected_ids = set(range(1, len(lines) + 1))

    if set(prediction_map) != expected_ids:
        missing = expected_ids - set(prediction_map)
        unexpected = set(prediction_map) - expected_ids

        raise ValueError(
            f"Invalid SLM predictions. "
            f"Missing IDs: {sorted(missing)}, "
            f"Unexpected IDs: {sorted(unexpected)}"
        )

    # --------------------------------------------------------
    # Step 2: Initialize segmentation
    # --------------------------------------------------------

    sections = []

    current_section = {
        "heading": None,
        "heading_type": "document_start",
        "heading_line_id": None,
        "heading_bbox": None,
        "content": [],
        "source_lines": []
    }

    # --------------------------------------------------------
    # Step 3: Process document lines
    # --------------------------------------------------------

    for index, line in enumerate(lines, start=1):

        label = prediction_map[index]
        text = line["text"].strip()

        if is_valid_section_boundary(line, label):

            # Preserve the previous section
            if (
                current_section["source_lines"]
                or current_section["heading"] is not None
            ):
                sections.append(current_section)

            # Begin a new section
            current_section = {
                "heading": text,
                "heading_type": label,
                "heading_line_id": index,
                "heading_bbox": line.get("bbox"),
                "content": [],
                "source_lines": []
            }

        else:

            # Preserve original line metadata
            current_section["content"].append(text)

            current_section["source_lines"].append({
                "line_id": index,
                "text": text,
                "bbox": line.get("bbox"),
                "label": label
            })

    # --------------------------------------------------------
    # Step 4: Save final section
    # --------------------------------------------------------

    if (
        current_section["source_lines"]
        or current_section["heading"] is not None
    ):
        sections.append(current_section)

    return sections
