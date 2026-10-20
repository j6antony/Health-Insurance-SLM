from src.extraction.reading_order import order_lines

def extract_digital_layout(page):
    data = page.get_text("dict")

    lines = []
    paragraphs = []

    for block_number, block in enumerate(data["blocks"]):
        if block["type"] != 0:
            continue

        block_lines = []

        for line_number, line in enumerate(block["lines"]):
            text = " ".join(
                span["text"] for span in line["spans"]
            ).strip()

            if not text:
                continue


            line_data = {
                "text": text,
                "bbox": list(line["bbox"]),
                "confidence": None,
                "line_id": [block_number, 0, line_number],

                # Font information
                "font_size": max(
                    span["size"] for span in line["spans"]
                ),
                "is_bold": any(
                    "bold" in span["font"].lower()
                    for span in line["spans"]
                )
            }


            lines.append(line_data)
            block_lines.append(line_data)

        if not block_lines:
            continue

        paragraphs.append({
            "paragraph_id": [block_number, 0],
            "text": " ".join(
                line["text"] for line in block_lines
            ),
            "bbox": list(block["bbox"])
        })

    return order_lines(lines), paragraphs
