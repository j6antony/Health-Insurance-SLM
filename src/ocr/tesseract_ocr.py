import pymupdf
import pytesseract
from PIL import Image
from pytesseract import Output
from src.preprocessing.coordinates import pixels_to_pdf_bbox
from collections import defaultdict
def extract_ocr_text(page):
    #render pdf page into an image at 300 DPI(dots per inch)
    pix = page.get_pixmap(dpi = 300)

    #convert the image into a tesseract object
    image = Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples        
    )
    text = pytesseract.image_to_string(image=image)
    return text
def extract_ocr_blocks(page):
    pix = page.get_pixmap(dpi = 300)

    image = Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples
    )

    data = pytesseract.image_to_data(
        image, 
        output_type=Output.DICT
    )
    

    blocks = []

    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        pixel_bbox = [
            data["left"][i],
            data["top"][i],
            data["left"][i] + data["width"][i],
            data["top"][i] + data["height"][i]
        ]

        pdf_bbox = pixels_to_pdf_bbox(
            pixel_bbox,
            page,
            dpi=300
        )
        blocks.append({
            "text": text,
            "bbox": pdf_bbox,
            "confidence": float(data["conf"][i]),

            # Tesseract's detected layout hierarchy
            "block_num": data["block_num"][i],
            "par_num": data["par_num"][i],
            "line_num": data["line_num"][i]
        })

    return blocks



from collections import defaultdict


def group_words_into_lines(words):
    grouped = defaultdict(list)

    for word in words:
        line_id = (
            word["block_num"],
            word["par_num"],
            word["line_num"]
        )

        grouped[line_id].append(word)

    lines = []

    for line_id, line_words in grouped.items():
        line_words.sort(key=lambda w: w["bbox"][0])

        text = " ".join(w["text"] for w in line_words)

        bbox = [
            min(w["bbox"][0] for w in line_words),
            min(w["bbox"][1] for w in line_words),
            max(w["bbox"][2] for w in line_words),
            max(w["bbox"][3] for w in line_words)
        ]

        confidence = sum(
            w["confidence"] for w in line_words
        ) / len(line_words)

        lines.append({
            "text": text,
            "bbox": bbox,
            "confidence": round(confidence, 2),
            "line_id": line_id
        })

    # Preserve Tesseract's detected block/paragraph/line order
    lines.sort(key=lambda line: line["line_id"])

    return lines


def group_lines_into_paragraphs(lines):
    paragraphs = {}

    for line in lines:
        block_num, par_num, line_num = line["line_id"]
        paragraph_id = (block_num, par_num)

        if paragraph_id not in paragraphs:
            paragraphs[paragraph_id] = []

        paragraphs[paragraph_id].append(line)

    results = []

    for paragraph_id, paragraph_lines in paragraphs.items():
        paragraph_lines.sort(
            key=lambda line: line["line_id"][2]
        )

        text = " ".join(
            line["text"] for line in paragraph_lines
        )

        bbox = [
            min(line["bbox"][0] for line in paragraph_lines),
            min(line["bbox"][1] for line in paragraph_lines),
            max(line["bbox"][2] for line in paragraph_lines),
            max(line["bbox"][3] for line in paragraph_lines)
        ]

        results.append({
            "paragraph_id": paragraph_id,
            "text": text,
            "bbox": bbox
        })

    results.sort(key=lambda p: p["paragraph_id"])

    return results
