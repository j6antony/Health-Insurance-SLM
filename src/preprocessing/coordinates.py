def pixels_to_pdf_bbox(bbox, page, dpi=300):
    scale = 72 / dpi

    return [
        round (coordinate * scale, 2) for coordinate in bbox
    ]
