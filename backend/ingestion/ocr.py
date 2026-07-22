"""
Zone 2: Document Processing Service — OCR + Confidence Scoring Engine.

For scanned images/PDF pages, runs Tesseract OCR and computes a per-page
confidence score from Tesseract's own word-level confidence output. This
score travels with the chunk through embedding and directly discounts its
retrieval ranking later (see agent/nodes/retrieval.py).

For native (non-scanned) PDFs and text files, confidence is 1.0 — there's
no OCR step, so no OCR-related uncertainty.
"""

import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from PIL import Image


def ocr_page_with_confidence(image) -> tuple[str, float]:
    """Run OCR on a single PIL image, return (text, confidence 0.0-1.0)."""
    data = pytesseract.image_to_data(image, output_type=Output.DICT)

    words, confidences = [], []
    for text, conf in zip(data["text"], data["conf"]):
        if text.strip():
            words.append(text)
            # Tesseract reports -1 for non-text regions; skip those
            if conf != -1:
                confidences.append(float(conf))

    full_text = " ".join(words)
    avg_confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
    return full_text, avg_confidence


def ocr_scanned_pdf(pdf_path: str) -> list[dict]:
    """Convert each page of a scanned PDF to text with a per-page confidence score."""
    pages = convert_from_path(pdf_path)
    results = []
    for i, page_image in enumerate(pages):
        text, confidence = ocr_page_with_confidence(page_image)
        results.append({"page_number": i + 1, "text": text, "ocr_confidence": confidence})
    return results

def ocr_scanned_image(image_path: str) -> str:
    """OCR a single standalone image file (jpg, png, etc). Returns just the text —
    confidence scoring reuses the same word-level logic as PDF pages."""
    image = Image.open(image_path)
    text, confidence = ocr_page_with_confidence(image)
    return text, confidence