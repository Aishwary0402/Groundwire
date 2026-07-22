"""
Zone 3: Knowledge Prep Service — semantic chunking + indexing.

Handles both native text (via Unstructured) and OCR'd scanned pages.
Every chunk written to the vectorstore carries an ocr_confidence tag in
its metadata — this is the field the OCR-Aware Scorer reads at retrieval
time (see agent/nodes/retrieval.py).
"""

import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

from ingestion.vectorstore import get_vectorstore
from ingestion.ocr import ocr_scanned_pdf
from ingestion.ocr import ocr_scanned_pdf, ocr_scanned_image


CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def ingest_native_document(file_path: str, source_name: str) -> int:
    """Ingest a text-based PDF — no OCR needed, confidence is 1.0."""
    reader = PdfReader(file_path)
    full_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return _chunk_and_index(full_text, source_name, ocr_confidence=1.0)

def ingest_text_file(file_path: str, source_name: str) -> int:
    """Ingest a plain .txt file directly — used for quick contradiction testing."""
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    return _chunk_and_index(full_text, source_name, ocr_confidence=1.0)


def ingest_scanned_pdf(file_path: str, source_name: str) -> int:
    """Ingest a scanned PDF — OCR each page, keep per-page confidence."""
    pages = ocr_scanned_pdf(file_path)
    total_chunks = 0
    for page in pages:
        total_chunks += _chunk_and_index(
            page["text"],
            f"{source_name} (page {page['page_number']})",
            ocr_confidence=page["ocr_confidence"],
        )
    return total_chunks

def ingest_scanned_image(file_path: str, source_name: str) -> int:
    """Ingest a standalone scanned image (not embedded in a PDF)."""
    text, confidence = ocr_scanned_image(file_path)
    return _chunk_and_index(text, source_name, ocr_confidence=confidence)

def _chunk_and_index(text: str, source_name: str, ocr_confidence: float) -> int:
    if not text.strip():
        return 0

    chunks = splitter.split_text(text)
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "chunk_id": str(uuid.uuid4())[:8],
                "source_document": source_name,
                "ocr_confidence": ocr_confidence,
            },
        )
        for chunk in chunks
    ]

    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents)
    return len(documents)
