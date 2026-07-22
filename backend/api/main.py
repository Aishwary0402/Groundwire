"""
Groundwire API.

Two endpoints only, on purpose:
  POST /query   — run a question through the agent, get back a structured result
  POST /upload  — ingest a document (native or scanned) into the knowledge base

No auth/gateway layer here — that's explicitly out of scope for the hackathon
build (see PRD section 7, Assumptions and Constraints). Add it back in front
of this app with a reverse proxy if you need it later.
"""

import os
import shutil
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

load_dotenv()

from agent.graph import groundwire_agent  # noqa: E402  (must load env vars first)
from ingestion.chunking import ingest_native_document, ingest_scanned_pdf, ingest_scanned_image, ingest_text_file

app = FastAPI(title="Groundwire", version="0.1.0")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    llm_provider: str = "gemini" 


class QueryResponse(BaseModel):
    decision: str
    decision_reasoning: str
    answer: str | None = None
    clarification_question: str | None = None
    low_confidence_caveat: str | None = None
    citations: list[str] = []
    retry_count: int = 0


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    initial_state = {
        "query": request.query,
        "original_query": request.query,
        "retry_count": 0,
        "llm_provider": request.llm_provider,
    }

    result = groundwire_agent.invoke(initial_state)

    return QueryResponse(
        decision=result.get("decision", "final_answer"),
        decision_reasoning=result.get("decision_reasoning", ""),
        answer=result.get("verified_answer"),
        clarification_question=result.get("clarification_question"),
        low_confidence_caveat=result.get("low_confidence_caveat"),
        citations=result.get("citations", []),
        retry_count=result.get("retry_count", 0),
    )


@app.post("/upload")
async def upload(file: UploadFile = File(...), scanned: bool = False):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in (".jpg", ".jpeg", ".png", ".tiff", ".bmp"):
            chunk_count = ingest_scanned_image(tmp_path, file.filename)
        elif ext == ".txt":
            chunk_count = ingest_text_file(tmp_path, file.filename)
        elif scanned:
            chunk_count = ingest_scanned_pdf(tmp_path, file.filename)
        else:
            chunk_count = ingest_native_document(tmp_path, file.filename)
    finally:
        os.remove(tmp_path)

    return {"filename": file.filename, "chunks_indexed": chunk_count}

@app.get("/health")
def health():
    return {"status": "ok"}

from fastapi.staticfiles import StaticFiles
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
