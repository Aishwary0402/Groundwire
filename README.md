# Groundwire

A self-correcting RAG agent. Instead of always answering, it decides — per query —
whether to answer, re-query, ask you a clarifying question, or hedge with a caveat.

This is the working implementation of the architecture you designed for the OneInbox AI
Internship Hackathon 2026. The Decision Router (`backend/agent/nodes/decision.py`) is the
core piece: an explicit, tunable rule table, not a black-box LLM judgment call.

## What's real vs. simplified here

Built at full fidelity (this is your actual innovation, so it's not stubbed):
- OCR confidence scoring at ingestion, propagated into retrieval ranking
- Evidence Intelligence (sufficiency scoring + contradiction detection)
- The four-path Decision Router with explicit thresholds
- The retry-capped re-query loop
- Chain-of-Verification on generated answers
- The evaluation harness (self-correction ON vs OFF)

Simplified for the hackathon build (per your PRD's own scope notes):
- No auth/API gateway layer — add a reverse proxy in front if you need it later
- ChromaDB instead of Qdrant (same interface, no separate service to run)
- No Prometheus/Grafana — the graph's own state IS your trace; add LangSmith
  tracing later if you want it (one env var, `LANGCHAIN_TRACING_V2=true`)

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then paste your OpenAI key into .env
```

You'll also need the Tesseract binary installed system-wide (not a pip package) for OCR:
- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt install tesseract-ocr poppler-utils`
- Windows: install from https://github.com/UB-Mannheim/tesseract/wiki, add to PATH

## Run it

```bash
uvicorn api.main:app --reload --port 8000
```

Upload a document:
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/document.pdf" \
  -F "scanned=false"   # set true for scanned/image-based PDFs
```

Ask a question:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund window?"}'
```

## Run the evaluation harness

Upload a few real documents first (mix of native + scanned, ideally with at least one
genuine contradiction between two documents), then:

```bash
python -m eval.harness
```

This runs all 15 test questions in `eval/test_questions.json` through both a naive RAG
baseline and the full Groundwire agent, and writes `eval/results.md` with the
correct-refusal rate and latency comparison.

**Note:** `test_questions.json` currently has generic placeholder questions. Replace them
with questions that actually match whatever documents you upload — the categories
(answerable / insufficient_context / contradictory_context / ocr_degraded_source /
ambiguous_question) should stay, but the specific queries need to reference real content
in your document set or the eval numbers won't mean anything.

## Project structure

```
backend/
├── agent/
│   ├── state.py           # shared state schema — the agent's "memory" per query
│   ├── graph.py            # StateGraph wiring — the whole architecture in one file
│   └── nodes/
│       ├── retrieval.py    # OCR-weighted hybrid retrieval
│       ├── evidence.py     # sufficiency + contradiction detection
│       ├── decision.py     # the Decision Router — read this one first
│       ├── refinement.py   # query reformulation for the re-query loop
│       ├── clarify.py      # generates the clarifying question
│       └── generation.py   # answer generation + Chain-of-Verification
├── ingestion/
│   ├── ocr.py               # Tesseract OCR + confidence scoring
│   ├── chunking.py          # chunking + indexing, tags chunks with ocr_confidence
│   └── vectorstore.py       # ChromaDB wrapper
├── api/
│   └── main.py              # FastAPI — /query, /upload, /health
└── eval/
    ├── test_questions.json
    └── harness.py
```

## Next steps

1. Get real test documents in — a mix of clean PDFs, scanned/low-quality PDFs, and at
   least one pair of documents that genuinely contradict each other. Without contradictory
   documents in your corpus, the `low_confidence` path will never actually trigger.
2. Update `eval/test_questions.json` to reference your real documents.
3. Run `python -m eval.harness` and drop the resulting numbers into your PRD's Success
   Metrics section.
4. Frontend comes after the backend is verified working end to end — don't start it until
   `/query` returns sensible results for all four decision paths.
