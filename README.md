# Hedge

**A self-correcting RAG agent that hedges instead of hallucinating.**

Hedge is a Retrieval-Augmented Generation system built for messy, real-world document sets — scanned PDFs, low-quality OCR, inconsistent formatting, and contradictory sources. Instead of confidently synthesizing an answer from whatever it retrieves, it makes an explicit decision on every query: answer normally, hedge with a visible caveat, ask the user a clarifying question, or re-query with a reformulated search — and it can prove which one it chose and why.

Built for the OneInbox AI Internship Hackathon 2026.

---

## Table of contents

- [The problem](#the-problem)
- [What makes this different from standard RAG](#what-makes-this-different-from-standard-rag)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [How a query actually flows through the system](#how-a-query-actually-flows-through-the-system)
- [Setup](#setup)
- [Environment variables](#environment-variables)
- [Running locally](#running-locally)
- [API reference](#api-reference)
- [Frontend](#frontend)
- [Evaluation harness](#evaluation-harness)
- [Deployment](#deployment)
- [Known limitations and honest scope notes](#known-limitations-and-honest-scope-notes)
- [Roadmap](#roadmap)

---

## The problem

Standard RAG systems assume retrieved context is legible, relevant, and truthful. Over real document sets, that assumption breaks in three specific ways:

1. **Garbled OCR text gets treated as valid evidence.** A badly scanned page can have high vector similarity to a query and still be gibberish.
2. **Contradictions between sources get silently resolved.** If two documents disagree, standard RAG just picks one and states it as fact.
3. **Missing information produces a confident wrong answer instead of an honest "I don't know."**

Groundwire's job is to catch all three before they reach the user.

---

## What makes this different from standard RAG

- **OCR-confidence-weighted retrieval.** Every chunk is tagged at ingestion time with a confidence score derived from OCR quality (word-level Tesseract confidence). That score directly discounts the chunk's similarity ranking at retrieval time — a smudgy, low-confidence chunk can't outrank a clean one just because it's semantically closer.
- **A contradiction graph, not a single confidence number.** An Evidence Intelligence step checks retrieved chunks against each other for genuinely conflicting facts (different dates, different figures for the same claim) before anything reaches the answer generator.
- **An explicit, auditable Decision Router — not a black-box LLM judgment call.** The routing decision (`re_query` / `clarify` / `low_confidence` / `final_answer`) is made by a deterministic rule table reading four signals: sufficiency score, contradiction count, aggregate OCR confidence, and retry count. This is a deliberate choice: rules are debuggable and don't burn an extra LLM call on the hottest path in the graph.
- **A capped self-correction loop.** `re_query` reformulates the search and retries, hard-capped at 2 attempts, so the system can't loop forever burning tokens.
- **Chain-of-Verification on every generated answer.** A second pass checks the draft answer's claims against the cited evidence before it's returned.
- **Contradictions are surfaced in the answer itself, not just flagged in metadata.** When sources disagree, the answer text names both versions and their sources explicitly, rather than silently picking one and hiding the caveat in a side field.
- **Provider-switchable at the request level.** Every LLM call for a given query runs entirely on either Gemini or Groq, selectable per-request — no mixing providers within a single answer.

---

## Architecture

The system is organized into 8 logical zones. The hackathon-scope implementation collapses these into a single FastAPI service plus a plain HTML/CSS/JS frontend — see [Known limitations](#known-limitations-and-honest-scope-notes) for what's simplified vs. full production fidelity.

```
Zone 1 — Entry & Security       -> FastAPI app, CORS, static frontend mount
Zone 2 — Ingestion & OCR        -> Document upload, Tesseract OCR, confidence scoring
Zone 3 — Knowledge Base         -> Chunking, embeddings, ChromaDB vector store
Zone 4 — Reasoning Engine       -> Query understanding, hybrid retrieval, OCR-aware scoring
Zone 5 — Evidence Intelligence  -> Sufficiency scoring, contradiction detection
Zone 6 — Self-Correction Loop   -> Decision Router (the core innovation) + query refinement
Zone 7 — Generation             -> Prompt fusion, LLM generation, Chain-of-Verification
Zone 8 — Monitoring & Eval      -> Evaluation harness (correct-refusal rate, latency)
```

### The Decision Router — four outcomes, one query

```
                    +----------------------+
                    |   Decision Router    |
                    |  reads: sufficiency,  |
                    |  contradictions,      |
                    |  OCR confidence,      |
                    |  retry_count          |
                    +----------+-----------+
           +------------+------+------+------------+
           v            v             v             v
      re_query      clarify     low_confidence  final_answer
   (reformulate,   (ask user a   (answer with    (normal,
    max 2 retries,  specific     visible caveat   high-confidence
    loop back to    question)    + citations)     response)
    retrieval)
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph (`StateGraph`, conditional edges) |
| LLM providers | Google Gemini (`gemini-flash-lite-latest`), Groq (`llama-3.1-8b-instant` / `llama-3.3-70b-versatile`) |
| Embeddings | Google `gemini-embedding-001` |
| Vector store | ChromaDB (local, persisted to disk) |
| OCR | Tesseract (via `pytesseract`), `pdf2image` for scanned PDF pages |
| Native PDF text extraction | `pypdf` |
| Backend framework | FastAPI, Uvicorn |
| Evaluation | Custom harness (correct-refusal rate, latency comparison) |
| Frontend | Plain HTML, CSS, JavaScript — no build tools, no framework |
| Deployment | Docker, hosted on Render |

---

## Project structure

```
groundwire/
├── Dockerfile
├── README.md
├── .gitignore
├── backend/
│   ├── requirements.txt
│   ├── .env.example
│   ├── agent/
│   │   ├── state.py              # shared state schema passed between all graph nodes
│   │   ├── graph.py               # StateGraph wiring — the architecture as executable code
│   │   ├── llm_factory.py          # provider selection (Gemini/Groq), retry wrapping
│   │   └── nodes/
│   │       ├── retrieval.py         # hybrid retrieval + OCR-aware scoring
│   │       ├── evidence.py           # sufficiency scoring + contradiction detection
│   │       ├── decision.py            # the Decision Router — explicit rule table
│   │       ├── refinement.py           # query reformulation for the re-query loop
│   │       ├── clarify.py               # generates the clarifying question
│   │       └── generation.py             # answer generation + Chain-of-Verification
│   ├── ingestion/
│   │   ├── ocr.py                  # Tesseract OCR + confidence scoring
│   │   ├── chunking.py              # chunking + indexing, tags chunks with ocr_confidence
│   │   └── vectorstore.py            # ChromaDB wrapper
│   ├── api/
│   │   └── main.py                  # FastAPI app — /query, /upload, /health, serves frontend
│   ├── eval/
│   │   ├── test_questions.json        # test question set spanning 5 categories
│   │   └── harness.py                  # runs self-correction ON vs OFF, reports metrics
│   ├── scripts/
│   │   └── test_providers.py            # standalone smoke test for the LLM factory
│   └── data/
│       └── chroma/                        # persisted vector store (committed for demo readiness)
└── frontend/
    ├── index.html
    ├── css/
    │   └── style.css
    └── js/
        ├── api.js                   # all backend fetch calls, isolated to one file
        ├── chat.js                    # renders the four decision outcomes distinctly
        ├── upload.js                    # drag-drop + file picker upload handling
        └── main.js                       # app bootstrap, provider state, composer wiring
```

---

## How a query actually flows through the system

1. **User submits a query** via the UI or `POST /query`, along with a selected `llm_provider`.
2. **Hybrid retrieval** (`retrieval.py`) fetches candidate chunks from ChromaDB and re-ranks them by `similarity_score x ocr_confidence`.
3. **Evidence Intelligence** (`evidence.py`) makes one structured-output LLM call to assess:
   - `sufficiency_score` (0–1): does this evidence answer the query?
   - `contradictions`: any pairs of chunks asserting conflicting facts?
4. **Aggregate OCR confidence** is computed as the mean confidence across all retrieved chunks.
5. **The Decision Router** (`decision.py`) — pure Python, no LLM call — applies its rule table to these signals and returns exactly one of four decisions.
6. Depending on the decision:
   - `re_query` → `refinement.py` reformulates the query, loops back to step 2 (capped at 2 retries)
   - `clarify` → `clarify.py` generates a specific clarifying question, returned to the user
   - `low_confidence` → `generation.py` produces an answer that explicitly surfaces the conflict/uncertainty, with a caveat
   - `final_answer` → `generation.py` produces a normal answer
7. **Chain-of-Verification** runs on any generated answer — a second LLM pass checks the draft against its cited evidence before returning it.
8. The response, including `decision`, `decision_reasoning`, `citations`, and `retry_count`, is returned to the frontend, which renders it as a trace node colored by outcome.

---

## Setup

### Prerequisites

- Python 3.11+
- Tesseract OCR (system binary, not a pip package)
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt install tesseract-ocr poppler-utils`
  - Windows: install from the [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki), add to PATH
- A Google AI Studio API key ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)) — free
- A Groq API key ([console.groq.com/keys](https://console.groq.com/keys)) — free

### Install

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then paste your real keys into .env
```

---

## Environment variables

Set these in `backend/.env` for local development, or as **secrets** (not plain variables) in your deployment platform.

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Gemini API key — used for chat when `llm_provider=gemini`, and always used for embeddings regardless of provider selection (Groq has no embeddings endpoint) |
| `GROQ_API_KEY` | Yes | Groq API key — used for chat when `llm_provider=groq` |

---

## Running locally

```bash
cd backend
python -m uvicorn api.main:app --reload --port 8000
```

Visit `http://localhost:8000` — FastAPI serves the frontend directly from this same origin (see the `StaticFiles` mount at the end of `api/main.py`), so there's no separate frontend server to run.

### Smoke-test the LLM providers in isolation

```bash
python -m scripts.test_providers
```

Confirms both Gemini and Groq authenticate correctly and the retry wrapper is functioning, without touching the full agent graph.

---

## API reference

### `POST /upload`

Ingests a document into the vector store. Routes automatically by file extension:

| Extension | Handling |
|---|---|
| `.pdf` (default) | Native text extraction via `pypdf`, confidence = 1.0 |
| `.pdf` with `scanned=true` | OCR via Tesseract per page, per-page confidence score |
| `.jpg` / `.jpeg` / `.png` / `.tiff` / `.bmp` | OCR via Tesseract, single confidence score |
| `.txt` | Direct text ingestion, confidence = 1.0 (useful for quick contradiction testing) |

**Request:** `multipart/form-data`
- `file`: the document
- `scanned` (optional, bool, default `false`): force OCR path for PDFs

**Response:**
```json
{ "filename": "policy.pdf", "chunks_indexed": 8 }
```

### `POST /query`

Runs a question through the full agent graph.

**Request:**
```json
{ "query": "What is the refund window?", "llm_provider": "gemini" }
```
`llm_provider` is `"gemini"` or `"groq"`, default `"gemini"`.

**Response:**
```json
{
  "decision": "final_answer",
  "decision_reasoning": "Sufficiency 1.00, no contradictions, OCR confidence 1.00 — proceeding normally.",
  "answer": "...",
  "clarification_question": null,
  "low_confidence_caveat": null,
  "citations": ["1ce99e1f", "890ddac6"],
  "retry_count": 0
}
```

### `GET /health`

Returns `{"status": "ok"}` — basic liveness check.

---

## Frontend

Plain HTML/CSS/JS, no build step, no framework. Design concept: a "continuity trace" running down the chat column — each turn is a node on the trace, colored by which Decision Router path fired:

| Color | Outcome |
|---|---|
| Green | `final_answer` |
| Amber | `low_confidence` |
| Sky blue | `clarify` |
| Violet | `re_query` (shown as a "reformulated Nx" badge on the eventual answer) |

Hovering any answer's trace node shows a tooltip with `decision_reasoning` — the actual signal values that produced that decision, useful for demoing the mechanism live.

The provider switch in the sidebar controls `llm_provider` for every subsequent `/query` call — switching mid-conversation is fully supported, each turn is independent.

---

## Evaluation harness

```bash
cd backend
python -m eval.harness
```

Runs the 15 questions in `eval/test_questions.json` — spanning `answerable`, `insufficient_context`, `contradictory_context`, `ocr_degraded_source`, and `ambiguous_question` categories — through both a naive RAG baseline (retrieve + generate, no self-correction) and the full Groundwire agent. Reports:

- **Correct-refusal rate**: how often Groundwire chose `clarify` or `low_confidence` when it should have, instead of confidently answering
- **Baseline always-answers rate**: 100% by construction — the naive baseline has no refusal mechanism at all, which is the point of the comparison
- **Latency overhead**: average delta between the two modes

Results are written to `eval/results.md`.

> **Note:** the shipped `test_questions.json` contains placeholder questions. Replace them with questions that reference whatever documents you've actually ingested, or the results won't be meaningful. The `low_confidence` category specifically requires at least one genuine contradiction between two ingested documents to trigger correctly.

---

## Deployment

Deployed via Docker on [Render](https://groundwire-uiud.onrender.com/) (free tier).

- `Dockerfile` at the repo root installs Tesseract + Python dependencies, copies `backend/` and `frontend/`, and starts Uvicorn bound to Render's injected `$PORT`.
- Environment secrets (`GOOGLE_API_KEY`, `GROQ_API_KEY`) are set in Render's dashboard, not committed to the repo.
- `backend/data/chroma/` is committed to the repository (not gitignored) so the deployed instance starts with test documents already indexed, rather than an empty state.

To redeploy: push to `main` on GitHub — Render auto-builds and deploys on every push.

> **Free-tier note:** Render's free web services spin down after 15 minutes of inactivity and take 30–60 seconds to wake on the next request. This is expected — not a bug — if the first request after a period of idleness feels slow.

---

## Known limitations and honest scope notes

Written deliberately, so the gap between what's described and what's implemented is never a surprise in a live demo or interview.

- **Retrieval is vector-only, not true hybrid.** The architecture describes BM25 + vector hybrid search; the current implementation is vector similarity only, re-ranked by OCR confidence. BM25 was not added given hackathon time constraints.
- **No document-level ACL filtering.** Retrieval does not currently filter by per-document permissions — this is out of scope for the hackathon build.
- **No semantic response cache.** Repeated or near-duplicate queries are not cached; every query runs the full graph.
- **Native PDF extraction uses `pypdf`, not `unstructured`.** `unstructured` was originally used but pulled in PyTorch as a transitive dependency, which exceeded free-tier hosting memory limits. `pypdf` is lighter but does simpler linear text extraction rather than layout-aware parsing — complex multi-column PDFs may extract less cleanly than they would have with `unstructured`.
- **No API gateway, WAF, or dedicated auth service.** The FastAPI app has no authentication layer. Fine for a demo link; would need addressing before any real multi-user deployment.
- **Simplified monitoring.** No Prometheus/Grafana/ELK — the evaluation harness and application logs are the only observability currently in place.
- **The Decision Router's routing logic is deterministic Python, not LLM-driven.** This is a deliberate design choice (auditability, no added latency/cost on the hottest path), not a limitation — but worth being precise about if asked "does the LLM decide what to do."

---

## Roadmap

- Add BM25 keyword search alongside vector retrieval for genuine hybrid search
- Document-level ACL filtering at the retrieval layer
- Semantic response cache for repeated/similar queries
- Expand the evaluation question set with real ground-truth documents rather than placeholders
