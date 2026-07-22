"""
Zone 4: Hybrid Retrieval Service

Retrieves candidate chunks from ChromaDB and re-ranks them using the
OCR-Aware Scorer: weighted_score = similarity_score * ocr_confidence.

This is the core differentiator vs standard RAG — a chunk from a badly
scanned page should never outrank a clean chunk purely because it's
semantically closer.
"""

from agent.state import AgentState, RetrievedChunk
from ingestion.vectorstore import get_vectorstore

TOP_K = 8


def retrieval_node(state: AgentState) -> dict:
    query = state["query"]
    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search_with_relevance_scores(query, k=TOP_K)

    chunks: list[RetrievedChunk] = []
    for doc, similarity_score in results:
        ocr_confidence = float(doc.metadata.get("ocr_confidence", 1.0))
        chunks.append(
            RetrievedChunk(
                chunk_id=doc.metadata.get("chunk_id", ""),
                text=doc.page_content,
                source_document=doc.metadata.get("source_document", "unknown"),
                similarity_score=similarity_score,
                ocr_confidence=ocr_confidence,
                weighted_score=similarity_score * ocr_confidence,
            )
        )

    # Re-rank by OCR-weighted score, not raw similarity
    chunks.sort(key=lambda c: c["weighted_score"], reverse=True)

    return {"retrieved_chunks": chunks}
