"""
Shared state for the Groundwire agent graph.

Every node in the LangGraph reads from and writes to this single state object.
This is what makes it an agent rather than a chain: the state accumulates
evidence and signals across steps, and the decision router reads that
accumulated state to choose its own next action.
"""

from typing import TypedDict, Literal, Optional


class RetrievedChunk(TypedDict):
    chunk_id: str
    text: str
    source_document: str
    similarity_score: float
    ocr_confidence: float          # 0.0-1.0, set at ingestion time
    weighted_score: float          # similarity_score * ocr_confidence


class ContradictionEdge(TypedDict):
    claim_a_chunk_id: str
    claim_b_chunk_id: str
    reason: str


class AgentState(TypedDict, total=False):
    # --- Input ---
    query: str
    original_query: str            # preserved across re-query rewrites
    retry_count: int                # incremented on each re-query loop
    llm_provider: Literal["gemini", "groq"]   # <-- add this line

    # --- Retrieval (Zone 4) ---
    retrieved_chunks: list[RetrievedChunk]

    # --- Evidence Intelligence (Zone 5) ---
    sufficiency_score: float        # 0-1, does context answer the query?
    contradictions: list[ContradictionEdge]
    aggregate_ocr_confidence: float

    # --- Decision Router (Zone 6) ---
    # This is the core innovation: the agent's own choice of next action.
    decision: Literal["re_query", "clarify", "low_confidence", "final_answer"]
    decision_reasoning: str          # why the router chose this path

    # --- Generation (Zone 7) ---
    draft_answer: Optional[str]
    verified_answer: Optional[str]   # after Chain-of-Verification
    citations: list[str]

    # --- Output flags for the UI ---
    clarification_question: Optional[str]
    low_confidence_caveat: Optional[str]
