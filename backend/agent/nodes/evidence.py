"""
Zone 5: Evidence Intelligence Service

Produces raw signals only — it never makes the routing decision itself.
The Decision Router (Zone 6) consumes these signals.

Two signals produced:
1. sufficiency_score  — does the retrieved context contain enough info to answer?
2. contradictions     — pairs of chunks asserting conflicting facts
"""

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from agent.llm_factory import get_raw_llm
from agent.state import AgentState, ContradictionEdge

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)



class EvidenceAssessment(BaseModel):
    sufficiency_score: float = Field(
        description="0.0-1.0. Does the context contain enough information to answer the query?"
    )
    contradictions: list[dict] = Field(
        default_factory=list,
        description=(
            "List of {claim_a_chunk_id, claim_b_chunk_id, reason} for any two chunks "
            "that assert mutually exclusive facts. Empty list if none found."
        ),
    )



EVIDENCE_PROMPT = """You are assessing retrieved evidence before it reaches an answer generator.

Query: {query}

Retrieved chunks:
{chunks_text}

Assess:
1. sufficiency_score: can this query be fully and accurately answered using only this
   evidence? 1.0 = fully sufficient, 0.0 = no relevant evidence at all.
2. contradictions: do any two chunks state conflicting facts (different dates, numbers,
   instructions, etc. for the same subject)? List each conflicting pair with chunk_ids
   and a one-sentence reason. If none, return an empty list.

Be conservative — only flag genuine factual contradictions, not differences in phrasing."""


def evidence_node(state: AgentState) -> dict:
    chunks = state["retrieved_chunks"]

    if not chunks:
        return {
            "sufficiency_score": 0.0,
            "contradictions": [],
            "aggregate_ocr_confidence": 0.0,
        }

    chunks_text = "\n\n".join(
        f"[{c['chunk_id']}] (source: {c['source_document']}, ocr_confidence: {c['ocr_confidence']:.2f})\n{c['text']}"
        for c in chunks
    )

    llm = get_raw_llm(state.get("llm_provider", "gemini"), temperature=0)
    structured_llm = llm.with_structured_output(EvidenceAssessment).with_retry(
        stop_after_attempt=3, wait_exponential_jitter=True
)
    assessment = structured_llm.invoke(
        EVIDENCE_PROMPT.format(query=state["query"], chunks_text=chunks_text)
)

    aggregate_ocr_confidence = sum(c["ocr_confidence"] for c in chunks) / len(chunks)

    contradictions: list[ContradictionEdge] = [
        ContradictionEdge(
            claim_a_chunk_id=c.get("claim_a_chunk_id", ""),
            claim_b_chunk_id=c.get("claim_b_chunk_id", ""),
            reason=c.get("reason", ""),
        )
        for c in assessment.contradictions
    ]

    return {
        "sufficiency_score": assessment.sufficiency_score,
        "contradictions": contradictions,
        "aggregate_ocr_confidence": aggregate_ocr_confidence,
    }
