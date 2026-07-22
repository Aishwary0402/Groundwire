"""
Zone 6: Self-Correction & Decision Router — the core innovation.

Deliberately rule-based, not an LLM judgment call. This is a design decision
worth defending in an interview: an explicit, tunable threshold table is
auditable and debuggable in a way a black-box "confidence score" isn't, and
it's cheap (no extra LLM call) on the hottest path in the graph.

Reads four signals and returns exactly one of four outcomes:
  - re_query        reformulate and retry retrieval (capped at MAX_RETRIES)
  - clarify         ask the user a direct question
  - low_confidence  answer, but caveated with citations
  - final_answer    normal high-confidence path
"""

from agent.state import AgentState

# Tunable thresholds — change these and re-run the eval harness to see the effect.
SUFFICIENCY_THRESHOLD = 0.6
OCR_CONFIDENCE_THRESHOLD = 0.5
MAX_RETRIES = 2


def decision_node(state: AgentState) -> dict:
    sufficiency = state.get("sufficiency_score", 0.0)
    contradictions = state.get("contradictions", [])
    aggregate_ocr_confidence = state.get("aggregate_ocr_confidence", 1.0)
    retry_count = state.get("retry_count", 0)

    has_contradictions = len(contradictions) > 0
    low_ocr_confidence = aggregate_ocr_confidence < OCR_CONFIDENCE_THRESHOLD
    insufficient = sufficiency < SUFFICIENCY_THRESHOLD

    # Rule 1: nothing relevant at all, and we still have retries left -> re-query
    if sufficiency < 0.2 and retry_count < MAX_RETRIES:
        return {
            "decision": "re_query",
            "decision_reasoning": (
                f"Sufficiency {sufficiency:.2f} is near-zero and retries remain "
                f"({retry_count}/{MAX_RETRIES}) — reformulating the query."
            ),
        }

    # Rule 2: insufficient evidence, retries exhausted -> ask the user directly
    if insufficient and retry_count >= MAX_RETRIES:
        return {
            "decision": "clarify",
            "decision_reasoning": (
                f"Sufficiency {sufficiency:.2f} still below {SUFFICIENCY_THRESHOLD} "
                f"after {retry_count} retries — asking the user to narrow the query."
            ),
        }

    # Rule 3: partial evidence but retries remain -> one more re-query attempt
    if insufficient and retry_count < MAX_RETRIES:
        return {
            "decision": "re_query",
            "decision_reasoning": (
                f"Sufficiency {sufficiency:.2f} below {SUFFICIENCY_THRESHOLD}, "
                f"retry {retry_count + 1}/{MAX_RETRIES}."
            ),
        }

    # Rule 4: sufficient evidence, but contradictory or OCR-degraded -> hedge
    if has_contradictions or low_ocr_confidence:
        reason = []
        if has_contradictions:
            reason.append(f"{len(contradictions)} contradiction(s) detected")
        if low_ocr_confidence:
            reason.append(f"aggregate OCR confidence {aggregate_ocr_confidence:.2f} is low")
        return {
            "decision": "low_confidence",
            "decision_reasoning": "Evidence is sufficient but " + " and ".join(reason) + ".",
        }

    # Rule 5: sufficient, consistent, clean evidence -> answer normally
    return {
        "decision": "final_answer",
        "decision_reasoning": (
            f"Sufficiency {sufficiency:.2f}, no contradictions, "
            f"OCR confidence {aggregate_ocr_confidence:.2f} — proceeding normally."
        ),
    }


def route_from_decision(state: AgentState) -> str:
    """Conditional edge function — maps the decision to the next graph node name."""
    return state["decision"]
