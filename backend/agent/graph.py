"""
Groundwire agent graph.

This is the literal implementation of the architecture diagram: retrieval ->
evidence intelligence -> decision router -> one of four branches. The
conditional edges out of "decision" are exactly the four paths from the
Decision Router — re_query loops back to retrieval (via refinement),
clarify/low_confidence/final_answer are terminal.
"""

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.retrieval import retrieval_node
from agent.nodes.evidence import evidence_node
from agent.nodes.decision import decision_node, route_from_decision
from agent.nodes.refinement import refinement_node
from agent.nodes.clarify import clarify_node
from agent.nodes.generation import final_answer_node, low_confidence_node


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieval", retrieval_node)
    graph.add_node("evidence", evidence_node)
    graph.add_node("decision", decision_node)
    graph.add_node("refinement", refinement_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("low_confidence", low_confidence_node)
    graph.add_node("final_answer", final_answer_node)

    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "evidence")
    graph.add_edge("evidence", "decision")

    # This is the Decision Router's four exit paths, made literal.
    graph.add_conditional_edges(
        "decision",
        route_from_decision,
        {
            "re_query": "refinement",
            "clarify": "clarify",
            "low_confidence": "low_confidence",
            "final_answer": "final_answer",
        },
    )

    # re_query loops back into retrieval — the retry cap lives in decision.py
    graph.add_edge("refinement", "retrieval")

    # terminal nodes
    graph.add_edge("clarify", END)
    graph.add_edge("low_confidence", END)
    graph.add_edge("final_answer", END)

    return graph.compile()


# Compiled once at import time, reused across requests
groundwire_agent = build_graph()
