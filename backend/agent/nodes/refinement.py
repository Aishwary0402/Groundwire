"""
Zone 6 (support): Query Refinement Service

Reformulates the query when the Decision Router chooses 're_query'.
Increments retry_count so the router can enforce the retry cap.
"""

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from agent.llm_factory import extract_text, get_llm

from agent.state import AgentState

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)


REFINEMENT_PROMPT = """The original query below did not retrieve sufficient evidence.
Rewrite it to be more specific and more likely to match relevant document content —
add likely synonyms, expand abbreviations, or narrow scope based on what's implied.
Return ONLY the rewritten query, nothing else.

Original query: {query}
Previous attempt count: {retry_count}"""


def refinement_node(state: AgentState) -> dict:
    llm = get_llm(state.get("llm_provider", "gemini"), temperature=0.3)
    response = llm.invoke(
        [
            HumanMessage(
                content=REFINEMENT_PROMPT.format(
                    query=state.get("original_query", state["query"]),
                    retry_count=state.get("retry_count", 0),
                )
            )
        ]
    )

    return {
        "query": extract_text(response).strip(),
        "retry_count": state.get("retry_count", 0) + 1,
    }
