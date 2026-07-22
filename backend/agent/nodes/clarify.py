"""
Zone 6 (support): Clarification path.

Generates a specific clarifying question referencing what's actually
ambiguous or missing — never a generic "can you clarify?".
"""

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from agent.llm_factory import get_llm

from agent.state import AgentState

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)


CLARIFY_PROMPT = """The available evidence is insufficient to answer this query, even
after retry attempts. Write ONE specific clarifying question to ask the user — reference
what's actually missing or ambiguous (e.g. "which year's policy", "which product version")
rather than a generic "can you clarify?". Keep it to one sentence.

Query: {query}
What evidence was found (may be sparse or irrelevant):
{chunks_text}"""


def clarify_node(state: AgentState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    chunks_text = "\n".join(f"- {c['text'][:150]}" for c in chunks[:3]) or "(no relevant evidence found)"
    llm = get_llm(state.get("llm_provider", "gemini"), temperature=0.3)
    question = llm.invoke(
        [HumanMessage(content=CLARIFY_PROMPT.format(query=state["query"], chunks_text=chunks_text))]
    ).content.strip()

    return {"clarification_question": question}
