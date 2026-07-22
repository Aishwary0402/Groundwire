"""
Zone 7: Prompt Engineering + LLM Reasoning Service

Two terminal paths land here: final_answer and low_confidence. Both fuse
evidence into a prompt, generate a draft, then run Chain-of-Verification
(CoVe) — a second pass that checks each claim in the draft against the
cited chunks before it's returned. This is what "verified_answer" means.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from agent.llm_factory import extract_text, get_llm
from agent.state import AgentState

# Stronger model for generation + verification — this is the path where quality matters most.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


GENERATION_PROMPT = """Answer the query using ONLY the evidence below. Write a clean,
natural answer — do not include chunk ids, bracketed references, or any citation markers
in the text itself.

Query: {query}

Evidence:
{chunks_text}

Answer:"""


COVE_PROMPT = """Verify this draft answer against its cited evidence. For each claim,
confirm it is directly supported by the cited chunk. Remove or soften any claim that
is not clearly supported.

Return ONLY the corrected answer text itself — no preamble, no explanation of what you
changed or why, no meta-commentary about the verification process. The user should see a
clean answer, not your editing notes.

Draft answer:
{draft}

Evidence:
{chunks_text}

Verified answer:"""

LOW_CONFIDENCE_PROMPT = """Answer the query using the evidence below, but the evidence has
a known issue: {caveat_reason}

Query: {query}

Evidence:
{chunks_text}

Write an answer that explicitly surfaces the conflict or uncertainty to the user — do not
just pick one version and state it confidently. If sources disagree, describe what each
source says in plain language, without inserting chunk ids or bracketed references into
the text itself."""
def _format_chunks(chunks) -> str:
    return "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)


def _generate_and_verify(query: str, chunks, provider: str) -> tuple[str, list[str]]:
    llm = get_llm(provider, temperature=0, quality=True)
    chunks_text = _format_chunks(chunks)

    draft = extract_text(llm.invoke(
    [HumanMessage(content=GENERATION_PROMPT.format(query=query, chunks_text=chunks_text))]
))

    verified = extract_text(llm.invoke(
    [HumanMessage(content=COVE_PROMPT.format(draft=draft, chunks_text=chunks_text))]
))
    citations = [c["chunk_id"] for c in chunks]
    return verified, citations


def final_answer_node(state: AgentState) -> dict:
    verified, citations = _generate_and_verify(
    state["query"], state["retrieved_chunks"], state.get("llm_provider", "gemini")
)
    return {
        "draft_answer": verified,
        "verified_answer": verified,
        "citations": citations,
        "low_confidence_caveat": None,
    }


# replace the low_confidence_node function with this:
def low_confidence_node(state: AgentState) -> dict:
    chunks = state["retrieved_chunks"]
    chunks_text = _format_chunks(chunks)
    provider = state.get("llm_provider", "gemini")
    llm = get_llm(provider, temperature=0, quality=True)

    draft = extract_text(llm.invoke(
        [HumanMessage(content=LOW_CONFIDENCE_PROMPT.format(
            query=state["query"],
            chunks_text=chunks_text,
            caveat_reason=state["decision_reasoning"],
        ))]
    ))

    verified = extract_text(llm.invoke(
        [HumanMessage(content=COVE_PROMPT.format(draft=draft, chunks_text=chunks_text))]
    ))

    citations = [c["chunk_id"] for c in chunks]
    return {
        "draft_answer": verified,
        "verified_answer": verified,
        "citations": citations,
        "low_confidence_caveat": state["decision_reasoning"],
    }
