"""
LLM provider factory.

A single user-selected provider (state["llm_provider"]) controls every chat
call in the agent — evidence assessment, refinement, clarify, and generation.
Selecting "groq" means every one of those calls goes to Groq; selecting
"gemini" means all of them go to Gemini. No mixing.

Exception: embeddings always use Gemini regardless of this setting, because
Groq has no embeddings endpoint. This only affects document ingestion, not
the per-query chat path, so it doesn't break the "everything on one provider"
experience the user sees.
"""

from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

GEMINI_MODEL = "gemini-flash-lite-latest"
GROQ_FAST_MODEL = "llama-3.1-8b-instant"
GROQ_QUALITY_MODEL = "llama-3.3-70b-versatile"

def get_raw_llm(provider: str, temperature: float = 0.0, quality: bool = False):
    """Returns the LLM client WITHOUT retry wrapping — use this when you need
    to call .with_structured_output() first, then wrap the result yourself."""
    provider = (provider or "gemini").lower()

    if provider == "groq":
        model = GROQ_QUALITY_MODEL if quality else GROQ_FAST_MODEL
        return ChatGroq(model=model, temperature=temperature)

    if provider == "gemini":
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature)

    raise ValueError(f"Unknown llm_provider: {provider!r} (expected 'gemini' or 'groq')")


@lru_cache(maxsize=8)
def get_llm(provider: str, temperature: float = 0.0, quality: bool = False):
    provider = (provider or "gemini").lower()

    if provider == "groq":
        model = GROQ_QUALITY_MODEL if quality else GROQ_FAST_MODEL
        llm = ChatGroq(model=model, temperature=temperature)
        return llm.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)

    if provider == "gemini":
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature)
        return llm.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)

    raise ValueError(f"Unknown llm_provider: {provider!r} (expected 'gemini' or 'groq')")

def extract_text(response) -> str:
    """
    Some model/provider combos (notably newer Gemini versions) return
    response.content as a list of content blocks instead of a plain string.
    Every node that reads LLM output should go through this instead of
    calling .content directly, so we don't hit this again in a different file.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)