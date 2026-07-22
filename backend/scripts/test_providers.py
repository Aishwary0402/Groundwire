"""
Standalone smoke test for the LLM provider factory — run this BEFORE testing
the full agent, so you know immediately whether the problem is provider
selection or something deeper in the graph.

Run: python -m scripts.test_providers
"""

import time
from dotenv import load_dotenv

load_dotenv()

from agent.llm_factory import get_llm, extract_text  # noqa: E402


def test_provider(provider: str, quality: bool = False):
    print(f"\n--- Testing provider={provider!r}, quality={quality} ---")
    llm = get_llm(provider, temperature=0, quality=quality)
    print(f"Instantiated class: {type(llm).__name__}")
    print(f"Model name: {getattr(llm, 'model', getattr(llm, 'model_name', 'unknown'))}")

    start = time.time()
    response = llm.invoke("Reply with exactly one word: which company made you?")
    elapsed = time.time() - start

    print(f"Response: {extract_text(response).strip()}")
    print(f"Latency: {elapsed:.2f}s")


if __name__ == "__main__":
    test_provider("gemini")
    test_provider("groq")
    test_provider("groq", quality=True)

    print("\n--- Checking isolation: same provider twice should return the SAME cached instance ---")
    a = get_llm("gemini", temperature=0)
    b = get_llm("gemini", temperature=0)
    print(f"Same object (cache working): {a is b}")