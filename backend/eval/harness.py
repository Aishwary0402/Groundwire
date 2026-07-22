"""
Evaluation harness: self-correction ON vs OFF.

"OFF" mode = naive RAG baseline: retrieve top-k, generate directly, no
evidence check, no decision router, no OCR weighting. This is the thing
your architecture is designed to beat.

"ON" mode = the full Groundwire agent graph.

Run: python -m eval.harness
Outputs a markdown comparison table to eval/results.md.
"""

import time
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent.graph import groundwire_agent  # noqa: E402
from agent.nodes.retrieval import retrieval_node  # noqa: E402
from agent.nodes.generation import _generate_and_verify  # noqa: E402

TEST_QUESTIONS_PATH = Path(__file__).parent / "test_questions.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def run_baseline(query: str) -> dict:
    """Naive RAG: retrieve + generate, no self-correction, no OCR weighting."""
    start = time.time()
    retrieval_result = retrieval_node({"query": query})
    chunks = retrieval_result["retrieved_chunks"][:4]
    answer, citations = _generate_and_verify(query, chunks) if chunks else ("(no evidence found)", [])
    latency = time.time() - start
    return {"answer": answer, "citations": citations, "latency": latency, "always_answers": True}


def run_groundwire(query: str) -> dict:
    start = time.time()
    result = groundwire_agent.invoke({"query": query, "original_query": query, "retry_count": 0})
    latency = time.time() - start
    return {
        "decision": result.get("decision"),
        "answer": result.get("verified_answer"),
        "clarification_question": result.get("clarification_question"),
        "latency": latency,
        "retry_count": result.get("retry_count", 0),
    }


def score_correct_refusal(question: dict, groundwire_result: dict) -> bool:
    """Did the agent choose the expected non-answer behavior when it should have?"""
    expected = question["expected_behavior"]
    actual = groundwire_result.get("decision")
    if expected in ("clarify", "low_confidence"):
        return actual == expected
    return actual == "final_answer"


def main():
    questions = json.loads(TEST_QUESTIONS_PATH.read_text())

    rows = []
    correct_refusals = 0
    refusal_questions = 0
    baseline_latencies, groundwire_latencies = [], []

    for q in questions:
        baseline = run_baseline(q["query"])
        groundwire = run_groundwire(q["query"])

        baseline_latencies.append(baseline["latency"])
        groundwire_latencies.append(groundwire["latency"])

        if q["expected_behavior"] != "final_answer":
            refusal_questions += 1
            if score_correct_refusal(q, groundwire):
                correct_refusals += 1

        rows.append(
            {
                "id": q["id"],
                "category": q["category"],
                "expected": q["expected_behavior"],
                "groundwire_decision": groundwire.get("decision"),
                "baseline_always_answered": True,
            }
        )

    correct_refusal_rate = (correct_refusals / refusal_questions * 100) if refusal_questions else 0.0
    avg_baseline_latency = sum(baseline_latencies) / len(baseline_latencies)
    avg_groundwire_latency = sum(groundwire_latencies) / len(groundwire_latencies)

    report_lines = [
        "# Groundwire evaluation results",
        "",
        f"- Correct-refusal rate (Groundwire): **{correct_refusal_rate:.1f}%** "
        f"({correct_refusals}/{refusal_questions} questions that should NOT get a confident answer)",
        f"- Baseline always answers: **100%** (naive RAG has no refusal mechanism)",
        f"- Avg latency — baseline: **{avg_baseline_latency:.2f}s**, Groundwire: **{avg_groundwire_latency:.2f}s**",
        f"- Latency overhead: **{avg_groundwire_latency - avg_baseline_latency:.2f}s**",
        "",
        "| id | category | expected | groundwire decision |",
        "|---|---|---|---|",
    ]
    for r in rows:
        report_lines.append(f"| {r['id']} | {r['category']} | {r['expected']} | {r['groundwire_decision']} |")

    RESULTS_PATH.write_text("\n".join(report_lines))
    print(f"Results written to {RESULTS_PATH}")
    print(f"Correct-refusal rate: {correct_refusal_rate:.1f}%")


if __name__ == "__main__":
    main()
