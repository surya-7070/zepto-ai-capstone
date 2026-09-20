"""
Module 3, Tasks 3-5 — the LangGraph StateGraph.

    classify_intent ──(conditional edge)──> retrieve_and_answer ──> END
                    └─────────────────────> direct_answer       ──> END

The routing logic itself does NOT depend on MOCK_LLM — only the generation
step inside each node does. Retrieval always runs for real.

Run directly for a self-contained demo:  python graph.py
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from ingest import retrieve
from llm import call_llm, generate_validated, is_mock
from prompts import build_answer_prompt, build_classify_prompt
from schemas import AnswerResponse

TOP_K = 3
SNIPPET_CHARS = 200

# Task 3's required keyword heuristic for mock-mode classification.
POLICY_KEYWORDS = (
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
)

CANNED_GENERAL_ANSWER = "I can only answer questions about Zepto policies right now."


class GraphState(TypedDict, total=False):
    """State passed between nodes."""

    query: str
    intent: Literal["policy_question", "general_question"]
    chunks: list[dict]
    answer: str
    sources: list[str]
    confidence: float


# ==========================================================================
# Node 1 — classify_intent
# ==========================================================================
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]

    if is_mock():
        # GRADED BASELINE: keyword heuristic, no LLM call.
        lowered = query.lower()
        intent = (
            "policy_question"
            if any(kw in lowered for kw in POLICY_KEYWORDS)
            else "general_question"
        )
    else:
        # Optional MOCK_LLM=0 extension: ask the LLM to classify.
        raw = call_llm(build_classify_prompt(query)).strip().lower()
        intent = "policy_question" if "policy_question" in raw else "general_question"

    return {"intent": intent}


def route_intent(state: GraphState) -> str:
    """Conditional edge: pure routing, independent of MOCK_LLM."""
    return (
        "retrieve_and_answer"
        if state.get("intent") == "policy_question"
        else "direct_answer"
    )


# ==========================================================================
# Node 2 — retrieve_and_answer
# ==========================================================================
def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    # Retrieval ALWAYS runs for real, in both modes: embedding the query with
    # all-MiniLM-L6-v2 and searching ChromaDB by cosine similarity needs no
    # API key and no network call to an LLM provider.
    chunks = retrieve(query, k=TOP_K)
    source_ids = [c["id"] for c in chunks]

    if is_mock():
        # GRADED BASELINE: canned templated answer from the top chunk, and a
        # deterministically populated schema. There is no LLM output that
        # could fail validation, because none was generated.
        top = chunks[0]["text"] if chunks else ""
        snippet = top[:SNIPPET_CHARS]
        return {
            "chunks": chunks,
            "answer": f"Based on the retrieved context: {snippet}",
            "sources": source_ids,
            "confidence": 1.0,
        }

    # Optional MOCK_LLM=0 extension: prompt the real LLM with the structured
    # template, grounded only in the retrieved chunks, with retry-on-invalid.
    validated: AnswerResponse = generate_validated(
        build_answer_prompt(query, chunks)
    )
    return {
        "chunks": chunks,
        "answer": validated.answer,
        # fall back to what retrieval actually returned if the model named none
        "sources": validated.sources or source_ids,
        "confidence": validated.confidence,
    }


# ==========================================================================
# Node 3 — direct_answer
# ==========================================================================
def direct_answer(state: GraphState) -> GraphState:
    if is_mock():
        # GRADED BASELINE: fixed canned string, no LLM call, no retrieval.
        return {
            "answer": CANNED_GENERAL_ANSWER,
            "sources": [],
            "confidence": 1.0,
        }

    # Optional MOCK_LLM=0 extension: prompt the LLM directly, no retrieval.
    raw = call_llm(
        "### ROLE\nYou are Zepto's support assistant.\n"
        "### TASK\nAnswer the customer briefly.\n"
        "### NEGATIVE CONSTRAINT\nDo not invent Zepto policy details.\n"
        "### LENGTH\nUnder 60 words.\n\n"
        f"Question: {state['query']}\n"
    )
    return {"answer": raw.strip(), "sources": [], "confidence": 0.5}


# ==========================================================================
# Graph assembly
# ==========================================================================
def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


_APP = None


def get_app():
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP


def answer_query(query: str) -> AnswerResponse:
    """Run the graph end to end and return the Pydantic-validated answer."""
    final = get_app().invoke({"query": query})
    return AnswerResponse(
        answer=final.get("answer", ""),
        sources=final.get("sources", []),
        confidence=float(final.get("confidence", 0.0)),
    )


if __name__ == "__main__":
    print(f"[graph] mode = {'mock' if is_mock() else 'real-llm'}")
    demos = [
        "How long do I have to return a damaged item?",   # -> retrieve_and_answer
        "What is the capital of France?",                 # -> direct_answer
    ]
    for q in demos:
        state = get_app().invoke({"query": q})
        print(f"\nQ: {q}")
        print(f"   intent : {state['intent']}")
        print(f"   answer : {state['answer']}")
        print(f"   sources: {state.get('sources')}")
        print(f"   conf   : {state.get('confidence')}")
