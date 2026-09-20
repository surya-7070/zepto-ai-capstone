"""
Module 3, Task 6 — the FastAPI wrapper.

    POST /ask   {"query": "..."}  ->  {"answer": ..., "sources": [...],
                                       "confidence": 0.0-1.0}

Run locally (graded default = mock mode, so do not set MOCK_LLM):

    uvicorn main:app --reload --port 7860
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from graph import answer_query, get_app
from ingest import build_index
from llm import mode_name
from schemas import AnswerResponse, AskRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the Chroma index and compile the graph once at startup so the
    # first request does not pay for it.
    build_index()
    get_app()
    print(f"[startup] support assistant ready (LLM mode: {mode_name()})")
    yield


app = FastAPI(
    title="Zepto Support Assistant",
    description=(
        "Grounded RAG over Zepto's policy corpus: ChromaDB retrieval + a "
        "LangGraph intent router, with a deterministic offline mock as the "
        "default LLM path."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_mode": mode_name()}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest) -> AnswerResponse:
    try:
        return answer_query(request.query)
    except Exception as exc:  # pragma: no cover - surfaced as a clean 500
        raise HTTPException(status_code=500, detail=str(exc)) from exc
