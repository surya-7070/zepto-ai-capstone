"""
Module 3 — the MOCK_LLM toggle and the optional real-LLM client.

Contract
--------
MOCK_LLM unset, or MOCK_LLM=1  -> mock mode. THE GRADED DEFAULT.
                                  No signup, no API key, no network call to any
                                  LLM provider. Every generation step is
                                  deterministic, rule-based code.
MOCK_LLM=0                     -> optional extension. Calls a real LLM
                                  (Groq's free tier by default) using the
                                  structured templates in prompts.py.

Only generation branches on this toggle. Embedding and ChromaDB retrieval
always run for real, in both modes.
"""

from __future__ import annotations

import json
import os
import re

from pydantic import ValidationError

from prompts import CORRECTIVE_INSTRUCTION
from schemas import AnswerResponse

MAX_RETRIES = 2  # 2 *additional* attempts after the first failure


def is_mock() -> bool:
    """Mock unless MOCK_LLM is explicitly set to 0."""
    return os.getenv("MOCK_LLM", "1").strip() != "0"


def mode_name() -> str:
    return "mock" if is_mock() else "real-llm"


# --------------------------------------------------------------------------
# Optional real-LLM path (only reachable when MOCK_LLM=0)
# --------------------------------------------------------------------------
def call_llm(prompt: str) -> str:
    """
    Send one prompt to the configured LLM and return its raw text.

    Never called in mock mode. Defaults to Groq's free tier; the base URL and
    model are overridable so any OpenAI-compatible free-tier provider can be
    substituted.
    """
    if is_mock():
        raise RuntimeError(
            "call_llm() reached while MOCK_LLM is at its default. This is a "
            "bug: mock mode must never make an LLM call."
        )

    from groq import Groq  # imported lazily so mock mode needs no dependency

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MOCK_LLM=0 was set but GROQ_API_KEY is not present in the "
            "environment. Unset MOCK_LLM to use the graded mock path."
        )

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    return completion.choices[0].message.content or ""


def _extract_json(raw: str) -> dict:
    """Pull the first JSON object out of a raw completion."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in the model output")
    return json.loads(match.group(0))


def generate_validated(prompt: str) -> AnswerResponse:
    """
    Task 5's retry-on-failure logic for the optional real-LLM path.

    Calls the LLM, validates its raw output against AnswerResponse, and on
    failure retries up to MAX_RETRIES additional times with a corrective
    instruction appended. If every attempt fails, returns a clearly marked
    error response rather than raising into the API layer.
    """
    attempt_prompt = prompt
    last_error = ""

    for attempt in range(MAX_RETRIES + 1):  # 1 initial + 2 corrective retries
        try:
            raw = call_llm(attempt_prompt)
            return AnswerResponse.model_validate(_extract_json(raw))
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            attempt_prompt = prompt + CORRECTIVE_INSTRUCTION.format(
                error=last_error
            )
        except Exception as exc:  # transport/provider failure - do not retry
            return AnswerResponse(
                answer=f"[ERROR] The language model could not be reached: {exc}",
                sources=[],
                confidence=0.0,
            )

    return AnswerResponse(
        answer=(
            "[ERROR] The language model returned output that failed schema "
            f"validation after {MAX_RETRIES + 1} attempts. Last error: "
            f"{last_error}"
        ),
        sources=[],
        confidence=0.0,
    )
