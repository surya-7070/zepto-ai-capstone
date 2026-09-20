"""
Module 3, Task 6 — record the example /ask calls.

Runs the real FastAPI app in-process with fastapi.testclient (same routing,
same Pydantic validation, same graph) and writes the raw JSON responses into
example_calls.md AND into the transcript block of README.md, between the
<!-- BEGIN EXAMPLE CALLS --> / <!-- END EXAMPLE CALLS --> markers.

Run with MOCK_LLM left at its default - that is the graded path:

    python demo_calls.py
"""

from __future__ import annotations

import json
import os
import re

from fastapi.testclient import TestClient

from graph import get_app
from llm import mode_name
from main import app

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "example_calls.md")
README = os.path.join(HERE, "README.md")

BEGIN = "<!-- BEGIN EXAMPLE CALLS -->"
END = "<!-- END EXAMPLE CALLS -->"

EXAMPLES = [
    ("Query that SHOULD trigger retrieval (routed to retrieve_and_answer)",
     "How long do I have to report a damaged item for a refund?"),
    ("Query that should NOT trigger retrieval (routed to direct_answer)",
     "Who won the football match last night?"),
    ("Second policy example - membership keyword",
     "What does a Zepto Pass+ membership cost per month?"),
]


def main() -> int:
    lines = [
        f"Recorded by `demo_calls.py` with `MOCK_LLM` left at its default "
        f"(mode: **{mode_name()}**), so no LLM call is made in any of these.",
        "",
    ]

    graph_app = get_app()
    with TestClient(app) as client:
        for label, query in EXAMPLES:
            state = graph_app.invoke({"query": query})   # to expose the route
            response = client.post("/ask", json={"query": query})
            payload = json.dumps(response.json(), indent=2, ensure_ascii=False)

            print(f"\n=== {label} ===")
            print(f"query : {query}")
            print(f"intent: {state['intent']} -> "
                  f"{'retrieve_and_answer' if state['intent'] == 'policy_question' else 'direct_answer'}")
            print(payload)

            lines += [
                f"**{label}**",
                "",
                f"Routed to: `{'retrieve_and_answer' if state['intent'] == 'policy_question' else 'direct_answer'}` "
                f"(intent = `{state['intent']}`)",
                "",
                "```bash",
                "curl -s -X POST http://localhost:7860/ask \\",
                '  -H "Content-Type: application/json" \\',
                f"  -d '{json.dumps({'query': query})}'",
                "```",
                "",
                f"Response (HTTP {response.status_code}):",
                "",
                "```json",
                payload,
                "```",
                "",
            ]

    body = "\n".join(lines)

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("# Example `/ask` calls\n\n" + body)
    print(f"\n[done] wrote {OUT_MD}")

    if os.path.exists(README):
        readme = open(README, encoding="utf-8").read()
        if BEGIN in readme and END in readme:
            readme = re.sub(
                re.escape(BEGIN) + r".*?" + re.escape(END),
                f"{BEGIN}\n\n{body}\n{END}",
                readme,
                flags=re.DOTALL,
            )
            open(README, "w", encoding="utf-8").write(readme)
            print(f"[done] refreshed the transcript block in {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
