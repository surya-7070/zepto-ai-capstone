# Module 3 — Support Assistant (`/support_assistant`)

A grounded GenAI service over Zepto's own policy documents: local embeddings in
ChromaDB, a LangGraph intent router, a Pydantic-enforced answer schema, and a
FastAPI endpoint, all runnable **fully offline**.

## The MOCK_LLM contract

| `MOCK_LLM` | Mode | Status |
|---|---|---|
| unset (default) or `1` | Deterministic, rule-based mock. No signup, no API key, no network call to any LLM provider. | **This is the graded path.** |
| `0` | Calls a real LLM (Groq free tier by default) using the structured templates in `prompts.py`. | Optional, ungraded extension. |

Everything below was run, and should be graded, with `MOCK_LLM` left at its
default. The optional real-LLM extension was **not** exercised — the
`MOCK_LLM=0` branches and the schema-retry logic are implemented in code
(`llm.generate_validated`, the `else` branches in `graph.py`), but no LLM
account was used.

## How to run

```bash
pip install -r requirements.txt        # or the consolidated one at repo root
python ingest.py                       # build the ChromaDB index (one-off)
python graph.py                        # graph demo: one routed each way
uvicorn main:app --reload --port 7860  # serve POST /ask
python demo_calls.py                   # record the example calls below
```

First run downloads the `all-MiniLM-L6-v2` weights (~90 MB) from Hugging Face
and caches them locally. That is a model download, not an LLM API call — no
account or key is involved, and nothing afterwards touches the network.

### Docker (required, graded baseline)

```bash
cd support_assistant
docker build -t zepto-support .
docker run -p 7860:7860 zepto-support
curl -s -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"How long do I have to return a spoiled item?"}'
```

The image bakes in the embedding weights and runs `ingest.py` at build time, so
the container starts with a populated index and serves `POST /ask` immediately
on `0.0.0.0:7860`. `MOCK_LLM` is deliberately left unset in the image, so the
default container run is the graded mock path. The optional Hugging Face Spaces
deployment was **not** attempted; there is no live URL to report.

## Files

| File | Role |
|---|---|
| `docs/doc_01.txt … doc_08.txt` | The 8-document policy corpus, verbatim |
| `ingest.py` | Ingestion, chunking, embedding, ChromaDB storage, `retrieve()` |
| `prompts.py` | The role–context–task–format–length prompt templates |
| `schemas.py` | `AskRequest` and the enforced `AnswerResponse` |
| `llm.py` | `MOCK_LLM` toggle, optional Groq client, schema-retry logic |
| `graph.py` | The LangGraph `StateGraph`: 3 nodes + conditional edge |
| `main.py` | FastAPI app exposing `POST /ask` and `GET /health` |
| `demo_calls.py` | Records the example calls below from an actual run |
| `Dockerfile` | Locally buildable/runnable container |

## Architecture — the RAG pipeline, stage by stage

```
  docs/*.txt
      │  ingestion        ingest.py :: load_documents() + chunk_text()
      ▼
  chunks (1 per policy doc; sliding window if a doc exceeds 1200 chars)
      │  embedding        ingest.py :: get_embedding_function()
      │                   sentence-transformers all-MiniLM-L6-v2, local
      ▼
  ChromaDB collection "zepto_policies"  (persisted at ./chroma_store,
      │                                  hnsw:space = cosine)
      │
      │   ── query time ───────────────────────────────────────────────
      │
  POST /ask  ──> main.py :: ask()  ──> graph.py :: answer_query()
                                            │
                                   ┌────────▼─────────┐
                                   │ classify_intent  │  keyword heuristic
                                   └────────┬─────────┘  (mock, graded)
                                    conditional edge
                                    route_intent()
                        policy_question │        │ general_question
                                        ▼        ▼
                    ┌───────────────────────┐  ┌────────────────┐
                    │  retrieve_and_answer  │  │ direct_answer  │
                    │  retrieval: ALWAYS    │  │ canned string  │
                    │  real (top-3 cosine)  │  │ no retrieval   │
                    │  generation: branches │  │ generation:    │
                    │  on MOCK_LLM          │  │ branches on    │
                    └───────────┬───────────┘  └────────┬───────┘
                                └──────────┬────────────┘
                                           ▼
                            AnswerResponse (Pydantic-validated)
                            {answer, sources, confidence}
```

**Ingestion.** `ingest.py::load_documents()` reads the eight `.txt` files from
`docs/` and `chunk_text()` splits each one. Because every policy is a single
short paragraph, each document becomes exactly one chunk (id `doc_0N::chunk_0`),
which keeps a policy semantically whole; a document longer than 1200 characters
would fall back to a fixed-size sliding window with 150 characters of overlap.

**Embedding.** `ingest.py::get_embedding_function()` wraps
`sentence-transformers` `all-MiniLM-L6-v2`, running entirely on the local
machine. `build_index()` writes one 384-dimensional vector per chunk into the
ChromaDB collection **`zepto_policies`**, persisted at `./chroma_store` with
`metadata={"hnsw:space": "cosine"}` so similarity is cosine. Each vector carries
its `source` document id and `chunk_index` as metadata.

**Retrieval.** `ingest.py::retrieve(query, k=3)` embeds the incoming query with
the same model and returns the top-3 nearest chunks with their ids, source
documents and similarity scores. This is called by the LangGraph node
**`retrieve_and_answer`**, and it runs for real in *both* modes — embedding and
vector search need no API key and no LLM provider.

**Generation.** The `classify_intent` node decides the route; the conditional
edge `route_intent()` sends `policy_question` to `retrieve_and_answer` and
`general_question` to `direct_answer`. In the default mock mode,
`retrieve_and_answer` produces
`f"Based on the retrieved context: {top_chunk[:200]}"` from the single most
similar chunk, and `direct_answer` returns the fixed string
*"I can only answer questions about Zepto policies right now."*. Either way the
node populates `answer`, `sources` and `confidence` in the graph state, and
`answer_query()` validates the result against `schemas.AnswerResponse` before
`main.py` returns it.

**What `MOCK_LLM` changes.** Exactly one stage branches on it: **generation**
(and the classification decision inside `classify_intent`). Ingestion,
embedding and retrieval are identical in both states.

- **Default (mock).** `classify_intent` uses the keyword heuristic over
  `"delivery", "return", "refund", "membership", "tracking", "cancel",
  "gift card", "support hours"`. `retrieve_and_answer` returns the canned
  template built from the top chunk. `direct_answer` returns the fixed string.
  `sources` = the ids of the retrieved chunks (empty for `general_question`),
  `confidence` = 1.0. Zero LLM calls.
- **`MOCK_LLM=0` (optional).** `classify_intent` asks the LLM to label the
  query using `CLASSIFY_PROMPT_TEMPLATE`; `retrieve_and_answer` sends
  `ANSWER_PROMPT_TEMPLATE` with the retrieved chunks as context and requires
  grounded JSON back; `direct_answer` prompts the LLM with no retrieval. Raw
  output is validated against `AnswerResponse` and, on failure, retried up to
  **2 additional times** with `CORRECTIVE_INSTRUCTION` appended before a
  clearly marked `[ERROR] …` response is returned.

## Prompt template (Task 2)

The full text lives in `prompts.py`. It carries all five skeleton components —
`### ROLE`, `### CONTEXT`, `### TASK`, `### FORMAT`, `### LENGTH` — plus a
`### NEGATIVE CONSTRAINT` section ("Do not answer using information that is not
present in the provided context…") and a `### EXAMPLE` section containing one
worked few-shot example (a phone-support question answered from `doc_08` with
its correct JSON output).

## Example `/ask` calls

> Regenerate this block from an actual run with `python demo_calls.py` — it
> rewrites everything between the markers below from live responses.

<!-- BEGIN EXAMPLE CALLS -->

Recorded by `demo_calls.py` with `MOCK_LLM` left at its default (mode: **mock**), so no LLM call is made in any of these.

**Query that SHOULD trigger retrieval (routed to retrieve_and_answer)**

Routed to: `retrieve_and_answer` (intent = `policy_question`)

```bash
curl -s -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How long do I have to report a damaged item for a refund?"}'
```

Response (HTTP 200):

```json
{
  "answer": "Based on the retrieved context: Damaged or Missing Items: \"If an order arrives with damaged, spoiled, or missing items, customers must report it within 24 hours of delivery through the 'Report an Issue' button on the order page. Zep",
  "sources": [
    "doc_06::chunk_0",
    "doc_02::chunk_0",
    "doc_08::chunk_0"
  ],
  "confidence": 1.0
}
```

**Query that should NOT trigger retrieval (routed to direct_answer)**

Routed to: `direct_answer` (intent = `general_question`)

```bash
curl -s -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Who won the football match last night?"}'
```

Response (HTTP 200):

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

**Second policy example - membership keyword**

Routed to: `retrieve_and_answer` (intent = `policy_question`)

```bash
curl -s -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What does a Zepto Pass+ membership cost per month?"}'
```

Response (HTTP 200):

```json
{
  "answer": "Based on the retrieved context: Membership Tiers: \"Zepto offers three account tiers: Basic (free, default tier, standard delivery fees apply), Zepto Pass (INR 49 per month, free standard delivery on all orders and 5% off select cate",
  "sources": [
    "doc_03::chunk_0",
    "doc_01::chunk_0",
    "doc_07::chunk_0"
  ],
  "confidence": 1.0
}
```

<!-- END EXAMPLE CALLS -->

## Design decisions

- **Per-document chunking.** Each policy is one dense paragraph; splitting it
  would separate a rule from its exception (e.g. the INR 1000 photo
  requirement in `doc_06`), which is exactly the kind of fact a grounded
  assistant must not retrieve half of.
- **Cosine over L2.** Sentence-transformer embeddings are direction-bearing;
  cosine is the similarity the model was trained for, so it is set explicitly
  rather than left to Chroma's default.
- **Retrieval never mocked.** Mocking it would hide the one part of the
  pipeline most likely to be wrong. Keeping it real means the graded path
  still proves that a returns question retrieves the returns document.
- **Schema enforced in both directions.** In mock mode the schema is populated
  deterministically from code, so validation can't fail; the retry logic exists
  for the real-LLM path, where a model can and does emit prose around its JSON.
