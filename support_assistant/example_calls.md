# Example `/ask` calls

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
