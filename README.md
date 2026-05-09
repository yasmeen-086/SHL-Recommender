# SHL Assessment Recommender

A conversational AI agent that helps hiring managers find the right SHL assessments through natural dialogue. Built for the SHL Labs AI Intern take-home assignment.

**Live API:** https://shl-recommender-t1ol.onrender.com

---

## What it does

Instead of keyword-searching a 377-item catalog, hiring managers describe what they need in plain language. The agent clarifies when needed, recommends 1–10 assessments from the real SHL catalog, refines the shortlist mid-conversation, and compares assessments on request — all grounded in catalog data, never hallucinated.

---

## API

### `GET /health`
```json
{"status": "ok", "catalog_size": 377}
```

### `POST /chat`
**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I am hiring a mid-level Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Are you looking to assess technical skills, stakeholder communication, or both?"},
    {"role": "user", "content": "Both"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are assessments for a mid-level Java developer with stakeholder responsibilities.",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/", "test_type": "K"},
    {"name": "Occupational Personality Questionnaire OPQ32r", "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

The API is **stateless** — send the full conversation history on every call.

---

## Stack

| Component | Choice | Reason |
|---|---|---|
| LLM | llama-3.3-70b-versatile (Groq) | Free tier, ~1s latency, strong JSON output |
| Framework | FastAPI + Pydantic v2 | Fast, clean validation, auto docs |
| Catalog | 377 SHL assessments, pipe-delimited in system prompt | Fits Groq's 12k TPM limit; no retrieval latency |
| Deployment | Render (free tier) | Auto-deploy on git push, health check support |

---

## Run locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key (get free key at console.groq.com)
export GROQ_API_KEY=gsk_your_key_here

# 3. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4. Test it
curl http://localhost:8000/health
```

---

## Project structure

```
├── main.py          # FastAPI app + agent logic
├── catalog.json     # Full SHL catalog (377 assessments, raw scraped format)
├── requirements.txt
└── render.yaml      # Render deployment config
```

---

## How the agent works

1. **Clarify** — vague queries get one focused question before any recommendation
2. **Recommend** — once role + context is clear, returns 1–10 catalog-grounded assessments
3. **Refine** — "add personality test" / "drop the SQL test" updates the shortlist in-place
4. **Compare** — "what's the difference between X and Y" answered from catalog data only
5. **Refuse** — legal questions, salary questions, and off-topic requests are declined

Every URL in recommendations is validated against the catalog whitelist before returning.
