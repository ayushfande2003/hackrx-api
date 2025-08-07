# HackRx Document Query API

LLM-powered API that answers questions from a provided PDF document using embeddings (FAISS) and GPT-4.

## Endpoints
- `GET /` – Welcome/info
- `GET /healthz` – Health check (200 OK)
- `POST /hackrx/run` – Run Q&A (requires `Authorization: Bearer test_token`)

## Quick demo (under 2 minutes)

1) Open Swagger UI
   - `https://hackrx-api-5l67.onrender.com/docs`
   - Click Authorize → enter `test_token` → Authorize → Close

2) Try the API
   - Expand `POST /hackrx/run` → Try it out
   - Use this body (or your own public PDF URL):
     ```json
     {
       "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?...",
       "questions": [
         "What is the grace period for premium payment?",
         "Is maternity covered?",
         "What is the waiting period for cataract?"
       ]
     }
     ```
   - Execute → Show the answers

3) cURL
   ```bash
   curl -X POST "https://hackrx-api-5l67.onrender.com/hackrx/run" \
     -H "Authorization: Bearer test_token" \
     -H "Content-Type: application/json" \
     -d '{"documents":"https://hackrx.blob.core.windows.net/assets/policy.pdf?...","questions":["What is the grace period for premium payment?","Is maternity covered?","What is the waiting period for cataract?"]}'
   ```

## Render deployment notes
- Service URL root (`/`) now shows a welcome JSON message with instructions
- Health check endpoint: `/healthz`
- Ensure env var `OPENAI_API_KEY` is set
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## How it works (talk track)
- Extracts text from the PDF (PyMuPDF)
- Splits into chunks; embeds with OpenAI
- FAISS index for semantic retrieval
- Retrieves top chunks per question
- GPT-4 synthesizes grounded answers
- Security via HTTP Bearer (`test_token`) for demo

## Performance tips
- Use a leaner embedding model (`text-embedding-3-small`) to keep latency <10s
- Cache embeddings for repeated documents (future work)
- Handle quota gracefully (TF‑IDF fallback included)

## Environment
- Required: `OPENAI_API_KEY`
- Optional: `CHAT_MODEL` (default `gpt-4`), `EMBEDDING_MODEL` (default `text-embedding-3-small`)