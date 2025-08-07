# HackRx Document Query API

LLM-powered API that answers questions from a provided PDF document using embeddings (FAISS) and GPT-4.

## Quickstart

1) Create a virtual env and install deps

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure environment

Create a `.env` (or export) with your OpenAI key:

```bash
export OPENAI_API_KEY=your_key_here
```

Alternatively copy `.env.example` to `.env` and export it in your shell.

3) Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

4) Open Swagger UI

Visit `http://127.0.0.1:8000/docs`

- Click the Authorize button
- For the HTTP bearer token field, enter only: `test_token`
  - Swagger will add the `Bearer` prefix automatically
- Click Authorize and close the modal

5) Try the endpoint

Use the sample body below or `test_payload.json`:

```json
{
  "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
  "questions": ["Is maternity covered?", "What is the waiting period?"]
}
```

### curl example

```bash
curl -X POST "http://127.0.0.1:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test_token" \
  -d @test_payload.json
```

If the Authorization header is missing or incorrect, the API returns `401 Unauthorized`.

## How it works

- Extracts text from the provided PDF URL (PyMuPDF)
- Splits into chunks and embeds with `text-embedding-ada-002`
- Builds a FAISS index and retrieves top-matching chunks per question
- Calls GPT-4 to produce answers grounded in retrieved context

## Environment

- `OPENAI_API_KEY` must be set in your shell before calling the endpoint

## Notes

- This project uses HTTP Bearer in Swagger. Enter only the token value `test_token` in the Authorize modal.
- For large PDFs, first request may take longer due to embedding.