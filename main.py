from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import openai
import faiss
import numpy as np
import requests
import fitz  # PyMuPDF
import os
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from requests.exceptions import RequestException

# === CONFIG ===
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI key in environment variables
MODEL = "gpt-4"

app = FastAPI()
security = HTTPBearer(auto_error=False)

# === HELPERS ===
def ensure_openai_key_present():
    if not openai.api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set. Configure the environment variable before starting the server.")

# === DATA MODELS ===
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# === UTILS ===
def extract_text_from_pdf(pdf_url: str) -> str:
    try:
        response = requests.get(pdf_url, timeout=30)
    except RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch document: {exc}")
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch document")

    with open("temp.pdf", "wb") as f:
        f.write(response.content)

    try:
        doc = fitz.open("temp.pdf")
        full_text = "\n".join([page.get_text() for page in doc])
        doc.close()
    finally:
        try:
            os.remove("temp.pdf")
        except OSError:
            pass
    return full_text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_embeddings(texts):
    ensure_openai_key_present()
    try:
        response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Embedding error: {exc}")
    return [np.array(d["embedding"], dtype=np.float32) for d in response["data"]]

def build_faiss_index(chunks):
    embeddings = get_embeddings(chunks)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, embeddings

def find_top_chunks(question, chunks, index, chunk_embeddings, k=3):
    q_embed = get_embeddings([question])[0]
    D, I = index.search(np.array([q_embed], dtype=np.float32), k)
    return [chunks[i] for i in I[0]]

def generate_answer(context, question):
    ensure_openai_key_present()
    prompt = f"""
You are a helpful assistant. Based on the following context, answer the question clearly:

Context:
{context}

Question:
{question}
"""
    try:
        completion = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")
    return completion.choices[0].message.content.strip()

# === ROUTE ===
@app.post("/hackrx/run", response_model=HackRxResponse)
def run_hackrx(req: HackRxRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or credentials.credentials != "test_token"
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")

    raw_text = extract_text_from_pdf(req.documents)
    chunks = chunk_text(raw_text)
    index, embeddings = build_faiss_index(chunks)

    answers = []
    for question in req.questions:
        top_chunks = find_top_chunks(question, chunks, index, embeddings)
        context = "\n".join(top_chunks)
        answer = generate_answer(context, question)
        answers.append(answer)

    return {"answers": answers}
