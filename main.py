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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()
# === CONFIG ===
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI key in environment variables
MODEL = os.getenv("CHAT_MODEL", "gpt-4")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

app = FastAPI()
security = HTTPBearer(auto_error=False)

@app.get("/")
def root():
    return {
        "name": "HackRx Document QA API",
        "version": "1.0",
        "message": "Welcome to the HackRx Document QA API. Use /docs for Swagger UI.",
        "docs_url": "/docs",
        "post_endpoint": "/hackrx/run",
        "auth": "Send Authorization: Bearer test_token"
    }

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

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
        response = openai.Embedding.create(input=texts, model=EMBEDDING_MODEL)
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
You are an insurance policy assistant. Use ONLY the text from the document below.
Give a short, exact sentence or clause from the document that answers the question.
Do NOT add explanations or extra words.

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

def find_top_chunks_tfidf(question: str, chunks, k: int = 3):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks + [question])
    chunk_matrix = matrix[:-1]
    question_vector = matrix[-1]
    similarities = cosine_similarity(question_vector, chunk_matrix).flatten()
    top_indices = np.argsort(similarities)[-k:][::-1]
    return [chunks[i] for i in top_indices]


def naive_answer_from_context(context: str, question: str, max_chars: int = 600) -> str:
    sentences = [s.strip() for s in context.replace("\n", " ").split(".") if s.strip()]
    keywords = [w.lower() for w in question.split() if len(w) > 3]
    ranked = []
    for sentence in sentences:
        score = sum(1 for kw in keywords if kw in sentence.lower())
        ranked.append((score, sentence))
    ranked.sort(reverse=True)
    answer_parts = []
    total_len = 0
    for _, sent in ranked[:5]:
        if total_len + len(sent) + 2 > max_chars:
            break
        answer_parts.append(sent)
        total_len += len(sent) + 2
    if not answer_parts:
        answer_parts = sentences[:2]
    return ". ".join(answer_parts).strip() or context[:max_chars]

# === ROUTE ===
@app.post("/hackrx/run", response_model=HackRxResponse)
def run_hackrx(req: HackRxRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    auth_value = credentials.credentials.strip()
    if credentials.scheme.lower() == "bearer" and auth_value.startswith("Bearer "):
        auth_value = auth_value.replace("Bearer ", "").strip()

    if auth_value != "test_token":
        raise HTTPException(status_code=403, detail="Forbidden: Invalid Token")

    raw_text = extract_text_from_pdf(req.documents)
    chunks = chunk_text(raw_text)

    use_faiss = True
    try:
        index, embeddings = build_faiss_index(chunks)
    except HTTPException as exc:
        if isinstance(exc.detail, str) and exc.detail.startswith("Embedding error"):
            use_faiss = False
        else:
            raise

    answers = []
    for question in req.questions:
        if use_faiss:
            top_chunks = find_top_chunks(question, chunks, index, embeddings)
        else:
            top_chunks = find_top_chunks_tfidf(question, chunks)
        context = "\n".join(top_chunks)
        try:
            answer = generate_answer(context, question)
        except HTTPException as exc:
            if isinstance(exc.detail, str) and exc.detail.startswith("LLM error"):
                answer = naive_answer_from_context(context, question)
            else:
                raise
        answers.append(answer)

    return {"answers": answers}
