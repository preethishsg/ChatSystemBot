from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag import RAGSystem

app = FastAPI(title="RAG System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGSystem()

@app.get("/")
def health():
    return {
        "status": "healthy",
        "message": "RAG System API is running",
        "version": "1.0.0"
    }

@app.get("/stats")
def stats():
    return rag.get_stats()

@app.post("/search")
def search(payload: dict):
    return rag.search(payload["query"], payload.get("k", 3))

@app.post("/query")
def query(payload: dict):
    return rag.query(
        payload["query"],
        payload.get("k", 3),
        payload.get("max_length", 150)
    )

@app.post("/insert")
def insert(payload: dict):
    return rag.insert_documents(payload["documents"])
