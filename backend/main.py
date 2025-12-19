from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import RAGSystem
import uvicorn
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGSystem()


class QueryRequest(BaseModel):
    query: str
    k: int = 3
    max_length: int = 150


@app.get("/")
def health():
    return {"status": "RAG service running"}


@app.get("/stats")
def get_stats():
    return {
        "total_documents": len(rag.vector_db.documents),
        "dimension": rag.vector_db.dimension
    }


@app.post("/query")
def query_rag(req: QueryRequest):
    return rag.query(
        req.query,
        k=req.k,
        max_length=req.max_length
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)