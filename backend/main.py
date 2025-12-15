from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from pathlib import Path

from rag_system import RAGSystem, initialize_from_documents

app = FastAPI(title="RAG System API", version="1.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
rag_system: Optional[RAGSystem] = None
DB_PATH = "vector_db.json"

# Request/Response Models
class Document(BaseModel):
    text: str
    metadata: Optional[Dict] = None

class InsertRequest(BaseModel):
    documents: List[Document]

class InsertResponse(BaseModel):
    success: bool
    document_ids: List[str]
    message: str

class SearchRequest(BaseModel):
    query: str
    k: int = 5

class SearchResponse(BaseModel):
    results: List[Dict]

class QueryRequest(BaseModel):
    query: str
    k: int = 3
    max_length: int = 150

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_documents: List[Dict]
    context: str

class StatsResponse(BaseModel):
    total_documents: int
    dimension: int
    next_id: int

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    
    print("Starting RAG System...")
    
    # Check if we need to initialize from documents.json
    documents_path = Path("documents.json")
    
    if documents_path.exists() and not Path(DB_PATH).exists():
        print("Initializing database from documents.json...")
        rag_system = initialize_from_documents(str(documents_path), DB_PATH)
    else:
        print("Loading existing database...")
        rag_system = RAGSystem(db_path=DB_PATH if Path(DB_PATH).exists() else None)
    
    print("RAG System ready!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "RAG System API is running",
        "version": "1.0.0"
    }

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics"""
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    stats = rag_system.get_stats()
    return StatsResponse(**stats)

@app.post("/insert", response_model=InsertResponse)
async def insert_documents(request: InsertRequest):
    """Insert documents into the vector database"""
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        # Convert Pydantic models to dicts
        documents = []
        for doc in request.documents:
            doc_dict = {"text": doc.text}
            if doc.metadata:
                doc_dict.update(doc.metadata)
            documents.append(doc_dict)
        
        # Insert documents
        doc_ids = rag_system.insert_documents(documents)
        
        # Save database
        rag_system.save_db(DB_PATH)
        
        return InsertResponse(
            success=True,
            document_ids=doc_ids,
            message=f"Successfully inserted {len(doc_ids)} documents"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inserting documents: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for similar documents"""
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        results = rag_system.retrieve(request.query, k=request.k)
        return SearchResponse(results=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Complete RAG query: retrieve +ßgenerate"""
    if rag_system is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        result = rag_system.query(
            request.query,
            k=request.k,
            max_length=request.max_length
        )
        return QueryResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)