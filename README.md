# RAG System - Complete Implementation

## 🚀 Project Overview

This is a complete Retrieval-Augmented Generation (RAG) system featuring:
- **Custom Vector Database** with flat index and dot product similarity
- **BGE-micro** embeddings for text encoding (384 dimensions)
- **GPT-2** for response generation
- **FastAPI** backend with RESTful APIs
- **React** frontend with modern chat interface
- **Production-ready** deployment configuration

## 📁 Project Structure

```
rag-system/
├── backend/
│   ├── vector_db.py          # Custom vector database implementation
│   ├── rag_system.py          # RAG pipeline (retrieval + generation)
│   ├── main.py                # FastAPI server
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile             # Docker configuration
│   └── documents.json         # Initial dataset
├── frontend/
│   ├── src/
│   │   └── App.jsx           # React chat interface
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## 🔧 Implementation Details

### 1. Vector Database (`vector_db.py`)

**Features:**
- **Flat index** for simplicity and fast prototyping
- **Insert operations**: Single and batch insertion
- **Search**: Top-k retrieval using dot product similarity
- **Persistence**: Save/load database to JSON
- **Normalization**: Vector normalization for cosine similarity

**Key Methods:**
```python
db = VectorDatabase(dimension=384)
doc_id = db.insert(vector, metadata)        # Single insert
doc_ids = db.batch_insert(vectors, metas)   # Batch insert
results = db.search(query_vector, k=5)      # Top-k search
db.save('db.json')                          # Persist to disk
```

**Similarity Calculation:**
- Uses normalized dot product (equivalent to cosine similarity)
- Formula: `similarity = normalize(query) · normalize(vector)`
- Returns results sorted by similarity score (descending)

### 2. RAG System (`rag_system.py`)

**Components:**
- **BGE-micro**: Sentence embedding model (384d vectors)
- **GPT-2**: Text generation model
- **Custom Vector DB**: For retrieval

**Pipeline:**
1. **Encoding**: Convert text to 384-dimensional embeddings
2. **Retrieval**: Search vector DB for top-k similar documents
3. **Generation**: Use retrieved context with GPT-2 to generate response

**Key Methods:**
```python
rag = RAGSystem(db_path='vector_db.json')
doc_ids = rag.insert_documents(documents)   # Batch insert docs
results = rag.retrieve(query, k=3)          # Retrieve similar docs
response = rag.query(query, k=3)            # Complete RAG pipeline
```

### 3. Backend API (`main.py`)

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/stats` | GET | Database statistics |
| `/insert` | POST | Insert documents |
| `/search` | POST | Vector similarity search |
| `/query` | POST | Complete RAG query |

**Example Requests:**

```bash
# Insert documents
curl -X POST http://localhost:8000/insert \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"text": "Machine learning is a subset of AI...", "metadata": {...}}
    ]
  }'

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is ML?", "k": 5}'

# Query (RAG)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain neural networks", "k": 3}'
```

### 4. Frontend (`App.jsx`)

**Features:**
- Modern chat interface with gradient design
- Real-time stats display
- Retrieved documents visualization
- Similarity scores display
- Configurable backend URL
- Loading states and error handling

**Tech Stack:**
- React with hooks
- Lucide icons
- Tailwind CSS
- Fetch API for backend communication

## 🚀 Local Development

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place documents.json in the backend directory

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Access the application at `http://localhost:5173`

## 🐳 Docker Deployment

```bash
cd backend

# Build image
docker build -t rag-system .

# Run container
docker run -p 8000:8000 rag-system
```

## ☁️ Cloud Deployment

### Option 1: Render (Backend)

1. Create new Web Service on Render
2. Connect your GitHub repository
3. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.10
4. Deploy

### Option 2: Vercel (Frontend)

```bash
cd frontend

# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

Update the `apiUrl` in frontend to point to your backend URL.

### Option 3: Railway

1. Create new project
2. Connect GitHub repo
3. Railway auto-detects configuration
4. Deploy

## 📊 Performance Characteristics

**Vector Database:**
- **Time Complexity**: O(n) for search (flat index)
- **Space Complexity**: O(n*d) where n=documents, d=384
- **Insert**: O(1) amortized
- **Suitable for**: <10K documents (for larger, use FAISS/Annoy)

**Models:**
- **BGE-micro**: ~50MB, fast inference (~10ms per embedding)
- **GPT-2**: ~500MB, moderate inference (~500ms per generation)

## 🔒 Production Considerations

**Currently Implemented:**
✅ Custom vector database with persistence
✅ RESTful API with proper error handling
✅ CORS configuration for cross-origin requests
✅ Docker containerization
✅ Model caching and reuse
✅ Batch operations support

**Would Implement with More Time:**
- Vector database improvements:
  - FAISS/Annoy integration for scalability
  - Approximate nearest neighbor (ANN) search
  - Database sharding for large datasets
  - Redis caching for frequent queries
  
- API enhancements:
  - Authentication and rate limiting
  - Streaming responses for generation
  - Batch query processing
  - WebSocket support for real-time updates
  
- Model improvements:
  - Fine-tuned models for specific domains
  - Model quantization for faster inference
  - GPU acceleration
  - Ensemble retrieval strategies
  
- Monitoring and observability:
  - Prometheus metrics
  - Request logging and tracing
  - Performance monitoring
  - Error tracking (Sentry)
  
- Frontend enhancements:
  - Chat history persistence
  - Document upload interface
  - Admin panel for database management
  - Response streaming
  - Mobile responsiveness improvements

## 📝 API Documentation

Once running, visit:
- **API Docs**: `http://localhost:8000/docs` (Swagger UI)
- **Alternative Docs**: `http://localhost:8000/redoc` (ReDoc)

## 🧪 Testing

```bash
# Test health endpoint
curl http://localhost:8000/

# Test stats
curl http://localhost:8000/stats

# Test query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is artificial intelligence?", "k": 3}'
```

## 🐛 Troubleshooting

**Issue: Models not downloading**
- Ensure internet connection
- Models download on first run (~550MB total)
- Check disk space

**Issue: CORS errors**
- Verify backend URL in frontend
- Check CORS middleware configuration
- Ensure backend is running

**Issue: Out of memory**
- Reduce batch sizes
- Use CPU-only inference
- Consider model quantization

## 📚 References

- [BGE-micro Model](https://huggingface.co/TaylorAI/bge-micro)
- [GPT-2 Model](https://huggingface.co/openai-community/gpt2)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Vector Similarity Search](https://en.wikipedia.org/wiki/Cosine_similarity)

## 👨‍💻 Author

Built as a demonstration of RAG system implementation with custom vector database.

## 📄 License

MIT License - feel free to use for learning and projects!