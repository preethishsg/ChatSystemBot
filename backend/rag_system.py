import torch
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from typing import List, Dict
from pathlib import Path
from vector_db import VectorDatabase


class RAGSystem:
    """
    Complete RAG system integrating:
    - BGE-micro for embeddings
    - Custom vector database for retrieval
    - GPT-2 for text generation
    """

    def __init__(self, db_path: str = None):
        print("Initializing RAG System...")

        # Initialize BGE-micro for embeddings
        print("Loading BGE-micro model...")
        self.embed_tokenizer = AutoTokenizer.from_pretrained("TaylorAI/bge-micro")
        self.embed_model = AutoModel.from_pretrained("TaylorAI/bge-micro")
        self.embed_model.eval()

        # Initialize GPT-2 for generation
        print("Loading GPT-2 model...")
        self.gen_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gen_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gen_model.eval()

        # Set padding token for GPT-2
        self.gen_tokenizer.pad_token = self.gen_tokenizer.eos_token

        # Initialize or load vector database
        if db_path and Path(db_path).exists():
            print(f"Loading database from {db_path}")
            self.db = VectorDatabase.load(db_path)
        else:
            print("Creating new database")
            self.db = VectorDatabase(dimension=384)  # BGE-micro dimension

        print("RAG System initialized!")

    def encode_text(self, text: str) -> np.ndarray:
        """Generate embeddings using BGE-micro"""
        with torch.no_grad():
            inputs = self.embed_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            outputs = self.embed_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        return embeddings[0]

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embeddings.append(self.encode_text(text))
        return embeddings

    def insert_document(self, text: str, metadata: Dict = None) -> str:
        """Insert a single document into the database"""
        embedding = self.encode_text(text)
        doc_id = self.db.insert(embedding, metadata or {"text": text})
        return doc_id

    def insert_documents(self, documents: List[Dict]) -> List[str]:
        """
        Insert multiple documents.
        Each document should have 'text' or 'data' field.
        """
        texts = []
        processed_docs = []

        for doc in documents:
            # Handle 'data' or 'text' field
            text = doc.get("data") or doc.get("text", "")
            texts.append(text)

            # Create standardized document
            processed_doc = {"text": text}

            if "id" in doc:
                processed_doc["id"] = doc["id"]

            # Add remaining metadata
            for key, value in doc.items():
                if key not in ["data", "text"]:
                    processed_doc[key] = value

            processed_docs.append(processed_doc)

        embeddings = self.encode_batch(texts)
        doc_ids = self.db.batch_insert(embeddings, processed_docs)
        return doc_ids

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k most relevant documents"""
        query_embedding = self.encode_text(query)
        results = self.db.search(query_embedding, k=k)

        formatted_results = []
        for doc_id, score, metadata in results:
            formatted_results.append(
                {
                    "id": doc_id,
                    "score": score,
                    "metadata": metadata,
                }
            )

        return formatted_results

    def generate_response(
        self, query: str, context: str, max_length: int = 150
    ) -> str:
        """Generate response using GPT-2 with retrieved context"""
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

        with torch.no_grad():
            inputs = self.gen_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            outputs = self.gen_model.generate(
                inputs["input_ids"],
                max_length=len(inputs["input_ids"][0]) + max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.gen_tokenizer.eos_token_id,
            )

            response = self.gen_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
            else:
                answer = response[len(prompt):].strip()

        return answer

    def query(self, query: str, k: int = 3, max_length: int = 150) -> Dict:
        """
        Complete RAG query pipeline:
        1. Retrieve relevant documents
        2. Generate response using context
        """
        retrieved_docs = self.retrieve(query, k=k)

        if not retrieved_docs:
            return {
                "query": query,
                "answer": "No relevant documents found in the database.",
                "retrieved_documents": [],
                "context": "",
            }

        context_parts = []
        for doc in retrieved_docs:
            text = doc["metadata"].get("text", "")
            context_parts.append(text)

        context = " ".join(context_parts)
        answer = self.generate_response(query, context, max_length)

        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "context": context[:500],
        }

    def save_db(self, filepath: str):
        """Save the vector database"""
        self.db.save(filepath)

    def get_stats(self) -> Dict:
        """Get database statistics"""
        return self.db.stats()


def initialize_from_documents(
    json_path: str, db_path: str = "vector_db.json"
):
    """Initialize RAG system and populate with documents from JSON file"""
    import json

    rag = RAGSystem()

    with open(json_path, "r") as f:
        documents = json.load(f)

    print(f"Loading {len(documents)} documents...")
    doc_ids = rag.insert_documents(documents)
    print(f"Inserted {len(doc_ids)} documents")

    rag.save_db(db_path)
    print(f"Database saved to {db_path}")

    return rag
