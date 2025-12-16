import os
import torch
import requests
import numpy as np
from typing import List, Dict
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from vector_db import VectorDatabase


class RAGSystem:
    """
    RAG System:
    - Local embeddings using BGE-micro
    - Custom vector database for retrieval
    - Hosted lightweight LLM (Hugging Face Inference API) for generation
    """

    def __init__(self, db_path: str = None):
        print("Initializing RAG System...")

        # -----------------------------
        # Embedding Model (Local)
        # -----------------------------
        print("Loading embedding model (BGE-micro)...")
        self.embed_tokenizer = AutoTokenizer.from_pretrained("TaylorAI/bge-micro")
        self.embed_model = AutoModel.from_pretrained("TaylorAI/bge-micro")
        self.embed_model.eval()

        # -----------------------------
        # Vector Database
        # -----------------------------
        if db_path and Path(db_path).exists():
            print(f"Loading vector DB from {db_path}")
            self.db = VectorDatabase.load(db_path)
        else:
            print("Creating new vector DB")
            self.db = VectorDatabase(dimension=384)

        # -----------------------------
        # Hosted LLM Config
        # -----------------------------
        self.hf_api_token = os.getenv("HF_API_TOKEN")
        self.hf_model_url = (
            "https://api-inference.huggingface.co/models/google/flan-t5-small"
        )

        if not self.hf_api_token:
            print("WARNING: HF_API_TOKEN not set. Generation will fail.")

        print("RAG System initialized successfully!")

    # --------------------------------------------------
    # Embedding
    # --------------------------------------------------
    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.embed_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            outputs = self.embed_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return embedding[0]

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self.encode_text(text) for text in texts]

    # --------------------------------------------------
    # Insert
    # --------------------------------------------------
    def insert_documents(self, documents: List[Dict]) -> List[str]:
        texts = []
        processed_docs = []

        for doc in documents:
            text = doc.get("data") or doc.get("text", "")
            texts.append(text)

            metadata = {"text": text}
            for k, v in doc.items():
                if k not in ["data", "text"]:
                    metadata[k] = v

            processed_docs.append(metadata)

        embeddings = self.encode_batch(texts)
        return self.db.batch_insert(embeddings, processed_docs)

    # --------------------------------------------------
    # Retrieve
    # --------------------------------------------------
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        query_embedding = self.encode_text(query)
        results = self.db.search(query_embedding, k=k)

        return [
            {"id": doc_id, "score": score, "metadata": metadata}
            for doc_id, score, metadata in results
        ]

    # --------------------------------------------------
    # Hosted LLM Generation (Optimized Prompt)
    # --------------------------------------------------
    def generate_response(self, query: str, context: str, max_length: int = 150) -> str:
        if not self.hf_api_token:
            return "HF_API_TOKEN not configured."

        headers = {
            "Authorization": f"Bearer {self.hf_api_token}",
            "Content-Type": "application/json",
        }

        # 🔥 Optimized RAG Prompt
        prompt = f"""
You are an intelligent assistant answering questions strictly using the provided context.

Rules:
- Use only the given context.
- If the answer is not present, say: "The information is not available in the provided documents."
- Answer clearly and concisely.

Context:
{context}

Question:
{query}

Answer:
"""

        payload = {
            "inputs": prompt.strip(),
            "parameters": {
                "max_new_tokens": max_length,
                "temperature": 0.2,
                "top_p": 0.9,
                "do_sample": False,
            },
        }

        try:
            response = requests.post(
                self.hf_model_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            if isinstance(result, list) and "generated_text" in result[0]:
                return result[0]["generated_text"].strip()

            return str(result)

        except Exception as e:
            return f"LLM generation error: {str(e)}"

    # --------------------------------------------------
    # Full RAG Query
    # --------------------------------------------------
    def query(self, query: str, k: int = 3, max_length: int = 150) -> Dict:
        retrieved_docs = self.retrieve(query, k=k)

        if not retrieved_docs:
            return {
                "query": query,
                "answer": "No relevant documents found.",
                "retrieved_documents": [],
                "context": "",
            }

        context = " ".join(
            doc["metadata"].get("text", "") for doc in retrieved_docs
        )

        answer = self.generate_response(query, context, max_length)

        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "context": context[:500],
        }

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def save_db(self, filepath: str):
        self.db.save(filepath)

    def get_stats(self) -> Dict:
        return self.db.stats()


def initialize_from_documents(json_path: str, db_path: str = "vector_db.json"):
    import json

    rag = RAGSystem()

    with open(json_path, "r") as f:
        documents = json.load(f)

    print(f"Loading {len(documents)} documents...")
    rag.insert_documents(documents)
    rag.save_db(db_path)

    print("Database initialized successfully.")
    return rag
