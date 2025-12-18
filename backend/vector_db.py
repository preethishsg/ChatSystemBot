import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorDB:
    def __init__(self, doc_path: str = "documents.json"):
        """
        Lightweight in-memory vector DB for RAG
        """
        self.doc_path = doc_path
        self.documents = []
        self.embeddings = []

        print("Loading embedding model (BGE-micro)...")
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

        self._load_documents()

    def _load_documents(self):
        if not os.path.exists(self.doc_path):
            raise FileNotFoundError(f"{self.doc_path} not found")

        with open(self.doc_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        if not isinstance(self.documents, list):
            raise ValueError("documents.json must be a list of objects")

        self.embeddings = []

        for idx, doc in enumerate(self.documents):
            text = (
                doc.get("data")
		or doc.get("text")
                or doc.get("content")
                or doc.get("page_content")
                or doc.get("description")
            )

            if not text:
                raise ValueError(
                    f"Document at index {idx} missing text field. "
                    f"Available keys: {list(doc.keys())}"
                )

            embedding = self.model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            self.embeddings.append(embedding)

        self.embeddings = np.array(self.embeddings)
        print(f"Loaded {len(self.documents)} documents into vector DB")

    def search(self, query: str, top_k: int = 3):
        """
        Semantic similarity search
        """
        query_emb = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        scores = np.dot(self.embeddings, query_emb)
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "score": float(scores[idx]),
                    "document": self.documents[idx],
                }
            )

        return results
