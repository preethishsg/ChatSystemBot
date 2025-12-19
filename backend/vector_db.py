import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class VectorDB:
    def __init__(self):
        # Load embedding model
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

        # Storage
        self.documents = []
        self.embeddings = []

        # Embedding dimension
        self.dimension = self.model.get_sentence_embedding_dimension()

    def load_documents(self, path="documents.json"):
        """
        Load documents from JSON file and create embeddings.
        Expected schema:
        [
          {
            "id": "...",
            "data": "text content"
          }
        ]
        """
        with open(path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        texts = [doc["data"] for doc in self.documents]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)

    def search(self, query):
        """
        Search documents using cosine similarity.
        Returns a list of:
        {
          id,
          score,
          metadata: { text }
        }
        """
        if not self.documents:
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)

        scores = cosine_similarity(query_embedding, self.embeddings)[0]

        results = []
        for idx, score in enumerate(scores):
            results.append({
                "id": self.documents[idx]["id"],
                "score": float(score),
                "metadata": {
                    "text": self.documents[idx]["data"]
                }
            })

        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x["score"], reverse=True)

        return results
