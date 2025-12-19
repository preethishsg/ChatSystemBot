from vector_db import VectorDB
import requests
import os


class RAGSystem:
    def __init__(self):
        self.vector_db = VectorDB()
        self.vector_db.load_documents()

        # ✅ Keep using the current endpoint since you said you can't switch
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

        self.headers = {
            "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
            "Content-Type": "application/json"
        }

    def query(self, query: str, k: int = 3, max_length: int = 150):
        docs = self.vector_db.search(query)[:k]

        if not docs:
            return {
                "answer": "No documents available.",
                "retrieved_documents": []
            }

        context = "\n".join(
            doc["metadata"]["text"] for doc in docs
        )

        prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_length,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            },
            timeout=60
        )

        # ✅ Safe JSON parsing
        try:
            result = response.json()
        except ValueError:
            # Ignore non‑JSON responses (like "Internal Server Error")
            return {
                "answer": "",
                "retrieved_documents": docs
            }

        # ✅ Only return generated text, ignore errors
        if isinstance(result, list) and "generated_text" in result[0]:
            answer = result[0]["generated_text"]
        else:
            answer = ""

        return {
            "answer": answer,
            "retrieved_documents": docs
        }
