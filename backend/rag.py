from vector_db import VectorDB
from llm import generate_answer, build_prompt

class RAGSystem:
    def __init__(self):
        print("Initializing RAG System...")
        self.db = VectorDB()
        print("RAG System ready!")

    def get_stats(self):
        return self.db.stats()

    def search(self, query, k=3):
        return self.db.search(query, k)

    def query(self, query, k=3, max_length=150):
        results = self.db.search(query, k)

        context = "\n".join(
            r["metadata"]["text"] for r in results["results"]
        )

        prompt = build_prompt(query, context)
        answer = generate_answer(prompt, max_length)

        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": results["results"]
        }

    def insert_documents(self, documents):
        return self.db.insert(documents)
