import os
import requests

HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HF_TOKEN = os.getenv("HF_API_TOKEN")

def build_prompt(query, context):
    return f"""
You are a helpful AI assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

def generate_answer(prompt, max_length=150):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_length,
            "temperature": 0.3,
            "top_p": 0.9,
            "return_full_text": False
        }
    }

    response = requests.post(
        HF_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"LLM generation error {response.status_code}: {response.text}"
        )

    return response.json()[0]["generated_text"]

