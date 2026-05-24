import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# Load API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1️⃣ Load & Chunk Data
def load_chunks(filepath, chunk_size=300, chunk_overlap=50):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare document chunks
chunks = load_chunks("data/sample.txt")
chunk_embeddings = model.encode(chunks)

# Create FAISS index
dimension = chunk_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_embeddings))

# Retrieve relevant chunks
def retrieve_chunks(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]

# Generate answer
def generate_answer(query):
    context = "\n".join(retrieve_chunks(query))
    prompt = f"""Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content'].strip()

# Main chatbot loop
if __name__ == "__main__":
    print("🤖 RAG Chatbot (type 'exit' to quit)\n")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        answer = generate_answer(query)
        print("Bot:", answer, "\n")
