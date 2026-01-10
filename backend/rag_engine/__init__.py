from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize empty FAISS index
embedding_dim = 384  # Dimension for all-MiniLM
index = faiss.IndexFlatL2(embedding_dim)
stored_chunks = []  # Keep original text chunks for retrieval

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_and_store(chunks):
    global stored_chunks
    embeddings = model.encode(chunks)
    index.add(np.array(embeddings).astype("float32"))
    stored_chunks.extend(chunks)

def query_top_k(question, k=3):
    embedding = model.encode([question])
    D, I = index.search(np.array(embedding).astype("float32"), k)
    results = [stored_chunks[i] for i in I[0]]
    return results
