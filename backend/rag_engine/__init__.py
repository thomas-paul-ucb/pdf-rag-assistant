import os
import traceback
import requests
from pathlib import Path
from dotenv import load_dotenv

# Hugging Face Official Client
from huggingface_hub import InferenceClient

# LangChain / Vector Store Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- PATH LOGIC ---
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
DB_PATH = str(BASE_DIR / "backend" / "faiss_index")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- THE MISSING FUNCTIONS ---
# You must have all three of these defined here for api.py to work:

def chunk_text(text: str):
    """Breaks long text into smaller pieces."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    return text_splitter.split_text(text)

def embed_and_store(chunks: list):
    """Turns text chunks into vectors and saves locally."""
    vectorstore = FAISS.from_texts(chunks, embeddings_model)
    vectorstore.save_local(DB_PATH)

def answer_question(query: str) -> str:
    if not os.path.exists(DB_PATH):
        return "Error: Please upload a PDF first."

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    try:
        # 1. RETRIEVE context from vector store
        db = FAISS.load_local(DB_PATH, embeddings_model, allow_dangerous_deserialization=True)
        docs = db.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        # 2. CONSTRUCT PROMPT
        prompt = f"""Based on the following context, answer the question concisely.

Context:
{context}

Question: {query}

Answer:"""

        # 3. GENERATE using Ollama
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 250
                }
            },
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response generated")
        else:
            return f"Ollama Error ({response.status_code}): {response.text}"

    except requests.exceptions.ConnectionError:
        return "Error: Ollama is not running. Start it with 'ollama serve' or check if it's installed."
    except Exception as e:
        traceback.print_exc()
        return f"System Error: {str(e)}"

