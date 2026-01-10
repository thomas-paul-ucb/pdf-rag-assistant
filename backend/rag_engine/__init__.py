import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# 1. New home for HuggingFace logic
from langchain_huggingface import HuggingFaceEmbeddings

# 2. Legacy home for the "Chains" and Hub
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

# 3. Community home for FAISS
from langchain_community.vectorstores import FAISS

# 4. Standard text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables (for the HuggingFace API Token)
# This finds the directory of the current file (__init__.py)
# and looks for the .env file one level up (in the backend/ folder)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# 1. Setup Configuration
# "all-MiniLM-L6-v2" turns text into 384-dimensional vectors
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = "faiss_index"

# Initialize the embedding model globally for efficiency
embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def chunk_text(text: str):
    """
    Breaks long text into smaller pieces so the AI can process them.
    RecursiveCharacterTextSplitter tries to split at paragraphs and sentences.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    return text_splitter.split_text(text)

def embed_and_store(chunks: list):
    """
    Turns text chunks into vectors and saves them to a local folder.
    """
    # Create the vector database from the text chunks
    vectorstore = FAISS.from_texts(chunks, embeddings_model)
    
    # Save the database locally to the 'faiss_index' folder
    vectorstore.save_local(DB_PATH)

def answer_question(query: str) -> str:
    """
    Retrieves relevant chunks and asks the LLM to answer based on them.
    """
    # Check if the database exists
    if not os.path.exists(DB_PATH):
        return "Error: No PDF has been processed yet. Please upload a PDF first."

    # 1. Load the stored vector database
    # allow_dangerous_deserialization is required for loading local pickle files
    db = FAISS.load_local(
        DB_PATH, 
        embeddings_model, 
        allow_dangerous_deserialization=True
    )

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # 2. Setup the LLM (Large Language Model)
    # google/flan-t5-base is a good lightweight model for RAG
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=hf_token,
        task="text2text-generation",
        max_new_tokens=512,
        temperature=0.1,
    )

    # 3. Create the RetrievalQA chain
    # "stuff" chain type simply 'stuffs' all retrieved context into the prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=db.as_retriever()
    )
    
    # 4. Run the query
    response = qa_chain.invoke(query)
    return response["result"]