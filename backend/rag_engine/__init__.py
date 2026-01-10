import os
from dotenv import load_dotenv

# 1. New home for HuggingFace logic
from langchain_huggingface import HuggingFaceEmbeddings

# 2. Legacy home for the "Chains" and Hub
from langchain_classic.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# 3. Community home for FAISS
from langchain_community.vectorstores import FAISS

# 4. Standard text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables (for the HuggingFace API Token)
load_dotenv()

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
    
    # 2. Setup the LLM (Large Language Model)
    # google/flan-t5-base is a good lightweight model for RAG
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base", 
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )

    # 3. Create the RetrievalQA chain
    # "stuff" chain type simply 'stuffs' all retrieved context into the prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=db.as_retriever(search_kwargs={"k": 3})
    )
    
    # 4. Run the query
    return qa_chain.invoke(query)["result"]