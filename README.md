# 📚 PDF RAG Assistant

A local, privacy-focused PDF question-answering system powered by Ollama and RAG (Retrieval-Augmented Generation).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange.svg)

## ✨ Features

- 🔒 **100% Local & Private** - All processing happens on your machine
- 💰 **Completely Free** - No API costs or rate limits
- 🚀 **Fast Responses** - Uses efficient embedding + vector search
- 📄 **PDF Support** - Upload and analyze any PDF document
- 🤖 **Powered by Ollama** - Leverages local LLMs (Llama 3.2)
- 🎨 **Clean UI** - Simple, modern chat interface

## 🛠️ Tech Stack

**Backend:**
- FastAPI - REST API framework
- LangChain - RAG orchestration
- FAISS - Vector similarity search
- Ollama - Local LLM inference
- HuggingFace Embeddings - Text vectorization

**Frontend:**
- Vanilla JavaScript
- Tailwind CSS

## 📋 Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/download) installed and running

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd pdf-rag-assistant
```

### 2. Install Ollama and pull the model
```bash
# Download from https://ollama.com/download
# Then pull the model:
ollama pull llama3.2:3b
```

### 3. Set up Python environment
```bash
cd backend
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the `backend` directory:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

### 5. Run the application
```bash
# Start the backend (from backend directory)
uvicorn api:app --reload

# Open in browser
# http://127.0.0.1:8000
```

## 📖 Usage

1. **Upload PDF**: Click "Upload PDF" and select your document
2. **Wait for indexing**: The system will process and vectorize the content
3. **Ask questions**: Type your question and get AI-powered answers based on the document

## 🏗️ Project Structure
```
pdf-rag-assistant/
├── backend/
│   ├── api.py              # FastAPI endpoints
│   ├── rag_engine/
│   │   └── __init__.py     # Core RAG logic
│   ├── faiss_index/        # Vector store (generated)
│   ├── requirements.txt    # Python dependencies
│   └── .env               # Environment config
└── frontend/
    └── index.html          # Web interface
```

## 🔧 How It Works

1. **Document Processing**: PDF is extracted and split into chunks
2. **Embedding**: Text chunks are converted to vectors using `all-MiniLM-L6-v2`
3. **Storage**: Vectors are stored in FAISS for fast retrieval
4. **Query**: User question is embedded and similar chunks are retrieved
5. **Generation**: Ollama generates an answer using retrieved context

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📝 License

MIT License - feel free to use this project for personal or commercial purposes.

## 🙏 Acknowledgments

- [Ollama](https://ollama.com) for local LLM inference
- [LangChain](https://langchain.com) for RAG framework
- [FastAPI](https://fastapi.tiangolo.com) for the backend API

---

**Made with ❤️ for privacy-focused AI applications**
