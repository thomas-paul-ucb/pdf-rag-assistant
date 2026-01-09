from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
from fastapi.middleware.cors import CORSMiddleware
from rag_engine import chunk_text, embed_and_store

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for dev only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "PDF AI Q&A is running"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "".join([page.get_text() for page in pdf])
        pdf.close()

        chunks = chunk_text(text)
        embed_and_store(chunks)

        return {"message": "PDF uploaded, text embedded", "chunks": len(chunks)}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})