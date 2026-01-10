# backend/api/api.py

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF

from rag_engine import chunk_text, embed_and_store

router = APIRouter()

@router.post("/upload-pdf")
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
