from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF

app = FastAPI()

@app.get("/")
def root():
    return {"message": "PDF AI Q&A is running"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        pdf.close()

        return JSONResponse(content={"message": "PDF processed", "text": text[:1000] + "..."})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})