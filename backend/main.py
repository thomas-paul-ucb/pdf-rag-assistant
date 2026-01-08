from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "PDF AI Q&A is running"}
