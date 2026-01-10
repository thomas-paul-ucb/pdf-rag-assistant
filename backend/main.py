from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import api  # This is your router file

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router)

@app.get("/")
def root():
    return {"message": "PDF AI Q&A is running"}
