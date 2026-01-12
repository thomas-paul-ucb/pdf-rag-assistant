from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from api import api  # This is your router file
from fastapi.staticfiles import StaticFiles  # Add this
from fastapi.responses import FileResponse    # Add this
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router)

# --- MOUNT STATIC FILES ---
# This serves everything in the 'static' folder at the root URL
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")
