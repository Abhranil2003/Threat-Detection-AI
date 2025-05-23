# src/api/__init__.py
from fastapi import FastAPI

app = FastAPI()

# Optionally include your routes here
@app.get("/v1/")
def health_check():
    return {"message": "Welcome to Threat Detection API v1"}