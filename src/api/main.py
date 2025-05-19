from fastapi import FastAPI
from api.v1.endpoints import router as v1_router

app = FastAPI(
    title="Threat Detection API",
    version="1.0.0"
)

app.include_router(v1_router, prefix="/api/v1")
