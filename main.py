"""
Multimodal RAG API
Entry point for the FastAPI application.
"""

import time
from fastapi import FastAPI
from src.api.routes import router

START_TIME = time.time()

app = FastAPI(
    title="2024TM05019 Multimodal RAG API",
    description="Multimodal RAG system for Weldshop Technical Manuals",
    version="1.0.0",
)

app.include_router(router)
app.state.start_time = START_TIME

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
