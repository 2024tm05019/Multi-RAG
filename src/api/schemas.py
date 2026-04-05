"""
Pydantic schemas for all API request and response models.
"""

from pydantic import BaseModel
from typing import List


class SourceReference(BaseModel):
    filename: str
    page: int
    chunk_type: str  # "text", "table", or "image"


class IngestResponse(BaseModel):
    filename: str
    text_chunks: int
    table_chunks: int
    image_chunks: int
    total_chunks: int
    processing_time: float


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    indexed_documents: int
    index_size: int
    uptime: float
