"""
FastAPI route definitions for the Bosch Weld Timer RAG API.
Implements /health, /ingest, /query endpoints.
"""

import os
import time
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from dotenv import load_dotenv

load_dotenv()

from src.api.schemas import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceReference,
)
from src.ingestion.parser import parse_pdf
from src.ingestion.embedder import embed_texts
from src.models.vision import summarize_image
from src.retrieval.vector_store import (
    add_chunks,
    query_store,
    get_doc_count,
    get_unique_documents,
    is_empty,
)
from src.retrieval.rag_chain import generate_answer

router = APIRouter()
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Returns system status including model readiness and index stats."""
    return HealthResponse(
        status="ok",
        model_ready=bool(os.getenv("OPENAI_API_KEY")),
        indexed_documents=get_unique_documents(),
        index_size=get_doc_count(),
        uptime=round(time.time() - _start_time, 2),
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF upload, parses text/tables/images,
    embeds all chunks, and adds them to ChromaDB.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    start = time.time()

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        parsed = parse_pdf(tmp_path, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {str(e)}")
    finally:
        os.unlink(tmp_path)

    text_chunks = parsed["text"]
    table_chunks = parsed["tables"]
    image_chunks = parsed["images"]

    all_contents = []
    all_metadatas = []
    all_ids = []

    # Process text chunks
    for i, chunk in enumerate(text_chunks):
        all_contents.append(chunk["content"])
        all_metadatas.append({
            "chunk_type": "text",
            "page": chunk["page"],
            "filename": chunk["filename"],
        })
        all_ids.append(f"{file.filename}_text_{i}")

    # Process table chunks
    for i, chunk in enumerate(table_chunks):
        all_contents.append(chunk["content"])
        all_metadatas.append({
            "chunk_type": "table",
            "page": chunk["page"],
            "filename": chunk["filename"],
        })
        all_ids.append(f"{file.filename}_table_{i}")

    # Process image chunks — summarize via VLM first
    image_summary_count = 0
    for i, chunk in enumerate(image_chunks):
        summary = summarize_image(chunk["image_b64"], chunk["ext"], chunk["page"])
        all_contents.append(summary)
        all_metadatas.append({
            "chunk_type": "image",
            "page": chunk["page"],
            "filename": chunk["filename"],
        })
        all_ids.append(f"{file.filename}_image_{i}")
        image_summary_count += 1

    # Embed all chunks together
    try:
        embeddings = embed_texts(all_contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    # Store in ChromaDB
    try:
        add_chunks(all_contents, embeddings, all_metadatas, all_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    return IngestResponse(
        filename=file.filename,
        text_chunks=len(text_chunks),
        table_chunks=len(table_chunks),
        image_chunks=image_summary_count,
        total_chunks=len(all_contents),
        processing_time=round(time.time() - start, 2),
    )


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Accepts a natural language question, retrieves relevant chunks,
    and returns a grounded answer with source references.
    """
    if is_empty():
        raise HTTPException(
            status_code=404,
            detail="No documents indexed yet. Please POST a PDF to /ingest first.",
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start = time.time()

    # Embed the query
    from src.ingestion.embedder import embed_texts
    query_embedding = embed_texts([request.question])[0]

    # Retrieve top-k chunks
    retrieved = query_store(query_embedding, top_k=request.top_k)

    # Generate answer
    result = generate_answer(request.question, retrieved)

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceReference(**s) for s in result["sources"]],
        processing_time=round(time.time() - start, 2),
    )
