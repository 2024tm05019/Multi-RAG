"""
Vector Store: ChromaDB-backed store for all chunk embeddings.
Supports metadata filtering by chunk_type (text/table/image).
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any

# Persistent local ChromaDB
_client = chromadb.PersistentClient(path="./chroma_db")
_collection = _client.get_or_create_collection(
    name="bosch_rag",
    metadata={"hnsw:space": "cosine"},
)


def add_chunks(
    contents: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
) -> None:
    """Add embedded chunks to ChromaDB."""
    _collection.add(
        documents=contents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def query_store(
    query_embedding: List[float],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Retrieve top-k most similar chunks for a query embedding."""
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i],  # cosine similarity
        })
    return chunks


def get_doc_count() -> int:
    """Return total number of indexed chunks."""
    return _collection.count()


def get_unique_documents() -> int:
    """Return number of unique source documents indexed."""
    if _collection.count() == 0:
        return 0
    results = _collection.get(include=["metadatas"])
    filenames = set(m["filename"] for m in results["metadatas"])
    return len(filenames)


def is_empty() -> bool:
    """Check if the vector store has any documents."""
    return _collection.count() == 0
