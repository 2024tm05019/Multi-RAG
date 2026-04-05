"""
Embedder: Generates vector embeddings for text, table, and image-summary chunks.
Uses OpenAI text-embedding-3-small model.
"""

import os
from typing import List
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of text strings into vectors.
    Batches requests for efficiency.
    """
    if not texts:
        return []

    # OpenAI allows up to 2048 inputs per request
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
