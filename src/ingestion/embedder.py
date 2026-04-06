from typing import List
from openai import OpenAI

client = OpenAI()  # ✅ correct for openai 2.x

EMBEDDING_MODEL = "text-embedding-3-small"

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings
