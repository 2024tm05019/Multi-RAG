"""
RAG Chain: Retrieves relevant chunks and generates grounded answers using GPT-4o.
Custom prompt template tailored for industrial weld timer documentation.
"""

import os
from typing import List, Dict, Any
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT_TEMPLATE = """You are a technical expert assistant specializing in Bosch Rexroth 
industrial welding equipment, specifically the PSx 6xxx.630 Timer and I/O Level systems.

Use ONLY the retrieved context below to answer the question.
If the answer is not found in the context, respond with:
"I could not find this information in the indexed documents."

Do NOT make up specifications, pin numbers, voltages, or any technical values.

Retrieved Context:
{context}

Question: {question}

Provide a clear, technically accurate answer. If the context includes table data or 
image descriptions relevant to the answer, reference them explicitly.

Answer:"""


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        chunk_type = meta.get("chunk_type", "text")
        page = meta.get("page", "?")
        filename = meta.get("filename", "unknown")
        context_parts.append(
            f"[Source {i} | File: {filename} | Page: {page} | Type: {chunk_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run the RAG chain: build context → fill prompt → call LLM → return answer.
    
    Returns:
        dict with 'answer' (str) and 'sources' (list of source metadata)
    """
    context = build_context(retrieved_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.1,  # Low temperature for factual technical answers
    )

    answer = response.choices[0].message.content.strip()

    sources = [
        {
            "filename": c["metadata"].get("filename", "unknown"),
            "page": c["metadata"].get("page", 0),
            "chunk_type": c["metadata"].get("chunk_type", "text"),
        }
        for c in retrieved_chunks
    ]

    return {"answer": answer, "sources": sources}
