"""
PDF Parser: Extracts text chunks, table chunks, and images from PDFs.
Uses PyMuPDF (fitz) for robust extraction from technical manuals.
"""

import fitz  # PyMuPDF
import base64
from typing import List, Dict, Any


def extract_text_chunks(doc: fitz.Document, filename: str) -> List[Dict[str, Any]]:
    """Extract text blocks from each page as individual chunks."""
    chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4].strip()
            if len(text) > 50:  # Skip very short fragments
                chunks.append({
                    "content": text,
                    "chunk_type": "text",
                    "page": page_num + 1,
                    "filename": filename,
                })
    return chunks


def extract_table_chunks(doc: fitz.Document, filename: str) -> List[Dict[str, Any]]:
    """
    Extract table-like content using PyMuPDF's table finder.
    Falls back to text blocks with tabular structure if no formal tables found.
    """
    chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        try:
            tables = page.find_tables()
            for table in tables.tables:
                rows = table.extract()
                if not rows:
                    continue
                # Convert table rows to readable text
                table_text = ""
                for row in rows:
                    cleaned = [str(cell).strip() if cell else "" for cell in row]
                    table_text += " | ".join(cleaned) + "\n"
                if len(table_text.strip()) > 20:
                    chunks.append({
                        "content": f"TABLE (Page {page_num + 1}):\n{table_text}",
                        "chunk_type": "table",
                        "page": page_num + 1,
                        "filename": filename,
                    })
        except Exception:
            continue
    return chunks


def extract_image_chunks(doc: fitz.Document, filename: str) -> List[Dict[str, Any]]:
    """
    Extract images from PDF pages and encode them as base64.
    Returns list of dicts with base64 image data for VLM summarization.
    """
    chunks = []
    seen_xrefs = set()

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Skip very small images (icons, decorations)
                if len(image_bytes) < 3000:
                    continue

                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                ext = base_image.get("ext", "png")

                chunks.append({
                    "image_b64": image_b64,
                    "ext": ext,
                    "chunk_type": "image",
                    "page": page_num + 1,
                    "filename": filename,
                })
            except Exception:
                continue

    return chunks


def parse_pdf(file_path: str, filename: str) -> Dict[str, List]:
    """
    Main entry point: parse a PDF into text, table, and image chunks.
    
    Returns:
        dict with keys: 'text', 'tables', 'images'
    """
    doc = fitz.open(file_path)
    text_chunks = extract_text_chunks(doc, filename)
    table_chunks = extract_table_chunks(doc, filename)
    image_chunks = extract_image_chunks(doc, filename)
    doc.close()

    return {
        "text": text_chunks,
        "tables": table_chunks,
        "images": image_chunks,
    }
