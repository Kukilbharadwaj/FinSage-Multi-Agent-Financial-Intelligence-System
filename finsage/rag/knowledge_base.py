# rag/knowledge_base.py
# FAISS index load and query functions for the RAG knowledge base.

import os
import pickle
import faiss
import numpy as np

from rag.embedder import embed_query

# Paths relative to project root
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss.index")
CHUNKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunks.pkl")


def load_index():
    """
    Load the FAISS index and chunk texts from disk.

    Returns:
        Tuple of (faiss_index, chunks_list) or (None, None) if files don't exist.
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return None, None

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def query_kb(query: str, top_k: int = 4) -> str:
    """
    Query the FAISS knowledge base and return matching text chunks.

    Args:
        query: User question or search query
        top_k: Number of top matches to return

    Returns:
        String of matching chunks joined with double newlines.
        Returns a helpful message if the index doesn't exist yet.
    """
    index, chunks = load_index()

    if index is None or chunks is None:
        return (
            "Knowledge base not built yet. "
            "Run 'python scripts/ingest_docs.py' to build the FAISS index from rag/docs/. "
            "Using general knowledge for now."
        )

    try:
        query_vector = embed_query(query)
        distances, indices = index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(chunks):
                results.append(chunks[idx])

        if not results:
            return "No relevant information found in the knowledge base."

        return "\n\n".join(results)

    except Exception as e:
        return f"Error querying knowledge base: {str(e)}"
