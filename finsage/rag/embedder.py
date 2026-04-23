# rag/embedder.py
# Sentence-transformers based embedding for FAISS vector store.
# Loads the model once at module level (~80MB download on first run).

import numpy as np
from sentence_transformers import SentenceTransformer

import os

# Create a cache directory locally if it doesn't exist
cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache")

# Load the model once — runs on CPU, no GPU needed
_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir)


def embed(texts: list) -> np.ndarray:
    """
    Batch embed a list of text strings.
    Used during ingestion to build the FAISS index.

    Args:
        texts: List of text chunks to embed

    Returns:
        numpy array of shape (len(texts), 384)
    """
    return _model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string.
    Used at runtime when searching the knowledge base.

    Args:
        text: Single query string

    Returns:
        numpy array of shape (1, 384)
    """
    return _model.encode([text], convert_to_numpy=True)
