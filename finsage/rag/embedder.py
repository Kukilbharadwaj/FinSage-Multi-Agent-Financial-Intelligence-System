# rag/embedder.py
# Embeddings via the Hugging Face Inference API.
#
# Model: sentence-transformers/all-MiniLM-L6-v2 -> 384 dimensions,
# which is what the Pinecone index is provisioned for.
#
# There is deliberately NO local model. Embeddings come from Hugging Face only.
# If the API is unavailable, calls raise so the failure is visible instead of
# being silently papered over.
#
# NOTE on the endpoint: the old host
#     https://api-inference.huggingface.co/pipeline/feature-extraction/{model}
# was retired and no longer resolves. The current serverless path is
#     https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction

import os
import time
from threading import Lock

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}/pipeline/feature-extraction"

HF_API_KEY = os.getenv("HUGGINGFACE_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or ""

_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30"))
_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))

# One pooled session — a fresh TCP+TLS handshake per request would roughly
# double the ~280ms warm latency of this endpoint.
_session = None
_session_lock = Lock()

# Query embeddings are cached: several agents can call RAG within a single user
# query, and expanded queries recur across turns.
_query_cache: dict = {}
_QUERY_CACHE_MAX = 512


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced."""


def _get_session() -> requests.Session:
    global _session
    if _session is not None:
        return _session

    with _session_lock:
        if _session is None:
            if not HF_API_KEY:
                raise EmbeddingError(
                    "HUGGINGFACE_KEY is not set - embeddings require a Hugging Face API key."
                )
            session = requests.Session()
            session.headers.update(
                {
                    "Authorization": f"Bearer {HF_API_KEY}",
                    "Content-Type": "application/json",
                }
            )
            _session = session
    return _session


def _post(texts: list) -> np.ndarray:
    """POST to the HF Inference API and normalise the response shape."""
    response = _get_session().post(API_URL, json={"inputs": texts}, timeout=_TIMEOUT)

    # 503 means the model is loading (cold start) — worth retrying.
    if response.status_code == 503:
        raise EmbeddingError("model is loading (503)")
    response.raise_for_status()

    vectors = np.asarray(response.json(), dtype=np.float32)

    # Some models return token-level output (n, tokens, dim) rather than a
    # pooled sentence vector — mean-pool it so shapes stay consistent.
    if vectors.ndim == 3:
        vectors = vectors.mean(axis=1)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    if vectors.shape[1] != EMBEDDING_DIM:
        raise EmbeddingError(
            f"Model returned {vectors.shape[1]}-dim vectors but the index expects "
            f"{EMBEDDING_DIM}. Check EMBEDDING_MODEL and the Pinecone index dimension."
        )

    # Normalise so cosine similarity in Pinecone behaves as expected.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def _embed(texts: list) -> np.ndarray:
    """Embed with retries and exponential backoff."""
    last_error = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return _post(texts)
        except Exception as exc:
            last_error = exc
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))  # 1s, 2s

    raise EmbeddingError(
        f"Hugging Face embedding API failed after {_MAX_RETRIES} attempts: {str(last_error)[:200]}"
    )


def warmup() -> None:
    """Open the API connection before the first user query."""
    try:
        _embed(["warmup"])
    except Exception as exc:
        print(f"[WARN] Embedder warmup failed: {str(exc)[:160]}")


def embed(texts: list) -> np.ndarray:
    """
    Batch embed a list of text strings. Used during ingestion.

    Returns a float32 array of shape (len(texts), EMBEDDING_DIM).
    """
    if not texts:
        return np.array([], dtype=np.float32)

    # The API rejects very large payloads, so send in batches.
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    chunks = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    return np.vstack([_embed(chunk) for chunk in chunks]).astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string for knowledge base search.

    Returns a float32 array of shape (1, EMBEDDING_DIM).
    """
    cached = _query_cache.get(text)
    if cached is not None:
        return cached

    vector = _embed([text])[:1]

    if len(_query_cache) >= _QUERY_CACHE_MAX:
        _query_cache.clear()
    _query_cache[text] = vector

    return vector
