# rag/vector_store.py
# Pinecone vector store for the FinSage knowledge base.
#
# Replaces the previous local FAISS index. Differences that matter:
#
#   - No index file or chunks.pkl on disk. Chunk text now lives in Pinecone
#     metadata under "text", so retrieval returns the passage directly and
#     there is no second artifact that can drift out of sync with the vectors.
#   - Index dimension must match the embedding model: all-MiniLM-L6-v2 -> 384.
#   - Metric is cosine, and rag/embedder.py L2-normalises every vector.
#
# The client and index handle are cached at module level; constructing them
# per query would add a round trip to every retrieval.

import os
from threading import Lock

from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "finsage")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")
API_KEY = os.getenv("PINECONE_API_KEY", "")

_index = None
_lock = Lock()
_init_error = ""


class VectorStoreError(RuntimeError):
    """Raised when the vector store is unavailable or misconfigured."""


def get_index():
    """Return a cached Pinecone index handle, connecting on first use."""
    global _index, _init_error

    if _index is not None:
        return _index

    with _lock:
        if _index is not None:
            return _index

        if not API_KEY:
            _init_error = "PINECONE_API_KEY is not set"
            raise VectorStoreError(_init_error)

        try:
            from pinecone import Pinecone

            client = Pinecone(api_key=API_KEY)

            if not client.has_index(INDEX_NAME):
                _init_error = (
                    f"Pinecone index '{INDEX_NAME}' does not exist. "
                    f"Create it with dimension 384 and metric cosine, or run "
                    f"'python scripts/ingest_docs.py --create-index'."
                )
                raise VectorStoreError(_init_error)

            _index = client.Index(INDEX_NAME)
            return _index

        except VectorStoreError:
            raise
        except Exception as exc:
            _init_error = f"Pinecone connection failed: {str(exc)[:200]}"
            raise VectorStoreError(_init_error) from exc


def warmup() -> None:
    """Establish the Pinecone connection at application startup."""
    try:
        get_index().describe_index_stats()
    except Exception as exc:
        print(f"[WARN] Pinecone warmup failed: {str(exc)[:160]}")


def is_ready() -> bool:
    """Return whether the vector store is reachable."""
    try:
        get_index()
        return True
    except Exception:
        return False


def vector_count() -> int:
    """Return how many vectors the index currently holds."""
    try:
        stats = get_index().describe_index_stats()
        return int(stats.total_vector_count or 0)
    except Exception:
        return 0


def upsert(vectors: list, batch_size: int = 100) -> int:
    """
    Upsert records into Pinecone.

    Args:
        vectors: list of {"id": str, "values": [...], "metadata": {...}}
        batch_size: records per request (Pinecone caps payload size)

    Returns the number of records sent.
    """
    index = get_index()
    sent = 0

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch, namespace=NAMESPACE)
        sent += len(batch)

    return sent


def query(vector, top_k: int = 4) -> list:
    """
    Query Pinecone for the nearest chunks.

    Returns a list of {"text": str, "score": float, "source": str}.
    """
    index = get_index()

    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE,
    )

    matches = []
    for match in result.get("matches", []):
        metadata = match.get("metadata") or {}
        matches.append(
            {
                "text": metadata.get("text", ""),
                "source": metadata.get("source", ""),
                "score": float(match.get("score") or 0.0),
            }
        )

    # Serverless Pinecone gathers results across shards and does not always
    # return them globally sorted (observed: 0.264 before 0.270). Agents
    # truncate the RAG context they receive, so the strongest match has to be
    # first or it can be cut off entirely.
    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches


def clear() -> None:
    """Delete every vector in the namespace — used before a clean re-ingest."""
    try:
        get_index().delete(delete_all=True, namespace=NAMESPACE)
    except Exception as exc:
        # A 404 here just means the namespace is already empty.
        if "not found" not in str(exc).lower() and "404" not in str(exc):
            raise
