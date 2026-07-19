# rag/knowledge_base.py
# Knowledge base query layer, backed by Pinecone.
#
# Previously this loaded a local FAISS index plus a chunks.pkl side file on
# every call. Both are gone: vectors and their source text now live together
# in Pinecone, so there is no local artifact to load, cache, or keep in sync.

from rag.embedder import embed_query
from rag.embedder import warmup as _warm_embedder
from rag.vector_store import VectorStoreError, is_ready
from rag.vector_store import query as _vector_query
from rag.vector_store import vector_count
from rag.vector_store import warmup as _warm_store

_EMPTY_INDEX_MESSAGE = (
    "Knowledge base is empty. Run 'python scripts/ingest_docs.py' to load "
    "rag/docs/ into Pinecone. Using general knowledge for now."
)

# Retrieval is now two network hops (HF embed + Pinecone query), so identical
# lookups are cached whole. The embedder's own cache only saves the first hop.
# The corpus is static between ingests, so results are safe to reuse.
_result_cache: dict = {}
_RESULT_CACHE_MAX = 256


def clear_cache() -> None:
    """Drop cached retrievals — call after re-ingesting."""
    _result_cache.clear()


def warmup() -> None:
    """Open the embedding API and Pinecone connections at application startup."""
    _warm_embedder()
    _warm_store()


def query_kb(query: str, top_k: int = 4) -> str:
    """
    Query the knowledge base and return matching text chunks.

    Args:
        query: User question or search query
        top_k: Number of top matches to return

    Returns:
        Matching chunks joined with double newlines, or an explanatory
        message when the store is unavailable or empty.
    """
    cache_key = (query, top_k)
    cached = _result_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        query_vector = embed_query(query)[0].tolist()
        matches = _vector_query(query_vector, top_k=top_k)

        if not matches:
            # Distinguish "nothing relevant" from "nothing ingested" — these
            # need very different fixes.
            if vector_count() == 0:
                return _EMPTY_INDEX_MESSAGE
            return "No relevant information found in the knowledge base."

        text = "\n\n".join(match["text"] for match in matches if match.get("text"))

        if len(_result_cache) >= _RESULT_CACHE_MAX:
            _result_cache.clear()
        _result_cache[cache_key] = text

        return text

    except VectorStoreError as exc:
        return f"Knowledge base unavailable: {str(exc)[:200]}"
    except Exception as exc:
        return f"Error querying knowledge base: {str(exc)[:200]}"


def query_kb_with_sources(query: str, top_k: int = 4) -> list:
    """
    Same as query_kb but returns structured matches with scores and sources.

    Returns a list of {"text": ..., "source": ..., "score": ...}.
    """
    try:
        query_vector = embed_query(query)[0].tolist()
        return _vector_query(query_vector, top_k=top_k)
    except Exception:
        return []


def is_available() -> bool:
    """Return whether the knowledge base is reachable and populated."""
    return is_ready() and vector_count() > 0
