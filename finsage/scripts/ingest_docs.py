# scripts/ingest_docs.py
# Load rag/docs/ into Pinecone.
#
# Usage:
#   python scripts/ingest_docs.py                 # upsert (overwrites by id)
#   python scripts/ingest_docs.py --clear         # wipe the namespace first
#   python scripts/ingest_docs.py --create-index  # create the index if missing
#
# Chunk ids are deterministic (<filename>-<n>), so re-running updates existing
# records in place instead of accumulating duplicates.

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from rag.embedder import EMBEDDING_DIM, MODEL_ID, embed
from rag.vector_store import INDEX_NAME, NAMESPACE, clear, get_index, upsert, vector_count

CHUNK_SIZE = 300


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """
    Split text into chunks of approximately `chunk_size` characters.
    Splits on word boundaries — never splits mid-word.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_length + word_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += word_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def create_index_if_missing() -> None:
    """Create the Pinecone index with the dimensions the embedder produces."""
    from pinecone import Pinecone, ServerlessSpec

    client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if client.has_index(INDEX_NAME):
        print(f"  Index '{INDEX_NAME}' already exists")
        return

    print(f"  Creating index '{INDEX_NAME}' (dim={EMBEDDING_DIM}, metric=cosine)...")
    client.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        ),
    )
    print("  Index created")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest rag/docs into Pinecone")
    parser.add_argument("--clear", action="store_true", help="delete all vectors before ingesting")
    parser.add_argument("--create-index", action="store_true", help="create the index if it is missing")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(root, "rag", "docs")

    print("=" * 62)
    print("FinSage AI - Knowledge Base Ingestion (Pinecone)")
    print("=" * 62)
    print(f"  Model:     {MODEL_ID} ({EMBEDDING_DIM}d, Hugging Face API)")
    print(f"  Index:     {INDEX_NAME}")
    print(f"  Namespace: {NAMESPACE or '(default)'}")
    print()

    if args.create_index:
        create_index_if_missing()

    # Verify the index dimension matches the embedder before doing any work.
    try:
        get_index()
    except Exception as exc:
        print(f"  ERROR: {exc}")
        print("  Hint: run with --create-index to create it automatically.")
        return 1

    txt_files = sorted(f for f in os.listdir(docs_dir) if f.endswith(".txt"))
    if not txt_files:
        print("  ERROR: No .txt files found in rag/docs/")
        return 1

    records = []
    for filename in txt_files:
        with open(os.path.join(docs_dir, filename), "r", encoding="utf-8") as f:
            content = f.read()

        chunks = split_into_chunks(content)
        print(f"  [DOC] {filename}: {len(chunks)} chunks")

        stem = filename.replace(".txt", "")
        for i, chunk in enumerate(chunks):
            records.append({"id": f"{stem}-{i}", "text": chunk, "source": filename})

    print(f"\n  Total chunks: {len(records)}")

    if args.clear:
        print("  Clearing existing vectors...")
        clear()

    print(f"  Embedding via Hugging Face API...")
    vectors = embed([r["text"] for r in records])
    print(f"  Embedding shape: {vectors.shape}")

    if vectors.shape[1] != EMBEDDING_DIM:
        print(f"  ERROR: got {vectors.shape[1]}-dim vectors, index expects {EMBEDDING_DIM}")
        return 1

    payload = [
        {
            "id": record["id"],
            "values": vectors[i].tolist(),
            # Chunk text rides along as metadata so retrieval needs no side file.
            "metadata": {"text": record["text"], "source": record["source"]},
        }
        for i, record in enumerate(records)
    ]

    print(f"  Upserting {len(payload)} vectors to Pinecone...")
    sent = upsert(payload)

    print(f"\n  [OK] Upserted {sent} vectors")
    print(f"  [OK] Index now reports {vector_count()} vectors")
    print("=" * 62)
    print("Knowledge base ready!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
