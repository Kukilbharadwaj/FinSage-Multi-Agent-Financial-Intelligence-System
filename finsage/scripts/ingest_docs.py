# scripts/ingest_docs.py
# Run this script once to build the FAISS index from rag/docs/.
# Usage: python scripts/ingest_docs.py

import os
import sys
import pickle
import faiss
import numpy as np

# Add project root to path so we can import rag modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.embedder import embed


def split_into_chunks(text: str, chunk_size: int = 300) -> list:
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


def main():
    """Build FAISS index from all .txt files in rag/docs/."""
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag", "docs")
    rag_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag")

    index_path = os.path.join(rag_dir, "faiss.index")
    chunks_path = os.path.join(rag_dir, "chunks.pkl")

    print("=" * 60)
    print("FinSage AI — Knowledge Base Ingestion")
    print("=" * 60)

    # Read all .txt files
    all_chunks = []
    txt_files = [f for f in os.listdir(docs_dir) if f.endswith(".txt")]

    if not txt_files:
        print("ERROR: No .txt files found in rag/docs/")
        sys.exit(1)

    for filename in sorted(txt_files):
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = split_into_chunks(content, chunk_size=300)
        print(f"  [DOC] {filename}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\n  Total chunks: {len(all_chunks)}")
    print(f"\n  Embedding chunks using all-MiniLM-L6-v2...")

    # Embed all chunks
    embeddings = embed(all_chunks)
    print(f"  Embedding shape: {embeddings.shape}")

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    print(f"  FAISS index built with {index.ntotal} vectors")

    # Save index and chunks
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\n  [OK] Index saved to: {index_path}")
    print(f"  [OK] Chunks saved to: {chunks_path}")
    print("=" * 60)
    print("Knowledge base ready!")


if __name__ == "__main__":
    main()
