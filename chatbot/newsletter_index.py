"""
newsletter_index.py

Builds and saves the unified MSCC knowledge index, which now combines:
  - newsletter PDF chunks (from newsletter_ingest.ingest_all)
  - website page chunks   (from web_ingest.ingest_pages)        [optional]

The single output file (newsletter_index.pkl) contains:
  - sentence-transformer embeddings (numpy array)
  - BM25 index (rank_bm25)
  - chunk metadata list

Run this script:

    # Index ONLY the PDFs already in content/Newsletters/  (legacy behavior)
    python3 newsletter_index.py

    # Or use ingest_website.py to crawl + download new PDFs + rebuild index in
    # one shot. That script calls build_index_from_chunks() under the hood.
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from newsletter_ingest import ingest_all

INDEX_PATH = os.path.join(os.path.dirname(__file__), "newsletter_index.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"   # fast, small, good quality


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def build_index_from_chunks(chunks: list[dict], save: bool = True) -> dict:
    """
    Build embeddings + BM25 from a chunk list and (optionally) persist
    the index. Returns the in-memory data dict whether or not it was saved.
    """
    if not chunks:
        print("[index] No chunks to index.")
        return {"chunks": [], "embeddings": np.zeros((0, 384), dtype=np.float32),
                "bm25": None, "model_name": MODEL_NAME}

    texts      = [c["text"]                         for c in chunks]
    searchable = [c.get("searchable", c["text"])    for c in chunks]

    print(f"[index] Building embeddings for {len(chunks)} chunks with '{MODEL_NAME}'…")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)

    print("[index] Building BM25 index…")
    tokenized = [tokenize(t) for t in searchable]
    bm25 = BM25Okapi(tokenized)

    data = {
        "chunks":     chunks,
        "embeddings": embeddings,
        "bm25":       bm25,
        "model_name": MODEL_NAME,
    }

    if save:
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(data, f)
        print(f"[index] Saved index to: {INDEX_PATH}")

    # Per-source-type counts for visibility
    by_type: dict[str, int] = {}
    for c in chunks:
        t = c.get("source_type", "newsletter_pdf")
        by_type[t] = by_type.get(t, 0) + 1
    print("[index] Chunks by source_type: " +
          ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    print(f"[index] Total chunks indexed: {len(chunks)}")

    return data


def build_index() -> None:
    """Legacy entrypoint — indexes only newsletter PDFs in content/Newsletters/."""
    print("[index] Ingesting newsletters…")
    chunks = ingest_all()
    if not chunks:
        print("[index] No chunks found. Make sure PDFs are in content/Newsletters/")
        return
    build_index_from_chunks(chunks, save=True)


if __name__ == "__main__":
    build_index()
