"""
Module 3 — ingestion, embedding and retrieval.

Stage 1 (ingestion): read the 8 policy documents from docs/ and chunk them.
Stage 2 (embedding):  embed every chunk locally with sentence-transformers
                      all-MiniLM-L6-v2 (open source, runs on your machine, no
                      account and no API key).
Stage 3 (retrieval):  store the vectors in a persistent ChromaDB collection and
                      expose top-k cosine-similarity search over it.

Nothing in this file depends on MOCK_LLM: embedding and retrieval always run
for real, in both modes, because neither needs a network call to an LLM
provider.

Run directly to (re)build the index:  python ingest.py
"""

from __future__ import annotations

import glob
import os
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(HERE, "docs")
CHROMA_DIR = os.path.join(HERE, "chroma_store")
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Documents are short (a single policy paragraph each), so a per-document chunk
# with a generous character cap keeps each policy semantically whole. Longer
# documents fall back to fixed-size chunking with overlap.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def chunk_text(text: str, size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Per-document chunk when short; fixed-size sliding window when not."""
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        start += size - overlap
    return [c for c in chunks if c]


def load_documents() -> list[tuple[str, str]]:
    """Return [(doc_id, text)] for every .txt file in docs/, sorted by name."""
    paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"no .txt documents found in {DOCS_DIR}")
    return [
        (os.path.splitext(os.path.basename(p))[0],
         open(p, encoding="utf-8").read())
        for p in paths
    ]


@lru_cache(maxsize=1)
def get_embedding_function():
    """all-MiniLM-L6-v2, run locally. Downloads the model weights once."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )


@lru_cache(maxsize=1)
def get_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def build_index(force: bool = False):
    """
    Create (or reuse) the ChromaDB collection holding one embedded vector per
    chunk. Cosine similarity is set explicitly via hnsw:space.
    """
    client = get_client()
    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0 and not force:
        return collection

    ids, documents, metadatas = [], [], []
    for doc_id, text in load_documents():
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{doc_id}::chunk_{i}")
            documents.append(chunk)
            metadatas.append({"source": doc_id, "chunk_index": i})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"[ingest] indexed {len(ids)} chunks from "
          f"{len(set(m['source'] for m in metadatas))} documents")
    return collection


@lru_cache(maxsize=1)
def get_collection():
    """Collection handle, building the index on first use."""
    return build_index()


def retrieve(query: str, k: int = 3) -> list[dict]:
    """
    Top-k chunks by cosine similarity. Always runs for real — no LLM involved.

    Returns [{id, text, source, distance, similarity}] ordered best-first.
    """
    collection = get_collection()
    result = collection.query(
        query_texts=[query],
        n_results=min(k, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for cid, text, meta, dist in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append(
            {
                "id": cid,
                "text": text,
                "source": meta.get("source", ""),
                "distance": float(dist),
                "similarity": round(1.0 - float(dist), 4),
            }
        )
    return hits


if __name__ == "__main__":
    collection = build_index(force=True)
    print(f"[ingest] collection '{COLLECTION_NAME}' now holds "
          f"{collection.count()} chunks at {CHROMA_DIR}")
    for q in ["How long do I have to return a spoiled item?",
              "What does Zepto Pass+ cost?"]:
        print(f"\n[query] {q}")
        for hit in retrieve(q):
            print(f"   {hit['id']:<22} sim={hit['similarity']:.4f}  "
                  f"{hit['text'][:80]}...")
