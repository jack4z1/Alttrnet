"""
core/db.py — ALTTRNET shared ChromaDB infrastructure
====================================================
Single access point for the persistent unified ChromaDB knowledge base:

    * opens/creates the persistent ChromaDB under the project's
      chroma_db/ directory
    * accesses the "knowledge_base" collection (cosine space, as the
      validated experiments use)
    * upserts chunk records
    * retrieves the chunks currently stored for one document
    * deletes stale chunks (chunks of a document whose content changed)

All ingestion entry points share this module so the collection name,
space and write semantics cannot drift between scripts.
"""

import sys
from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "knowledge_base"


def get_collection(chroma_dir=CHROMA_DIR, collection_name=COLLECTION_NAME):
    """
    Open (creating if needed) the persistent collection.

    Cosine space matches the embedding model (nomic-embed-text) and the
    validated experiments. The metadata is only applied at creation, so
    re-opening an existing collection is a no-op.
    """
    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        return client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        print(f"ChromaDB initialization failed: {exc}")
        sys.exit(1)


def upsert_chunks(collection, ids, documents, metadatas, embeddings):
    """
    Insert or replace chunk records by ID. Upsert (not add) makes
    re-ingesting a document idempotent: identical chunks are replaced
    in place instead of duplicated.
    """
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def get_document_chunk_ids(collection, doc_id):
    """
    IDs of the chunks currently stored for a document ([] if none).
    Relies on every chunk carrying the document's ID in its `document_id`
    metadata field (the unified metadata model).
    """
    res = collection.get(where={"document_id": doc_id}, include=["metadatas"])
    return res["ids"] if res else []


def delete_chunks(collection, ids):
    """Delete chunk records by ID (no-op for an empty list)."""
    if ids:
        collection.delete(ids=ids)


def remove_stale_chunks(collection, doc_id, current_ids):
    """
    Delete stored chunks of a document whose IDs are no longer current
    (the document's content changed since the last ingestion).
    Returns the list of removed IDs.
    """
    existing = get_document_chunk_ids(collection, doc_id)
    current = set(current_ids)
    stale = [cid for cid in existing if cid not in current]
    if stale:
        delete_chunks(collection, stale)
        print(f"Removed {len(stale)} stale chunk(s) for document {doc_id[:8]}...")
    return stale
