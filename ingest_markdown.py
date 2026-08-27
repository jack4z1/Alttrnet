"""
ALTTRNET — Step 2A: Multi-Source Markdown Knowledge Ingestion
=============================================================
Recursively ingests Markdown documents under a knowledge root into ONE
persistent unified ChromaDB collection ("knowledge_base").

Pipeline per file (strictly text -> chunk -> embed -> store):

    1. normalize relative path (portable, never absolute)
    2. read as UTF-8 text (Markdown treated as DATA: headings, code
       blocks, URLs and terminology are preserved; minimal cleanup)
    3. chunk with the frozen 400/50 word strategy (text.split())
    4. generate nomic-embed-text embeddings via Ollama (precomputed
       vectors — ChromaDB is never given its own embedding function)
    5. delete stale chunks from previous versions of the document
    6. upsert current chunks

Failures are never silent: a failing file is reported and the run
continues with the remaining files.

Markdown is DATA — nothing in the document is ever executed (no
eval/exec, no subprocess, no dynamic import, no interpreter).

Usage:
    python ingest_markdown.py [knowledge_dir]
  (default: ./knowledge relative to this script)

Re-ingesting an unchanged file is idempotent; a shortened file has its
obsolete trailing chunks deleted.
"""

import datetime
import hashlib
import sys
from pathlib import Path

from core.chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_text
from core.db import (
    COLLECTION_NAME,
    delete_chunks,
    get_collection,
    get_document_chunk_ids,
    upsert_chunks,
)
from core.embedder import check_ollama, get_embedding
from core.ids import (
    make_chunk_id,
    make_document_id,
    normalize_doc_path,
    title_from_markdown,
    url_for_file,
)

DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

SOURCE_TYPE_FILE = "file"
LANGUAGE_MARKDOWN = "markdown"


# ---------------------------------------------------------------------------
# Markdown discovery & reading
# ---------------------------------------------------------------------------

def find_markdown_files(knowledge_dir):
    """
    Recursively collect every *.md path under `knowledge_dir`, sorted by
    path for deterministic ingestion order. Non-Markdown files are
    ignored. Files whose name starts with "_" or "." and hidden
    directory segments are skipped.
    """
    files = []
    for path in sorted(knowledge_dir.rglob("*.md")):
        if any(part.startswith(".") for part in path.parts):
            continue
        if path.name.startswith("_"):
            continue
        files.append(path)
    return files


def read_markdown(path):
    """
    Read a Markdown file as untrusted text with minimal cleanup only:
    UTF-8 decode (undecodable bytes replaced), BOM strip, CRLF/CR -> LF.
    Everything else is preserved verbatim.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.strip()


# ---------------------------------------------------------------------------
# Per-file ingestion
# ---------------------------------------------------------------------------

def chunk_metadata(chunk_id, chunk_index, doc_id, file_path, url, title,
                   chunk_text_value, ingested_at):
    """
    Unified FLAT metadata model (ChromaDB key/value pairs). Fields are
    kept for all future source types even though for Markdown
    source_id == document_id == the file.
    """
    return {
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "source_type": SOURCE_TYPE_FILE,   # future: repository, website, ...
        "source_id": doc_id,               # file source: same as document
        "document_id": doc_id,
        "file_path": file_path,            # normalized relative path
        "url": url,                        # file://<relative path>
        "title": title,
        "language": LANGUAGE_MARKDOWN,
        "heading": "",                     # heading-aware chunking excluded in 2A
        "content_hash": hashlib.sha256(chunk_text_value.encode()).hexdigest(),
        "ingested_at": ingested_at,
    }


def ingest_file(collection, path, knowledge_dir, ordinal, total):
    """
    Ingest one Markdown file. Raises on failure so the caller can
    report it and continue with the remaining files.
    """
    file_path = normalize_doc_path(path, knowledge_dir)
    doc_id = make_document_id(file_path)
    url = url_for_file(file_path)

    print(f"[{ordinal}/{total}] {file_path}")
    text = read_markdown(path)
    title = title_from_markdown(text, path.stem)

    words = text.split()
    print(f"    Words: {len(words)}")

    chunks = chunk_text(text)
    if not chunks:
        print("    Chunks: 0 (no words to chunk — skipped)")
        return {"chunks": 0, "created": 0, "updated": 0, "stale": 0}

    print(f"    Chunks: {len(chunks)}")
    chunk_ids = [make_chunk_id(file_path, i) for i in range(len(chunks))]

    ingested_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    metadatas = [
        chunk_metadata(chunk_ids[i], i, doc_id, file_path, url, title,
                       chunks[i], ingested_at)
        for i in range(len(chunks))
    ]

    embeddings = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"    Embedding chunk {i}/{len(chunks)}")
        embedding = get_embedding(chunk)
        if embedding is None:
            raise RuntimeError(f"embedding failed for chunk {i} of {file_path}")
        embeddings.append(embedding)

    # Stale deletion BEFORE upsert (spec order):
    #   generate current IDs -> find existing -> delete stale -> upsert
    existing_ids = get_document_chunk_ids(collection, doc_id)
    existing = set(existing_ids)
    current = set(chunk_ids)
    stale = sorted(existing - current)
    if stale:
        delete_chunks(collection, stale)
        print(f"    Stale removed: {len(stale)}")

    upsert_chunks(collection, ids=chunk_ids, documents=chunks,
                  metadatas=metadatas, embeddings=embeddings)

    created = len(current - existing)
    updated = len(current & existing)
    print(f"    Upserted: {len(chunks)} (created {created}, updated {updated})")
    return {"chunks": len(chunks), "created": created,
            "updated": updated, "stale": len(stale)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    print("=" * 60)
    print("Alttrnet Markdown Ingestion")
    print("=" * 60)
    print(f"Chunking (frozen): {CHUNK_SIZE} words, {CHUNK_OVERLAP} words overlap")
    print()

    check_ollama()
    print()

    knowledge_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_KNOWLEDGE_DIR
    if not knowledge_dir.is_dir():
        print(f"Knowledge root not found: {knowledge_dir}")
        print("Create it and place Markdown (.md) documents inside.")
        sys.exit(1)

    files = find_markdown_files(knowledge_dir)
    print(f"Knowledge root: {knowledge_dir}")
    print(f"Discovered: {len(files)} Markdown files")
    print()
    if not files:
        print("Nothing to ingest.")
        return

    collection = get_collection()
    before = collection.count()

    total_chunks = 0
    total_created = 0
    total_updated = 0
    total_stale = 0
    successful = 0
    failed = 0

    for i, path in enumerate(files, start=1):
        file_path = normalize_doc_path(path, knowledge_dir)
        try:
            result = ingest_file(collection, path, knowledge_dir, i, len(files))
            successful += 1
            total_chunks += result["chunks"]
            total_created += result["created"]
            total_updated += result["updated"]
            total_stale += result["stale"]
        except Exception as exc:
            failed += 1
            print(f"    FAILED: {file_path} — {exc}")
            print("    Continuing with the remaining files.")
        print()

    after = collection.count()

    print("=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Documents processed: {len(files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Chunks upserted: {total_chunks}")
    print(f"  created: {total_created}")
    print(f"  updated: {total_updated}")
    print(f"Stale chunks removed: {total_stale}")
    print(f"Total chunks in '{COLLECTION_NAME}': {before} before -> {after} after")
    print()

    if failed:
        print(f"{failed} document(s) failed. See the FAILED lines above.")
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
