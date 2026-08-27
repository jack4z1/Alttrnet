"""
ALTTRNET — RAG Prototype (milestone 1)
======================================
Terminal-based RAG pipeline:

    WEBSITE URL -> EXTRACT TEXT -> CHUNK TEXT -> CREATE EMBEDDINGS
    -> STORE IN CHROMADB -> USER QUESTION
    -> RETRIEVE TOP 5 CHUNKS (frozen architecture)
    -> QWEN3:14B -> ANSWER

Retrieval now uses the frozen, experimentally-validated architecture
from core.retriever.py (dense top-20 + BM25 top-20 -> union ->
cross-encoder reranker -> top-5). The answering model (qwen3:14b) and
the prompt construction are unchanged.

Dependencies (install once, manually):
    pip install trafilatura chromadb ollama

Required Ollama models (pull once, manually):
    ollama pull qwen3:14b
    ollama pull nomic-embed-text

This script never installs packages or downloads models by itself.
"""

import hashlib
import sys
from pathlib import Path

import ollama
import trafilatura
import chromadb

from core.retriever import Retriever

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path to ChromaDB storage, relative to THIS script (not the working directory).
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"

COLLECTION_NAME = "knowledge_base"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:14b"

CHUNK_SIZE = 400      # approximate words per chunk
CHUNK_OVERLAP = 50    # words of overlap between neighbouring chunks
CHUNK_STEP = CHUNK_SIZE - CHUNK_OVERLAP

TOP_K = 5             # number of chunks to retrieve per question

SYSTEM_PROMPT = (
    "You are a helpful coding and reasoning assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Do not invent information that is not supported by the provided context. "
    "If the answer cannot be determined from the provided context, clearly "
    "state that the provided context does not contain enough information to "
    "answer the question."
)


# ---------------------------------------------------------------------------
# Step 1 — Ollama check
# ---------------------------------------------------------------------------

def check_ollama():
    """Verify Ollama is reachable and the required models are available."""
    try:
        available = ollama.list()
    except Exception:
        print("Ollama is not running. Please start Ollama before running this script.")
        sys.exit(1)

    # Newer ollama clients return a ListResponse object whose models expose
    # a "model" attribute; older ones use dicts with a "name" key.
    models = available.get("models", []) if isinstance(available, dict) else getattr(available, "models", [])
    model_names = [
        getattr(m, "model", None) or (m.get("name", "") if hasattr(m, "get") else "")
        for m in models
    ]

    missing = []
    for required in (EMBED_MODEL, LLM_MODEL):
        # Accept both "qwen3:14b" and "qwen3:14b:latest" style names.
        if not any(name == required or name.startswith(required + ":") for name in model_names):
            missing.append(required)

    if missing:
        print("The following required Ollama model(s) are missing:")
        for model in missing:
            print(f"  - {model}")
        print("Please pull them first, e.g.:")
        print(f"  ollama pull {missing[0]}")
        sys.exit(1)

    print("Ollama is available.")


# ---------------------------------------------------------------------------
# Step 3 — Content extraction
# ---------------------------------------------------------------------------

def extract_content(url):
    """
    Fetch and extract the main textual content of a webpage with trafilatura.

    SECURITY: the returned text is UNTRUSTED DATA from the web. It is only
    ever used as a plain string (embedded, stored, shown to the LLM as text).
    It is NEVER passed to eval(), exec(), subprocess, a shell, or any other
    code-execution mechanism.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception as exc:
        print(f"Failed to reach {url}: {exc}")
        sys.exit(1)

    if downloaded is None:
        print(f"Failed to reach {url}. The site may be unreachable or the URL may be invalid.")
        sys.exit(1)

    text = trafilatura.extract(downloaded)

    if not text:
        print("Text extraction failed: no meaningful content could be extracted from the page.")
        sys.exit(1)

    print(f"Successfully extracted {len(text)} characters.")
    return text


# ---------------------------------------------------------------------------
# Step 4 — Chunking (simple word-based, no tokenizer)
# ---------------------------------------------------------------------------

def chunk_text(text):
    """Split text into ~400-word chunks with ~50-word overlap."""
    words = text.split()

    if not words:
        print("Text extraction failed: the page contained no words.")
        sys.exit(1)

    chunks = []
    start = 0
    while start < len(words):
        chunk_words = words[start:start + CHUNK_SIZE]
        chunks.append(" ".join(chunk_words))
        if start + CHUNK_SIZE >= len(words):
            break
        start += CHUNK_STEP

    print(f"Created {len(chunks)} chunks.")
    return chunks


# ---------------------------------------------------------------------------
# Step 5 / Step 9 — Embeddings (same model for chunks and questions)
# ---------------------------------------------------------------------------

def get_embedding(text):
    """Get a nomic-embed-text embedding vector from Ollama for the given text."""
    try:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    except Exception as exc:
        print(f"Embedding failed: {exc}")
        return None
    return response["embedding"]


# ---------------------------------------------------------------------------
# Step 6 / Step 7 — ChromaDB storage
# ---------------------------------------------------------------------------

def get_collection():
    """Open the persistent ChromaDB collection (creating it if needed)."""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        # Cosine distance suits nomic-embed-text vectors.
        return client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        print(f"ChromaDB initialization failed: {exc}")
        sys.exit(1)


def embed_chunks(url, chunks):
    """
    Embed every chunk with nomic-embed-text and prepare the records for storage.

    IDs are deterministic MD5 hashes of "source_url_chunk_index".
    MD5 is used ONLY as a deterministic ID generator, never for security
    or cryptographic integrity.
    """
    ids = []
    documents = []
    metadatas = []
    embeddings = []

    total = len(chunks)
    for i, chunk in enumerate(chunks, start=1):
        print(f"Embedding chunk {i} of {total}")

        embedding = get_embedding(chunk)
        if embedding is None:
            sys.exit(1)

        chunk_index = i - 1
        chunk_id = hashlib.md5(f"{url}_{chunk_index}".encode()).hexdigest()

        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({"source": url, "chunk_index": chunk_index})
        embeddings.append(embedding)

    return ids, documents, metadatas, embeddings


def store_chunks(collection, url, chunk_data):
    """Upsert prepared chunk records into ChromaDB."""
    ids, documents, metadatas, embeddings = chunk_data
    try:
        # upsert (not add): re-ingesting the same URL replaces existing
        # chunks instead of duplicating them or crashing on existing IDs.
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
    except Exception as exc:
        print(f"Failed to store chunks in ChromaDB: {exc}")
        sys.exit(1)

    print(f"Stored {len(documents)} chunks from [{url}]")
    print(f"Total chunks in knowledge_base: {collection.count()}")


# ---------------------------------------------------------------------------
# Step 10 — Retrieval
# ---------------------------------------------------------------------------

def retrieve_chunks(collection, question_embedding, top_k=TOP_K):
    """Retrieve the top-k most relevant chunks for a question embedding."""
    try:
        # Don't ask for more results than the collection actually holds.
        n = min(top_k, collection.count())
        if n == 0:
            print("The knowledge base is empty. Please ingest a URL first.")
            return None
        return collection.query(
            query_embeddings=[question_embedding],
            n_results=n,
        )
    except Exception as exc:
        print(f"ChromaDB retrieval failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 11 — Qwen3:14B answer
# ---------------------------------------------------------------------------

def ask_qwen(question, chunks, metadatas):
    """Send the retrieved chunks plus the question to qwen3:14b and print its answer."""
    context_parts = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas), start=1):
        source = meta.get("source", "unknown")
        context_parts.append(f"[Retrieved Chunk {i}]\nSource: {source}\n{chunk}")

    context = "\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n\n{context}\n\nUSER QUESTION:\n{question}"},
    ]

    try:
        # options={"think": False} disables Qwen3 extended thinking for this prototype.
        response = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            options={"think": False},
        )
    except Exception as exc:
        print(f"Failed to get an answer from {LLM_MODEL}: {exc}")
        return

    answer = response["message"]["content"]
    print("\n--- Qwen3:14B Answer ---")
    print(answer)
    print("------------------------")


# ---------------------------------------------------------------------------
# Step 8 / Step 12 — Interactive question loop
# ---------------------------------------------------------------------------

def question_loop(collection):
    """Ask multiple questions in a loop, retrieving context for each one."""
    retriever = Retriever(collection=collection)
    while True:
        try:
            question = input("Ask a question (or type 'exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        # Retrieve through the frozen architecture
        # (dense top-20 + BM25 top-20 -> union -> cross-encoder -> top-5).
        results = retriever.search(question)
        if not results:
            continue

        chunks = [r["document"] for r in results]
        metadatas = [r["metadata"] for r in results]

        print(f"\nRetrieved {len(chunks)} relevant chunks:")
        for i, (chunk, meta) in enumerate(zip(chunks, metadatas), start=1):
            source = meta.get("source", "unknown")
            preview = chunk[:100].replace("\n", " ")
            print(f"[{i}]")
            print(f"Source: {source}")
            print(f"Preview: {preview}...")
            print()

        ask_qwen(question, chunks, metadatas)
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Windows consoles often default to cp1252, which cannot print the full
    # range of unicode characters found in web content. Use UTF-8 instead.
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    print("=" * 40)
    print("ALTTRNET - RAG PROTOTYPE")
    print("=" * 40)

    # Step: URL input — plain sys.argv, no argparse.
    if len(sys.argv) < 2 or not sys.argv[1]:
        print("Usage:")
        print('python ingest.py "https://example.com/article"')
        sys.exit(1)

    url = sys.argv[1]
    if not (url.startswith("http://") or url.startswith("https://")):
        print(f"Invalid URL: {url}")
        print("The URL must start with http:// or https://")
        sys.exit(1)

    print("\nSource:")
    print(url)
    print()

    print("[1/5] Checking Ollama...")
    check_ollama()
    print()

    print("[2/5] Extracting content...")
    text = extract_content(url)
    print()

    print("[3/5] Creating chunks...")
    chunks = chunk_text(text)
    print()

    print("[4/5] Generating embeddings...")
    chunk_data = embed_chunks(url, chunks)
    print()

    print("[5/5] Storing in ChromaDB...")
    collection = get_collection()
    store_chunks(collection, url, chunk_data)
    print()

    print("Ready for questions.")
    print()

    question_loop(collection)

    print("\nGoodbye.")


if __name__ == "__main__":
    main()
