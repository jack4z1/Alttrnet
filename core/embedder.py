"""
core/embedder.py — ALTTRNET shared embedding infrastructure
===========================================================
Single reusable interface to the FROZEN embedding model:

    nomic-embed-text (served by Ollama)

The same model and the same call style used by ingest.py and by every
validated retrieval experiment live here, so all ingestion entry points
embed identically. Ollama errors are surfaced clearly instead of
failing silently.
"""

import sys

import ollama

EMBED_MODEL = "nomic-embed-text"


def check_ollama():
    """
    Verify Ollama is reachable and nomic-embed-text is available.
    Exits with a clear message otherwise.
    """
    try:
        available = ollama.list()
    except Exception:
        print("Ollama is not running. Start Ollama before embedding.")
        sys.exit(1)

    models = available.get("models", []) if isinstance(available, dict) else getattr(available, "models", [])
    model_names = [
        getattr(m, "model", None) or (m.get("name", "") if hasattr(m, "get") else "")
        for m in models
    ]

    if not any(n == EMBED_MODEL or n.startswith(EMBED_MODEL + ":") for n in model_names):
        print(f"The required embedding model {EMBED_MODEL!r} is missing.")
        print(f"Pull it first: ollama pull {EMBED_MODEL}")
        sys.exit(1)

    print("Ollama is available.")


def get_embedding(text):
    """
    Return the nomic-embed-text embedding vector for `text`.

    Returns None on failure (the caller decides whether to abort or skip)
    after printing the underlying Ollama error.
    """
    try:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    except Exception as exc:
        print(f"Embedding failed: {exc}")
        return None
    return response["embedding"]
