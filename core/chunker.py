"""
core/chunker.py — ALTTRNET shared chunking infrastructure
=========================================================
Implements the FROZEN Step-1 chunking strategy (the exact algorithm used
by ingest.py and validated by the retrieval experiments):

    * ~400 words per chunk
    * ~50 words of overlap between neighbouring chunks
    * word splitting via text.split() (whitespace only — no tokenizer)
    * chunks reconstructed by joining words with a single space
    * deterministic chunk ordering (pure left-to-right sliding window)

Deliberately NO heading-aware, semantic, recursive or library-based
splitting — chunking is frozen for Step 2A and will only change in a
separate future experiment.
"""

CHUNK_SIZE = 400      # approximate words per chunk
CHUNK_OVERLAP = 50    # words of overlap between neighbouring chunks
CHUNK_STEP = CHUNK_SIZE - CHUNK_OVERLAP


def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split text into word-based chunks of `chunk_size` words with
    `chunk_overlap` words of overlap.

    Deterministic: identical input always yields identical chunks.
    Returns [] for empty/whitespace-only input.
    """
    words = (text or "").split()
    if not words:
        return []

    step = chunk_size - chunk_overlap
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += step

    return chunks
