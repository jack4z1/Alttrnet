"""
ALTTRNET — Step 1B, Part 1: Retrieval Evaluation Harness
========================================================
Objective baseline of the CURRENT retrieval quality, measured before
any retrieval-system changes are made.

Pipeline per question:

    QUESTION
      -> nomic-embed-text embedding (the SAME model ingest.py uses)
      -> ChromaDB query for the top-5 chunks in `knowledge_base`
      -> inspect each retrieved chunk's `source` metadata
      -> Precision@5 = relevant retrieved chunks / 5
      -> Hit@1       = 1 if the top-ranked chunk matches, else 0

Design constraints (from the Step 1B spec):
  * Qwen is NEVER used to judge retrieval. Judgement is based purely on
    chunk metadata, so a good pretrained model cannot mask bad retrieval
    (BAD RETRIEVAL + GOOD PRETRAINED MODEL = MISLEADINGLY GOOD ANSWER).
  * No BM25, reranking, hybrid search, new embedding models, new
    chunking, GUI, web interface, agents, or frameworks.
  * ingest.py is reused (imported) for its constants and helpers but is
    never modified.
"""

import sys

import ollama

from ingest import (
    COLLECTION_NAME,
    EMBED_MODEL,
    TOP_K,
    get_collection,
    get_embedding,
    retrieve_chunks,
)


# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------

QUESTIONS = [
    {
        "question": "What is Python and what are some of its main features?",
        "expected_source": "https://docs.python.org/3/tutorial/",
        "expected_label": "Python documentation",
        "short_label": "Python",
    },
    {
        "question": "What is retrieval-augmented generation?",
        "expected_source": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "expected_label": "RAG Wikipedia article",
        "short_label": "RAG",
    },
    {
        "question": "What is Ollama?",
        "expected_source": "https://en.wikipedia.org/wiki/Ollama",
        "expected_label": "Ollama Wikipedia article",
        "short_label": "Ollama",
    },
]


# ---------------------------------------------------------------------------
# Source matching (metadata-only judgement)
# ---------------------------------------------------------------------------

def normalize_url(url):
    """Trim and strip trailing slashes so URLs compare reliably."""
    return (url or "").strip().rstrip("/")


def source_matches(retrieved_source, expected_source):
    """
    True when a retrieved chunk's `source` metadata belongs to the
    expected source. Handles exact matches and prefix matches, so a
    chunk from any sub-page of the expected page/directory counts as
    relevant (e.g. https://docs.python.org/3/tutorial/introduction.html
    matches https://docs.python.org/3/tutorial/).
    """
    got = normalize_url(retrieved_source)
    want = normalize_url(expected_source)
    if not got or not want:
        return False
    if got == want:
        return True
    return got.startswith(want + "/") or want.startswith(got + "/")


def display_label(source_url):
    """Short human-readable label for a retrieved chunk's source URL."""
    url = (source_url or "").strip()
    if not url:
        return "unknown"
    for item in QUESTIONS:
        if source_matches(url, item["expected_source"]):
            return item["short_label"]
    # Fallback for sources outside the evaluation dataset:
    # hostname, plus the last non-empty path segment if there is one.
    rest = url.split("://", 1)[-1]
    host, _, path = rest.partition("/")
    segments = [s for s in path.split("/") if s]
    last = segments[-1].replace("_", " ").replace("-", " ") if segments else ""
    return f"{host} / {last}".strip() if last else host


def fmt_pct(fraction):
    """Format a fraction as a percentage string, without trailing .0."""
    pct = fraction * 100
    if abs(pct - round(pct)) < 1e-9:
        return str(int(round(pct)))
    return f"{pct:.1f}"


# ---------------------------------------------------------------------------
# Ollama check (embedding model only — qwen3 is NOT needed for evaluation)
# ---------------------------------------------------------------------------

def check_embed_model():
    """Verify Ollama is reachable and nomic-embed-text is available."""
    try:
        available = ollama.list()
    except Exception:
        print("Ollama is not running. Start Ollama before running this script.")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # UTF-8 so the ✅/❌ marks and any unicode in URLs print correctly on Windows.
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    print("=" * 40)
    print("ALTTRNET RETRIEVAL BASELINE")
    print("=" * 40)
    print()

    check_embed_model()
    collection = get_collection()

    print("Config:")
    print(f"  Collection:       {COLLECTION_NAME} ({collection.count()} chunks)")
    print(f"  Embedding model:  {EMBED_MODEL}")
    print(f"  Retrieval:        top-{TOP_K} dense vector similarity")
    print(f"  Judgement:        chunk metadata only (no LLM)")
    print()

    per_question = []

    for i, item in enumerate(QUESTIONS, start=1):
        question = item["question"]
        expected = item["expected_source"]

        if i > 1:
            print("-" * 40)
            print()

        print(f"Question {i}:")
        print(question)
        print()
        print("Expected source:")
        print(expected)
        print()
        print("Expected source relevance:")
        print(item["expected_label"])
        print()

        question_embedding = get_embedding(question)
        if question_embedding is None:
            print("ERROR: failed to embed the question. Skipping.")
            continue

        result = retrieve_chunks(collection, question_embedding)
        if result is None:
            print("ERROR: retrieval failed. Skipping.")
            continue

        metadatas = result.get("metadatas", [[]])[0]

        print("Retrieved sources:")
        relevant = 0
        for rank, meta in enumerate(metadatas, start=1):
            source = meta.get("source", "unknown")
            is_relevant = source_matches(source, expected)
            if is_relevant:
                relevant += 1
            mark = "✅" if is_relevant else "❌"
            print(f"{rank}. {display_label(source):<18}{mark}  {source}")

        retrieved = len(metadatas)
        precision = relevant / retrieved if retrieved else 0.0
        top_source = metadatas[0].get("source", "unknown") if metadatas else ""
        hit = 1.0 if metadatas and source_matches(top_source, expected) else 0.0

        per_question.append((precision, hit))

        print()
        print(f"Precision@5: {fmt_pct(precision)}%")
        print(f"Hit@1:       {fmt_pct(hit)}%")
        print()

    print("=" * 40)
    print("OVERALL")
    print("=" * 40)
    print()

    if per_question:
        avg_precision = sum(p for p, _ in per_question) / len(per_question)
        avg_hit = sum(h for _, h in per_question) / len(per_question)
        print(f"Average Precision@5: {fmt_pct(avg_precision)}%")
        print(f"Average Hit@1:       {fmt_pct(avg_hit)}%")
        print()
        print(f"({len(per_question)} question(s) evaluated)")
    else:
        print("No questions could be evaluated.")

    print()
    print("=" * 40)


if __name__ == "__main__":
    main()
