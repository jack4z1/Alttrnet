"""
ALTTRNET — Step 1B.3: Expanded Retrieval Evaluation Baseline
============================================================
Objective measurement of the CURRENT retrieval quality against the
restored 400/50 knowledge base. This is the stronger baseline used
before any retrieval-system improvements are tested.

Pipeline per question:

    QUESTION
      -> nomic-embed-text embedding (the SAME model ingest.py uses)
      -> ChromaDB query for the top-5 chunks in `knowledge_base`
      -> inspect each retrieved chunk's `source` metadata
      -> Precision@5 = relevant retrieved chunks / 5
      -> Hit@1       = 1 if the top-ranked chunk matches, else 0
      -> RR          = 1 / rank of the first relevant chunk (0 if none)
                        in the top 5
      -> MRR         = mean of RR across all questions

Design constraints (from the Step 1B.3 spec):
  * Qwen is NEVER used to judge retrieval. Judgement is based purely on
    chunk metadata, so a good pretrained model cannot mask bad retrieval.
  * No BM25, hybrid retrieval, RRF, reranking, cross-encoder, query
    expansion, semantic chunking, new embeddings, metadata/source
    filtering, or any other retrieval change.
  * ingest.py is reused (imported) for its constants and helpers but is
    never modified by this harness.

Dataset: 15 questions — 5 per source (Python, RAG, Ollama) — each
verified to be answerable from the currently indexed chunk content.
"""

import sys

import ollama

from ingest import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBED_MODEL,
    TOP_K,
    get_collection,
    get_embedding,
    retrieve_chunks,
)


# ---------------------------------------------------------------------------
# Evaluation dataset — 15 questions, 5 per source, mixed question types.
# Every question was checked against the actual indexed chunk text before
# finalizing, so none refers to content the knowledge base lacks.
# ---------------------------------------------------------------------------

QUESTIONS = [
    # --- Python documentation — tutorial control flow chapter (1-5) ---
    {
        "source": "Python",
        "type": "Broad",
        "question": "What topics does the Python tutorial chapter on control flow cover?",
        "expected_source": "https://docs.python.org/3/tutorial/controlflow.html",
        "expected_label": "Python documentation",
    },
    {
        "source": "Python",
        "type": "Specific factual",
        "question": "According to the tutorial, when does the else clause of a for or while loop execute?",
        "expected_source": "https://docs.python.org/3/tutorial/controlflow.html",
        "expected_label": "Python documentation",
    },
    {
        "source": "Python",
        "type": "Keyword-heavy",
        "question": "How are the break, continue, and pass statements described in the Python tutorial?",
        "expected_source": "https://docs.python.org/3/tutorial/controlflow.html",
        "expected_label": "Python documentation",
    },
    {
        "source": "Python",
        "type": "Conceptual",
        "question": "Why does the tutorial say the object returned by range() is iterable rather than a list?",
        "expected_source": "https://docs.python.org/3/tutorial/controlflow.html",
        "expected_label": "Python documentation",
    },
    {
        "source": "Python",
        "type": "Specific",
        "question": "What does the tutorial say about default argument values in function definitions?",
        "expected_source": "https://docs.python.org/3/tutorial/controlflow.html",
        "expected_label": "Python documentation",
    },
    # --- RAG Wikipedia (6-10) ---
    {
        "source": "RAG",
        "type": "Broad",
        "question": "What is retrieval-augmented generation?",
        "expected_source": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "expected_label": "RAG Wikipedia article",
    },
    {
        "source": "RAG",
        "type": "Specific factual",
        "question": "When was retrieval-augmented generation first proposed?",
        "expected_source": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "expected_label": "RAG Wikipedia article",
    },
    {
        "source": "RAG",
        "type": "Keyword-heavy",
        "question": "What are retrieval, augmentation, and generation in a RAG system?",
        "expected_source": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "expected_label": "RAG Wikipedia article",
    },
    {
        "source": "RAG",
        "type": "Conceptual",
        "question": "How does RAG help reduce hallucinations?",
        "expected_source": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "expected_label": "RAG Wikipedia article",
    },
    {
        "source": "RAG",
        "type": "Specific",
        "question": "What limitations of RAG are described in the source?",
        "expected_source": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "expected_label": "RAG Wikipedia article",
    },
    # --- Ollama FAQ documentation (11-15) ---
    {
        "source": "Ollama",
        "type": "Broad",
        "question": "What does the Ollama FAQ documentation describe about installing, updating, and configuring Ollama?",
        "expected_source": "https://github.com/ollama/ollama/blob/main/docs/faq.mdx",
        "expected_label": "Ollama FAQ documentation",
    },
    {
        "source": "Ollama",
        "type": "Specific factual",
        "question": "What is the default context window size Ollama uses, and how can it be overridden?",
        "expected_source": "https://github.com/ollama/ollama/blob/main/docs/faq.mdx",
        "expected_label": "Ollama FAQ documentation",
    },
    {
        "source": "Ollama",
        "type": "Keyword-heavy",
        "question": "What do the OLLAMA_HOST, OLLAMA_MODELS, and OLLAMA_ORIGINS environment variables control?",
        "expected_source": "https://github.com/ollama/ollama/blob/main/docs/faq.mdx",
        "expected_label": "Ollama FAQ documentation",
    },
    {
        "source": "Ollama",
        "type": "Conceptual",
        "question": "According to the FAQ, how does Ollama decide where to load a model when GPU inference is available?",
        "expected_source": "https://github.com/ollama/ollama/blob/main/docs/faq.mdx",
        "expected_label": "Ollama FAQ documentation",
    },
    {
        "source": "Ollama",
        "type": "Specific",
        "question": "What does the FAQ say about the keep_alive parameter and how long models stay in memory?",
        "expected_source": "https://github.com/ollama/ollama/blob/main/docs/faq.mdx",
        "expected_label": "Ollama FAQ documentation",
    },
]

SOURCE_ORDER = ["Python", "RAG", "Ollama"]


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
    relevant.
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
    # Known expected sources get their short name.
    for item in QUESTIONS:
        if source_matches(url, item["expected_source"]):
            return item["source"]
    # Fallback for sources outside the evaluation dataset.
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
# Metrics
# ---------------------------------------------------------------------------

def reciprocal_rank(metadatas, expected_source):
    """
    RR for one question: 1 / rank of the first chunk whose source matches
    the expected source, or 0 if none of the top-5 chunks match.
    """
    for rank, meta in enumerate(metadatas, start=1):
        if source_matches(meta.get("source", "unknown"), expected_source):
            return 1.0 / rank
    return 0.0


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

    print("Configuration:")
    print(f"Chunk size: {CHUNK_SIZE} words")
    print(f"Overlap: {CHUNK_OVERLAP} words")
    print(f"Embedding: {EMBED_MODEL}")
    print(f"Retrieval: dense vector")
    print(f"Top-K: {TOP_K}")
    print()
    print("Knowledge base:")
    print(f"{collection.count()} chunks")
    print(f"{len(set(m['source'] for m in collection.get(include=['metadatas'])['metadatas']))} sources")
    print()

    all_metrics = []          # (precision, hit, rr) per question
    per_source = {src: [] for src in SOURCE_ORDER}

    for i, item in enumerate(QUESTIONS, start=1):
        question = item["question"]
        expected = item["expected_source"]

        print("=" * 40)
        print(f"QUESTION {i} — {item['source']}")
        print(f"Type: {item['type']}")
        print()
        print("Expected source:")
        print(item["expected_label"])
        print(f"({expected})")
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

        print("Retrieved:")
        relevant = 0
        for rank, meta in enumerate(metadatas, start=1):
            source = meta.get("source", "unknown")
            is_relevant = source_matches(source, expected)
            if is_relevant:
                relevant += 1
            mark = "✅" if is_relevant else "❌"
            print(f"{rank}. {display_label(source):<14}{mark}  {source}")

        retrieved = len(metadatas)
        precision = relevant / retrieved if retrieved else 0.0
        top_source = metadatas[0].get("source", "unknown") if metadatas else ""
        hit = 1.0 if metadatas and source_matches(top_source, expected) else 0.0
        rr = reciprocal_rank(metadatas, expected)

        all_metrics.append((precision, hit, rr))
        per_source[item["source"]].append((precision, hit, rr))

        print()
        print(f"Precision@5: {fmt_pct(precision)}%")
        print(f"Hit@1: {fmt_pct(hit)}%")
        print(f"Reciprocal Rank: {rr:.4f}")
        print()

    print("=" * 40)
    print("OVERALL")
    print("=" * 40)
    print()

    if all_metrics:
        n = len(all_metrics)
        avg_precision = sum(p for p, _, _ in all_metrics) / n
        avg_hit = sum(h for _, h, _ in all_metrics) / n
        mrr = sum(rr for _, _, rr in all_metrics) / n

        print(f"Average Precision@5: {fmt_pct(avg_precision)}%")
        print(f"Average Hit@1: {fmt_pct(avg_hit)}%")
        print(f"MRR: {mrr:.4f}")
        print()

        print("Per-source averages:")
        for src in SOURCE_ORDER:
            metrics = per_source[src]
            if not metrics:
                continue
            m = len(metrics)
            sp = sum(p for p, _, _ in metrics) / m
            sh = sum(h for _, h, _ in metrics) / m
            smrr = sum(rr for _, _, rr in metrics) / m
            print(f"{src}:")
            print(f"  Precision@5: {fmt_pct(sp)}%")
            print(f"  Hit@1: {fmt_pct(sh)}%")
            print(f"  MRR: {smrr:.4f}")
    else:
        print("No questions could be evaluated.")

    print()
    print("=" * 40)


if __name__ == "__main__":
    main()
