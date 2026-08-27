"""
ALTTRNET — Step 1B.4: Retrieval Failure Mode Diagnosis
======================================================
Diagnostic companion to eval_retrieval.py. It does NOT change any
retrieval behavior: it re-runs the exact same retrieval the system
performs (nomic-embed-text question embedding -> ChromaDB top-5 dense
query) and then inspects WHY failing questions fail.

Deliberately NO LLM is used anywhere:
  * relevance is judged purely by source metadata (as in the harness)
  * lexical overlap is a deterministic token comparison
  * "semantic" evidence is the existing nomic-embed-text cosine
    similarity between the question and stored chunk embeddings
    (read from the database, no new model, no re-embedding)

It reuses the evaluation dataset (QUESTIONS) from eval_retrieval.py so
the two scripts cannot drift apart.

Categories used for classification of each wrong retrieved chunk:
  A. Lexically similar      -> distinctive query tokens appear in the chunk
  B. Semantically similar   -> chunk embedding is at least as close to the
                               query as the best expected-source chunk
  C. Both                   -> A and B
  D. Neither                -> low overlap AND lower similarity than the
                               best expected-source chunk (rank filler)
  UNCERTAIN                 -> evidence is ambiguous (documented rule)
"""

import re
import sys

import numpy as np
import ollama

from eval_retrieval import QUESTIONS, SOURCE_ORDER
from ingest import (
    COLLECTION_NAME,
    EMBED_MODEL,
    TOP_K,
    get_collection,
    get_embedding,
)

# ---------------------------------------------------------------------------
# Lexical analysis helpers (deterministic, no NLP framework)
# ---------------------------------------------------------------------------

STOPWORDS = set("""
the a an and or but if of to in on for with at by from is are was were be been
what when why how does do did according source says say said about as it its that
this these those which who whom can could will would should may might not no yes so
than then there here their them they we you your our us me my into out over under up
down again further once also only own same such very just dont don't s t per more
than via where while etc
""".split())

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokens(text):
    """Lowercased, punctuation-stripped tokens."""
    return TOKEN_RE.findall((text or "").lower())


def distinctive_tokens(text):
    """Query tokens with common stop words removed."""
    return [t for t in tokens(text) if t not in STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Similarity helpers (read-only use of the stored nomic-embed-text vectors)
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def classify_chunk(sim_wrong, sim_best_expected, lex_score):
    """
    Deterministic classification of a wrong retrieved chunk.

    lex_score    = fraction of distinctive query tokens found in the chunk
    sim_wrong    = cosine similarity(question, wrong chunk)   [nomic-embed-text]
    sim_best_expected = cosine similarity(question, best expected-source chunk)

    Rules:
      lex_sim  = lex_score >= 0.30
      sem_sim  = sim_wrong >= sim_best_expected
      if lex_sim and sem_sim:                       -> Both
      if lex_sim:                                   -> Lexically similar
      if sem_sim:                                   -> Semantically similar
      else:                                         -> Neither (rank filler)
      UNCERTAIN if 0.15 <= lex_score < 0.30 and
                   |sim_wrong - sim_best_expected| < 0.02
    """
    lex_sim = lex_score >= 0.30
    sem_sim = sim_wrong >= sim_best_expected
    ambiguous = (0.15 <= lex_score < 0.30) and abs(sim_wrong - sim_best_expected) < 0.02
    if ambiguous:
        return "UNCERTAIN"
    if lex_sim and sem_sim:
        return "Both (lexical + semantic)"
    if lex_sim:
        return "Lexically similar"
    if sem_sim:
        return "Semantically similar"
    return "Neither (low-similarity rank filler)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    print("=" * 70)
    print("ALTTRNET — STEP 1B.4 RETRIEVAL FAILURE MODE DIAGNOSIS")
    print("=" * 70)
    print()

    try:
        available = ollama.list()
    except Exception:
        print("Ollama is not running. Start Ollama first.")
        sys.exit(1)
    models = available.get("models", []) if isinstance(available, dict) else getattr(available, "models", [])
    names = [getattr(m, "model", None) or (m.get("name", "") if hasattr(m, "get") else "") for m in models]
    if not any(n == EMBED_MODEL or n.startswith(EMBED_MODEL + ":") for n in names):
        print(f"Missing embedding model {EMBED_MODEL!r}.")
        sys.exit(1)

    collection = get_collection()
    data = collection.get(include=["metadatas", "documents", "embeddings"])

    ids = data["ids"]
    metas = data["metadatas"]
    docs = data["documents"]
    embs = data["embeddings"]
    total = len(ids)

    # source -> [(chunk_index, doc, emb)]
    by_source = {}
    for cid, meta, doc, emb in zip(ids, metas, docs, embs):
        src = meta["source"]
        by_source.setdefault(src, []).append((meta["chunk_index"], doc, emb))
    for src in by_source:
        by_source[src].sort(key=lambda t: t[0])

    def preview(text, n=300):
        return (text or "").replace("\n", " ").strip()[:n]

    print(f"Configuration: {EMBED_MODEL} | ChromaDB dense top-{TOP_K} | "
          f"{total} chunks | {len(by_source)} sources")
    for src in sorted(by_source):
        print(f"  {len(by_source[src]):>3} chunks  {src}")
    print()

    # ------------------------------------------------------------------
    # PART 1 — run all 15 questions exactly as the harness does
    # ------------------------------------------------------------------
    rows = []
    for qnum, item in enumerate(QUESTIONS, start=1):
        qtext = item["question"]
        expected = item["expected_source"]

        qemb = get_embedding(qtext)
        if qemb is None:
            print(f"Q{qnum}: embedding failed, skipping")
            continue

        res = collection.query(
            query_embeddings=[qemb],
            n_results=min(TOP_K, total),
        )
        r_metas = res["metadatas"][0]
        r_docs = res["documents"][0]
        r_dist = res.get("distances", [[]])[0] or []

        hits = [m["source"] for m in r_metas]
        relevant = sum(1 for s in hits if s == expected)
        retrieved_n = len(hits)
        precision = relevant / retrieved_n if retrieved_n else 0.0
        hit1 = 1.0 if hits and hits[0] == expected else 0.0
        rr = 0.0
        for rank, s in enumerate(hits, start=1):
            if s == expected:
                rr = 1.0 / rank
                break

        # ChromaDB cosine distance -> similarity (1 - distance)
        sims_retrieved = [1.0 - d for d in r_dist] if r_dist else [None] * retrieved_n

        # Diagnostic: full-corpus similarity ranking from stored embeddings
        all_sims = [cosine_similarity(qemb, e) for e in embs]
        order = sorted(range(total), key=lambda i: -all_sims[i])
        exp_idx = [i for i in range(total) if metas[i]["source"] == expected]
        best_exp_rank = None
        best_exp_sim = None
        exp_ranks = []
        for rank, i in enumerate(order, start=1):
            if i in exp_idx:
                exp_ranks.append((rank, metas[i]["chunk_index"], all_sims[i]))
                if best_exp_rank is None:
                    best_exp_rank = rank
                    best_exp_sim = all_sims[i]

        rows.append({
            "qnum": qnum, "item": item, "hits": hits,
            "r_metas": r_metas, "r_docs": r_docs, "sims": sims_retrieved,
            "precision": precision, "hit1": hit1, "rr": rr,
            "best_exp_rank": best_exp_rank, "best_exp_sim": best_exp_sim,
            "exp_ranks": exp_ranks, "expected": expected,
        })

    # PART 1 table
    print("-" * 70)
    print("PART 1 — ALL 15 QUESTIONS (source-level, as evaluated)")
    print("-" * 70)
    print(f"{'Q':>2} {'Type':<16} {'R1':<10} {'R2':<10} {'R3':<10} {'R4':<10} {'R5':<10} {'P@5':>5} {'Hit@1':>6} {'RR':>6}")
    for r in rows:
        short = [s.split("/")[-1].split(".")[-1][:9] for s in r["hits"][:5]]
        short += [""] * (5 - len(short))
        print(f"{r['qnum']:>2} {r['item']['type']:<16} " + " ".join(f"{s:<10}" for s in short) +
              f" {r['precision']*100:>4.0f}% {r['hit1']*100:>5.0f}% {r['rr']:>5.2f}")

    failing = [r for r in rows if r["precision"] < 1.0 or r["hit1"] == 0.0]
    print()
    print(f"FAILING QUESTIONS ({len(failing)}): " + ", ".join(f"Q{r['qnum']}" for r in failing))
    print()

    # ------------------------------------------------------------------
    # PART 2-4 — per failing question: chunks, lexical, semantic
    # ------------------------------------------------------------------
    for r in failing:
        qnum = r["qnum"]
        item = r["item"]
        expected = r["expected"]
        dq = distinctive_tokens(item["question"])

        print("=" * 70)
        print(f"FAILING QUESTION {qnum}")
        print("-" * 70)
        print(f"Question: {item['question']}")
        print(f"Type: {item['type']}")
        print(f"Expected source: {expected}")
        print(f"Expected label: {item['expected_label']}")
        print(f"Precision@5: {r['precision']*100:.0f}%  Hit@1: {r['hit1']*100:.0f}%  RR: {r['rr']:.4f}")
        print()
        print(f"Distinctive query tokens (stop words removed): {dq}")
        print()
        print("RETRIEVED TOP-5 (as the system retrieves them):")
        for rank, (meta, doc, sim) in enumerate(
            zip(r["r_metas"], r["r_docs"], r["sims"]), start=1
        ):
            ok = "✅" if meta["source"] == expected else "❌"
            sim_s = f"{sim:.4f}" if sim is not None else " n/a "
            print(f"  {rank}. {ok} {meta['source']}")
            print(f"      chunk_index={meta['chunk_index']}  sim={sim_s}  "
                  f"preview: {preview(doc)}...")
        print()
        print(f"DIAGNOSTIC — full-corpus ranking of expected-source chunks "
              f"({len(r['exp_ranks'])} chunks):")
        for rank, ci, sim in r["exp_ranks"]:
            print(f"  chunk {ci}: rank {rank}/{total}, sim {sim:.4f}")
        print(f"  -> best expected chunk rank {r['best_exp_rank']}/{total}, "
              f"sim {r['best_exp_sim']:.4f}")
        print()
        print("WRONG CHUNKS — LEXICAL OVERLAP & SEMANTIC CLASSIFICATION:")
        for rank, (meta, doc, sim) in enumerate(
            zip(r["r_metas"], r["r_docs"], r["sims"]), start=1
        ):
            if meta["source"] == expected:
                continue
            chunk_toks = set(tokens(doc))
            found = [t for t in dq if t in chunk_toks]
            missing = [t for t in dq if t not in chunk_toks]
            lex_score = len(found) / len(dq) if dq else 0.0
            classification = classify_chunk(
                sim if sim is not None else 0.0,
                r["best_exp_sim"] or 0.0,
                lex_score,
            )
            print(f"  Wrong chunk (rank {rank}, {meta['source']} chunk "
                  f"{meta['chunk_index']}, sim {sim:.4f}):" if sim is not None else
                  f"  Wrong chunk (rank {rank}, {meta['source']} chunk {meta['chunk_index']}):")
            print(f"    Distinctive query terms FOUND in wrong chunk: {found}")
            print(f"    Distinctive query terms NOT found: {missing}")
            print(f"    Lexical overlap: {len(found)}/{len(dq)} = {lex_score*100:.0f}%")
            print(f"    Classification: {classification}")
        print()

    # ------------------------------------------------------------------
    # PART 7 — summary table
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PART 7 — DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"{'Q':>2} {'P@5':>5} {'Hit@1':>6} {'RR':>6}  Failure type (see report for evidence)")
    for r in rows:
        flag = ""
        if r["precision"] < 1.0 or r["hit1"] == 0.0:
            flag = "FAILING"
        print(f"{r['qnum']:>2} {r['precision']*100:>4.0f}% {r['hit1']*100:>5.0f}% "
              f"{r['rr']:>5.2f}  {flag}")

    print()
    print("Failure classifications are determined per wrong chunk in the")
    print("per-question sections above; the human-readable report assigns")
    print("each failing question its primary failure mode from that evidence.")
    print("=" * 70)


if __name__ == "__main__":
    main()
