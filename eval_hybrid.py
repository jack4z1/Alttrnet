"""
ALTTRNET — Step 1B.5: BM25 + Dense Retrieval + RRF Experiment
============================================================
Controlled retrieval experiment measuring ONE architectural change:

    DENSE RETRIEVAL + BM25 + RECIPROCAL RANK FUSION (RRF)

against the existing dense-only retrieval baseline (eval_retrieval.py).

Everything else is held fixed:
  * the same ChromaDB collection (30 chunks, 3 sources) — read-only
  * the same nomic-embed-text embedding function (ingest.get_embedding)
  * the same 15 evaluation questions (eval_retrieval.QUESTIONS)
  * the same source-metadata relevance judgement and metrics
    (Precision@5, Hit@1, MRR) — NO LLM is called anywhere

Only the retrieval method differs:
  * Dense-only: nomic-embed-text -> ChromaDB top-5          (baseline)
  * Hybrid:     nomic-embed-text -> ChromaDB top-20
                + BM25Okapi over the 30 chunk documents -> top-20
                + RRF (k=60) -> fused top-5

This script creates NOTHING and modifies NOTHING: it reads the existing
collection, embeds questions with the existing function, and reports
retrieval metrics only. ingest.py, eval_retrieval.py and chroma_db are
untouched, so the dense-only baseline remains reproducible.
"""

import re
import sys

from rank_bm25 import BM25Okapi

from eval_retrieval import (
    QUESTIONS,
    SOURCE_ORDER,
    check_embed_model,
    display_label,
    fmt_pct,
    source_matches,
)
from ingest import (
    COLLECTION_NAME,
    EMBED_MODEL,
    get_collection,
    get_embedding,
    retrieve_chunks,
)

# ---------------------------------------------------------------------------
# Experiment constants — the ONLY retrieval parameters that differ from the
# production baseline (which stays at dense top-5).
# ---------------------------------------------------------------------------

DENSE_TOP_K = 20   # dense candidates per question (baseline uses 5)
BM25_TOP_K = 20    # BM25 candidates per question
RRF_K = 60         # Reciprocal Rank Fusion constant
FINAL_TOP_K = 5    # fused result returned (matches baseline top-5)

# Deterministic, framework-free tokenizer shared by chunks and questions:
# lowercase -> keep word-like tokens -> drop punctuation. No stop words, no
# stemming, no LLM, no semantic preprocessing.
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase, keep word-like tokens, drop punctuation. Deterministic."""
    return TOKEN_RE.findall((text or "").lower())


# ---------------------------------------------------------------------------
# BM25 (chunk-level index over the EXISTING 30 chunks)
# ---------------------------------------------------------------------------

def build_bm25(chunk_docs):
    """Build a BM25Okapi index where each existing chunk is one document."""
    corpus = [tokenize(doc) for doc in chunk_docs]
    return BM25Okapi(corpus)


def bm25_top_ids(bm25, question, ids, top_k=BM25_TOP_K):
    """Rank all chunks by BM25 score; return the top-k chunk IDs (ties broken
    by ID for determinism)."""
    scores = bm25.get_scores(tokenize(question))
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
    return [ids[i] for i in order[:top_k]]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def rrf_fuse(dense_ids, bm25_ids, k=RRF_K):
    """
    Fuse two ranked ID lists with Reciprocal Rank Fusion.

    Each list contributes 1 / (k + rank) per chunk; a chunk present in both
    lists accumulates both contributions. Chunk identity = the existing
    ChromaDB chunk ID (never the source URL alone).
    """
    scores = {}
    for rank, cid in enumerate(dense_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(bm25_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [cid for cid, _ in ranked]


# ---------------------------------------------------------------------------
# Metrics (identical to the baseline harness: metadata-only judgement)
# ---------------------------------------------------------------------------

def compute_metrics(metadatas, expected_source):
    """Return (Precision@5, Hit@1, Reciprocal Rank) for a top-5 list."""
    n = len(metadatas)
    relevant = sum(
        1 for m in metadatas
        if source_matches(m.get("source", "unknown"), expected_source)
    )
    precision = relevant / n if n else 0.0
    hit = 1.0 if metadatas and source_matches(
        metadatas[0].get("source", "unknown"), expected_source
    ) else 0.0
    rr = 0.0
    for rank, m in enumerate(metadatas, start=1):
        if source_matches(m.get("source", "unknown"), expected_source):
            rr = 1.0 / rank
            break
    return precision, hit, rr


# ---------------------------------------------------------------------------
# Evaluation — runs all 15 questions through both retrieval methods.
# Pure function of the (immutable) DB + questions, so it can be re-run for
# the reproducibility check.
# ---------------------------------------------------------------------------

def run_questions(collection, bm25, ids, metas):
    """Run all 15 questions. Returns one result dict per question."""
    id_index = {cid: i for i, cid in enumerate(ids)}
    results = []

    for qnum, item in enumerate(QUESTIONS, start=1):
        expected = item["expected_source"]

        question_embedding = get_embedding(item["question"])
        if question_embedding is None:
            print(f"ERROR: embedding failed for Q{qnum}. Stopping.")
            sys.exit(1)

        # --- Dense retrieval: existing implementation, top-20 for this
        # experiment only (production stays at top-5). --------------------
        res = retrieve_chunks(collection, question_embedding, top_k=DENSE_TOP_K)
        if res is None:
            print(f"ERROR: dense retrieval failed for Q{qnum}. Stopping.")
            sys.exit(1)
        dense_ids = res["ids"][0]

        # --- BM25 retrieval over the same 30 chunks. ---------------------
        bm25_ids = bm25_top_ids(bm25, item["question"], ids)

        # --- RRF fusion -> final top-5. -----------------------------------
        fused_ids = rrf_fuse(dense_ids, bm25_ids)[:FINAL_TOP_K]

        def metas_for(id_list):
            return [metas[id_index[cid]] for cid in id_list]

        dense_top5 = metas_for(dense_ids[:FINAL_TOP_K])
        bm25_top5 = metas_for(bm25_ids[:FINAL_TOP_K])
        fused_top5 = metas_for(fused_ids)

        results.append({
            "qnum": qnum,
            "item": item,
            "dense_ids_top5": dense_ids[:FINAL_TOP_K],
            "dense_metas_top5": dense_top5,
            "bm25_ids_top5": bm25_ids[:FINAL_TOP_K],
            "bm25_metas_top5": bm25_top5,
            "fused_ids_top5": fused_ids,
            "fused_metas_top5": fused_top5,
            "dense": compute_metrics(dense_top5, expected),
            "hybrid": compute_metrics(fused_top5, expected),
        })

    return results


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate(results):
    """Overall (Precision@5, Hit@1, MRR) for dense and hybrid."""
    n = len(results)
    dense = tuple(sum(r["dense"][k] for r in results) / n for k in range(3))
    hybrid = tuple(sum(r["hybrid"][k] for r in results) / n for k in range(3))
    return dense, hybrid


def per_source(results):
    """{source: (dense (p, h, rr), hybrid (p, h, rr))} for the 3 sources."""
    out = {}
    for src in SOURCE_ORDER:
        subset = [r for r in results if r["item"]["source"] == src]
        n = len(subset)
        dense = tuple(sum(r["dense"][k] for r in subset) / n for k in range(3))
        hybrid = tuple(sum(r["hybrid"][k] for r in subset) / n for k in range(3))
        out[src] = (dense, hybrid)
    return out


def cmp_metrics(dense, hybrid):
    """Compare one question's dense vs hybrid metrics."""
    better = sum(1 for k in range(3) if hybrid[k] > dense[k])
    worse = sum(1 for k in range(3) if hybrid[k] < dense[k])
    if better and not worse:
        return "improved"
    if worse and not better:
        return "worse"
    if better and worse:
        return "mixed"
    return "unchanged"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def source_line(rank, meta, expected):
    """One line of a retrieved top-5 listing: label, relevance mark, URL."""
    src = meta.get("source", "unknown")
    mark = "✅" if source_matches(src, expected) else "❌"
    return f"  {rank}. {display_label(src):<10}{mark}  {src}"


def fmt_pp(pp):
    """Format a difference in percentage points with sign, e.g. '+5.3'."""
    return f"{pp:+.1f}"


def print_question_block(r):
    """Full per-question listing: question, expected source, dense / BM25 /
    RRF top-5 sources, and both metric triples."""
    item = r["item"]
    expected = item["expected_source"]
    print("=" * 70)
    print(f"QUESTION {r['qnum']} — {item['source']} ({item['type']})")
    print("=" * 70)
    print(f"Question: {item['question']}")
    print(f"Expected source: {expected}")
    print()
    print("Dense top-5 sources:")
    for rank, m in enumerate(r["dense_metas_top5"], start=1):
        print(source_line(rank, m, expected))
    print("BM25 top-5 sources:")
    for rank, m in enumerate(r["bm25_metas_top5"], start=1):
        print(source_line(rank, m, expected))
    print("RRF top-5 sources:")
    for rank, m in enumerate(r["fused_metas_top5"], start=1):
        print(source_line(rank, m, expected))
    print()
    dp, dh, drr = r["dense"]
    hp, hh, hrr = r["hybrid"]
    print(f"Dense-only: P@5 {fmt_pct(dp)}%  Hit@1 {fmt_pct(dh)}%  RR {drr:.4f}")
    print(f"Hybrid RRF: P@5 {fmt_pct(hp)}%  Hit@1 {fmt_pct(hh)}%  RR {hrr:.4f}")
    print()


def print_comparison(dense, hybrid):
    """Overall comparison table: Metric | Dense-only | Hybrid RRF | Difference."""
    print("=" * 70)
    print("COMPARISON — OVERALL (15 questions)")
    print("=" * 70)
    header = f"{'Metric':<12} | {'Dense-only':>10} | {'Hybrid RRF':>10} | {'Difference':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'Precision@5':<12} | {fmt_pct(dense[0]) + '%':>10} | "
          f"{fmt_pct(hybrid[0]) + '%':>10} | {fmt_pp((hybrid[0] - dense[0]) * 100):>10}")
    print(f"{'Hit@1':<12} | {fmt_pct(dense[1]) + '%':>10} | "
          f"{fmt_pct(hybrid[1]) + '%':>10} | {fmt_pp((hybrid[1] - dense[1]) * 100):>10}")
    print(f"{'MRR':<12} | {dense[2]:>10.4f} | {hybrid[2]:>10.4f} | "
          f"{hybrid[2] - dense[2]:>+10.4f}")
    print()


def print_per_source(per_src):
    """Per-source comparison tables for P@5, Hit@1 and MRR."""
    for metric_name, k in (("Precision@5", 0), ("Hit@1", 1), ("MRR", 2)):
        print(f"Per-source {metric_name}:")
        header = f"{'Source':<8} | {'Dense-only':>10} | {'Hybrid RRF':>10} | {'Difference':>10}"
        print(header)
        print("-" * len(header))
        for src in SOURCE_ORDER:
            d, h = per_src[src]
            if metric_name == "MRR":
                diff = f"{h[k] - d[k]:+.4f}"
                print(f"{src:<8} | {d[k]:>10.4f} | {h[k]:>10.4f} | {diff:>10}")
            else:
                diff = fmt_pp((h[k] - d[k]) * 100)
                print(f"{src:<8} | {fmt_pct(d[k]) + '%':>10} | {fmt_pct(h[k]) + '%':>10} | {diff:>10}")
        print()


def print_failure_analysis(results):
    """Show dense / BM25 / RRF rankings for questions where the hybrid
    ranking differs from dense-only. Claims about 'fixed' questions are made
    only where the retrieved ranking actually demonstrates improvement."""
    print("=" * 70)
    print("FAILURE ANALYSIS — QUESTIONS WHERE HYBRID DIFFERS FROM DENSE-ONLY")
    print("=" * 70)
    differing = [r for r in results if r["dense_ids_top5"] != r["fused_ids_top5"]]
    if not differing:
        print("None — the hybrid ranking is identical to dense-only for every question.")
        return
    for r in differing:
        item = r["item"]
        expected = item["expected_source"]
        verdict = cmp_metrics(r["dense"], r["hybrid"])
        print()
        print(f"Q{r['qnum']} ({item['source']}) — {item['question']}")
        print(f"Expected source: {expected}")
        print("Dense top-5:")
        for rank, m in enumerate(r["dense_metas_top5"], start=1):
            print(source_line(rank, m, expected))
        print("BM25 top-5:")
        for rank, m in enumerate(r["bm25_metas_top5"], start=1):
            print(source_line(rank, m, expected))
        print("RRF top-5:")
        for rank, m in enumerate(r["fused_metas_top5"], start=1):
            print(source_line(rank, m, expected))
        dp, dh, drr = r["dense"]
        hp, hh, hrr = r["hybrid"]
        print(f"Dense-only: P@5 {fmt_pct(dp)}%  Hit@1 {fmt_pct(dh)}%  RR {drr:.4f}")
        print(f"Hybrid RRF: P@5 {fmt_pct(hp)}%  Hit@1 {fmt_pct(hh)}%  RR {hrr:.4f}")
        print(f"Question-level outcome: {verdict}")
        print()


def print_improved_worse(results):
    """List which questions improved / became worse / stayed identical."""
    print("=" * 70)
    print("PER-QUESTION OUTCOMES")
    print("=" * 70)
    buckets = {"improved": [], "worse": [], "mixed": [], "unchanged": []}
    for r in results:
        buckets[cmp_metrics(r["dense"], r["hybrid"])].append(r["qnum"])
    print(f"Improved:  {', '.join(f'Q{n}' for n in buckets['improved']) or 'none'}")
    print(f"Worse:     {', '.join(f'Q{n}' for n in buckets['worse']) or 'none'}")
    print(f"Mixed:     {', '.join(f'Q{n}' for n in buckets['mixed']) or 'none'}")
    print(f"Unchanged: {', '.join(f'Q{n}' for n in buckets['unchanged']) or 'none'}")
    print()

    # Focus cases from the diagnosis: Q1 and Q9-Q15.
    print("Focus cases (previously diagnosed failures):")
    for r in results:
        if r["qnum"] == 1 or r["qnum"] >= 9:
            item = r["item"]
            verdict = cmp_metrics(r["dense"], r["hybrid"])
            dp, dh, drr = r["dense"]
            hp, hh, hrr = r["hybrid"]
            print(f"  Q{r['qnum']}: {verdict}  "
                  f"(dense P@5 {fmt_pct(dp)}%/Hit@1 {fmt_pct(dh)}%/RR {drr:.4f} -> "
                  f"hybrid P@5 {fmt_pct(hp)}%/Hit@1 {fmt_pct(hh)}%/RR {hrr:.4f})")
    print()


def print_interpretation(dense, hybrid, per_src):
    """Decision section per the Step 1B.5 spec thresholds."""
    print("=" * 70)
    print("EXPERIMENT INTERPRETATION")
    print("=" * 70)
    p5_pp = (hybrid[0] - dense[0]) * 100
    hit_pp = (hybrid[1] - dense[1]) * 100
    mrr_delta = hybrid[2] - dense[2]
    print(f"Absolute Precision@5 change: {fmt_pp(p5_pp)} pp")
    print(f"Absolute Hit@1 change: {fmt_pp(hit_pp)} pp")
    print(f"Absolute MRR change: {mrr_delta:+.4f}")
    print()
    print("Per-source changes (P@5 / Hit@1 / MRR):")
    for src in SOURCE_ORDER:
        d, h = per_src[src]
        print(f"  {src}: P@5 {fmt_pp((h[0] - d[0]) * 100)} pp, "
              f"Hit@1 {fmt_pp((h[1] - d[1]) * 100)} pp, MRR {h[2] - d[2]:+.4f}")
    print()
    print("Decision criteria (thresholds from the Step 1B.5 spec):")
    src_min = min((h[0] - d[0]) * 100 for d, h in per_src.values())
    print(f"  1) Average Precision@5 improves by >= 5 pp?  "
          f"{'YES' if p5_pp >= 5.0 else 'NO'}  ({fmt_pp(p5_pp)} pp)")
    print(f"  2) No source decreases by more than 5 pp?     "
          f"{'YES' if src_min >= -5.0 else 'NO'}  (worst source {fmt_pp(src_min)} pp)")
    print(f"  3) MRR does not materially decrease?          "
          f"{'YES' if mrr_delta >= -0.01 else 'NO'}  ({mrr_delta:+.4f})")
    print()
    if p5_pp >= 5.0 and src_min >= -5.0 and mrr_delta >= -0.01:
        verdict = "IMPROVEMENT"
        why = (f"Average Precision@5 rose {fmt_pp(p5_pp)} pp (>= +5 pp), no source "
               f"dropped more than 5 pp (worst {fmt_pp(src_min)} pp), and MRR did not "
               f"materially decrease ({mrr_delta:+.4f}).")
    elif p5_pp < 0.0 and mrr_delta < 0.0:
        verdict = "REGRESSION"
        why = (f"Both Precision@5 ({fmt_pp(p5_pp)} pp) and MRR ({mrr_delta:+.4f}) "
               f"decreased relative to dense-only.")
    else:
        verdict = "NO CLEAR IMPROVEMENT"
        why = (f"The measured change does not meet the bar for an improvement "
               f"(P@5 {fmt_pp(p5_pp)} pp, MRR {mrr_delta:+.4f}) and is not a clear "
               f"regression.")
    print(f"VERDICT: {verdict}")
    print(f"Explanation: {why}")
    print()


def print_final_report(dense, hybrid, per_src, results, q1_verdict):
    """16-item final report required by the Step 1B.5 spec."""
    p5_pp = (hybrid[0] - dense[0]) * 100
    hit_pp = (hybrid[1] - dense[1]) * 100
    mrr_delta = hybrid[2] - dense[2]

    improved = [r for r in results if cmp_metrics(r["dense"], r["hybrid"]) == "improved"]
    worse = [r for r in results if cmp_metrics(r["dense"], r["hybrid"]) == "worse"]

    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print("1. Files created/changed:")
    print("     created: eval_hybrid.py")
    print("     changed: none — eval_retrieval.py, ingest.py and chroma_db untouched")
    print("2. BM25 library used: rank_bm25 (BM25Okapi, k1=1.5, b=0.75)")
    print("3. Existing 30 chunks used: YES (16 Python / 7 RAG / 7 Ollama), "
          "read from the existing chroma_db collection")
    print(f"4. Dense top-K: {DENSE_TOP_K}")
    print(f"5. BM25 top-K: {BM25_TOP_K}")
    print(f"6. RRF k: {RRF_K}")
    print(f"7. Dense-only results: P@5 {fmt_pct(dense[0])}%  Hit@1 {fmt_pct(dense[1])}%  "
          f"MRR {dense[2]:.4f}")
    print(f"8. Hybrid results:     P@5 {fmt_pct(hybrid[0])}%  Hit@1 {fmt_pct(hybrid[1])}%  "
          f"MRR {hybrid[2]:.4f}")
    print(f"9. Absolute metric differences: P@5 {fmt_pp(p5_pp)} pp, "
          f"Hit@1 {fmt_pp(hit_pp)} pp, MRR {mrr_delta:+.4f}")
    print("10. Per-source differences:")
    for src in SOURCE_ORDER:
        d, h = per_src[src]
        print(f"      {src}: P@5 {fmt_pp((h[0] - d[0]) * 100)} pp, "
              f"Hit@1 {fmt_pp((h[1] - d[1]) * 100)} pp, MRR {h[2] - d[2]:+.4f}")
    print(f"11. Questions improved: "
          f"{', '.join(f'Q{r['qnum']}' for r in improved) or 'none'}")
    print(f"12. Questions worse: "
          f"{', '.join(f'Q{r['qnum']}' for r in worse) or 'none'}")
    print(f"13. Q1: {q1_verdict}")
    q9_15 = [r for r in results if r["qnum"] >= 9]
    for r in q9_15:
        print(f"      Q{r['qnum']}: {cmp_metrics(r['dense'], r['hybrid'])}")
    print("14. Q9-Q15 overall: " + (
        "improved" if all(cmp_metrics(r['dense'], r['hybrid']) == 'improved' for r in q9_15)
        else "mixed / not uniformly improved"
    ))
    print("15-16. Verdict and explanation: see EXPERIMENT INTERPRETATION section above.")
    print()


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
    print("ALTTRNET — STEP 1B.5: BM25 + DENSE RETRIEVAL + RRF EXPERIMENT")
    print("=" * 70)
    print()
    print("This experiment changes ONLY the retrieval method. The baseline")
    print("(eval_retrieval.py), the corpus (chroma_db), ingest.py and the")
    print("15-question dataset are untouched. No LLM is called — metrics are")
    print("computed purely from retrieved chunk metadata.")
    print()

    check_embed_model()

    collection = get_collection()
    data = collection.get(include=["metadatas", "documents"])
    ids = data["ids"]
    metas = data["metadatas"]
    docs = data["documents"]
    total = len(ids)

    print("Configuration:")
    print(f"  Embedding model: {EMBED_MODEL}")
    print(f"  Vector database: ChromaDB collection '{COLLECTION_NAME}' (read-only)")
    print(f"  Dense top-K: {DENSE_TOP_K}")
    print(f"  BM25 top-K: {BM25_TOP_K}")
    print(f"  RRF constant k: {RRF_K}")
    print(f"  Final fused top-K: {FINAL_TOP_K}")
    print(f"  BM25 library: rank_bm25 (BM25Okapi, k1=1.5, b=0.75)")
    print(f"  LLM: none — retrieval metrics only")
    print()
    print("Knowledge base:")
    print(f"  {total} chunks")
    from collections import Counter
    for src, count in sorted(Counter(m["source"] for m in metas).items()):
        print(f"  {count:>2} chunks  {src}")
    if total != 30:
        print(f"  WARNING: expected 30 chunks, found {total}.")
    print()

    print("Building BM25 index over the existing chunks (chunk-level)...")
    bm25 = build_bm25(docs)
    print("BM25 index ready.")
    print()

    print("Running the full evaluation — pass 1 of 2 (reproducibility)...")
    run1 = run_questions(collection, bm25, ids, metas)
    print("Running the full evaluation — pass 2 of 2 (reproducibility)...")
    run2 = run_questions(collection, bm25, ids, metas)
    print()

    print("=" * 70)
    print("REPRODUCIBILITY CHECK")
    print("=" * 70)
    if run1 != run2:
        print("RUN 1 AND RUN 2 DIFFER — the experiment is not reproducible.")
        for a, b in zip(run1, run2):
            if a != b:
                print(f"First difference at Q{a['qnum']}:")
                print("  run 1:", a)
                print("  run 2:", b)
                break
        print("STOP: do not report these results until the cause is understood.")
        sys.exit(1)
    print("Run 1 and Run 2 are IDENTICAL for all 15 questions, rankings and")
    print("metrics. The experiment is reproducible.")
    print()
    results = run1

    for r in results:
        print_question_block(r)

    dense, hybrid = aggregate(results)
    per_src = per_source(results)

    print_comparison(dense, hybrid)
    print_per_source(per_src)
    print_failure_analysis(results)
    print_improved_worse(results)
    print_interpretation(dense, hybrid, per_src)
    print_final_report(dense, hybrid, per_src, results, cmp_metrics(results[0]["dense"], results[0]["hybrid"]))

    print("=" * 70)


if __name__ == "__main__":
    main()
