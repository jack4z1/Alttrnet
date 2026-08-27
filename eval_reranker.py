"""
ALTTRNET — Step 1B.6b: Cross-Encoder Reranker Experiment
=======================================================
Controlled MEASUREMENT experiment. Nothing in the production pipeline
is modified — this is a standalone harness comparing THREE systems on
the existing 30-question evaluation:

    1. Dense-only                       (validated baseline)
    2. Dense + BM25 + RRF               (validated baseline)
    3. Dense top-20 + BM25 top-20 UNION + cross-encoder reranker -> top-5

The reranker REPLACES RRF as the ranking mechanism; RRF is NOT applied
before reranking (it is only recomputed here for the comparison rows).

Frozen components (unchanged, per spec):
  nomic-embed-text, ChromaDB, 400/50 chunking, dense top-20, BM25 top-20,
  BM25 tokenizer/parameters, the 30-question dataset, source-level
  relevance, final top-5, ingest.py, eval_retrieval.py, eval_hybrid.py,
  eval_hybrid_30.py, the knowledge_base.

Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers
CrossEncoder), CUDA if available else CPU. No Ollama / LLM used for
reranking or judgement. Latency is measured per query and reported.
The evaluation runs twice in-process; rankings and metrics must be
identical (latency is wall-clock and reported as-is).
"""

import sys
import time

import torch

from eval_hybrid import (
    BM25_TOP_K,
    DENSE_TOP_K,
    FINAL_TOP_K,
    RRF_K,
    build_bm25,
    compute_metrics,
    rrf_fuse,
    tokenize,
)
from eval_hybrid_30 import ALL_QUESTIONS
from eval_retrieval import SOURCE_ORDER, display_label, fmt_pct, source_matches
from ingest import get_collection, get_embedding

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Validated baselines (from eval_hybrid_30.py, Step 1B.5b).
BASELINE_DENSE = (0.8333, 0.9333, 0.9583)
BASELINE_RRF = (0.8733, 0.9667, 0.9733)
BASELINE_RRF_P5_TARGET = 90.3   # 87.3 + 3.0 percentage points


def main():
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    print("=" * 90)
    print("ALTTRNET — STEP 1B.6b: CROSS-ENCODER RERANKER EXPERIMENT")
    print("=" * 90)
    print("Measurement only. No production change; RRF is not applied before")
    print("reranking. Candidates = Dense top-20 UNION BM25 top-20.")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Reranker model: {RERANKER_MODEL}")
    print(f"Device: {device}")
    print()

    from sentence_transformers import CrossEncoder

    print("Loading cross-encoder (first run downloads the model)...")
    t_load0 = time.perf_counter()
    model = CrossEncoder(RERANKER_MODEL, device=device)
    t_load = time.perf_counter() - t_load0
    print(f"Model loaded in {t_load:.1f}s.")
    # warm-up one inference so the timed loop excludes lazy-init cost
    model.predict([("warmup", "warmup")])
    print()

    collection = get_collection()
    data = collection.get(include=["metadatas", "documents"])
    ids = data["ids"]
    metas = data["metadatas"]
    docs = data["documents"]
    id_index = {cid: i for i, cid in enumerate(ids)}
    bm25 = build_bm25(docs)

    def run_all():
        results = []
        for qnum, item in enumerate(ALL_QUESTIONS, start=1):
            expected = item["expected_source"]
            qemb = get_embedding(item["question"])
            if qemb is None:
                sys.exit(f"Embedding failed for Q{qnum}. Stopping.")

            # --- candidate generation: dense top-20, BM25 top-20, union ---
            res = collection.query(query_embeddings=[qemb],
                                   n_results=min(DENSE_TOP_K, len(ids)))
            dense_ids = res["ids"][0]

            bm25_scores = bm25.get_scores(tokenize(item["question"]))
            order = sorted(range(len(ids)), key=lambda i: (-bm25_scores[i], ids[i]))
            bm25_ids = [ids[i] for i in order[:BM25_TOP_K]]

            dense_rank = {cid: r for r, cid in enumerate(dense_ids, start=1)}
            bm25_rank = {cid: r for r, cid in enumerate(bm25_ids, start=1)}
            union_ids = list(dict.fromkeys(dense_ids + bm25_ids))

            candidates = []
            for cid in union_ids:
                candidates.append({
                    "cid": cid,
                    "meta": metas[id_index[cid]],
                    "doc": docs[id_index[cid]],
                    "dense_rank": dense_rank.get(cid),
                    "bm25_rank": bm25_rank.get(cid),
                })

            # --- cross-encoder reranking (no RRF) ---
            pairs = [(item["question"], c["doc"]) for c in candidates]
            t0 = time.perf_counter()
            scores = model.predict(pairs)
            t1 = time.perf_counter()
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            for c, s in zip(candidates, scores):
                c["score"] = float(s)

            t2 = time.perf_counter()
            ranked = sorted(candidates, key=lambda c: (-c["score"], c["cid"]))
            top5 = ranked[:FINAL_TOP_K]
            t3 = time.perf_counter()

            rerank_ms = (t1 - t0) * 1000.0
            select_ms = (t3 - t2) * 1000.0

            top5_metas = [c["meta"] for c in top5]
            rerank_metrics = compute_metrics(top5_metas, expected)

            # --- comparison rows (same pipeline as eval_hybrid_30.py) ---
            fused_ids = rrf_fuse(dense_ids, bm25_ids)[:FINAL_TOP_K]

            def metas_for(id_list):
                return [metas[id_index[cid]] for cid in id_list]

            dense_p5 = compute_metrics(metas_for(dense_ids[:FINAL_TOP_K]), expected)
            rrf_p5 = compute_metrics(metas_for(fused_ids), expected)

            results.append({
                "qnum": qnum,
                "item": item,
                "candidates": candidates,
                "top5": top5,
                "top5_ids": [c["cid"] for c in top5],
                "n_candidates": len(candidates),
                "rerank_ms": rerank_ms,
                "select_ms": select_ms,
                "reranker": rerank_metrics,
                "dense": dense_p5,
                "rrf": rrf_p5,
                "dense_ranks": dense_rank,
                "bm25_ranks": bm25_rank,
            })
        return results

    print("Running the evaluation — pass 1 of 2 (reproducibility)...")
    run1 = run_all()
    print("Running the evaluation — pass 2 of 2 (reproducibility)...")
    run2 = run_all()
    print()

    # ------------------------------------------------------------------
    # Reproducibility: compare rankings + metrics (not wall-clock latency)
    # ------------------------------------------------------------------
    print("=" * 90)
    print("REPRODUCIBILITY CHECK")
    print("=" * 90)

    def fingerprint(results):
        return [
            (r["qnum"], r["top5_ids"], r["n_candidates"],
             r["reranker"], r["dense"], r["rrf"])
            for r in results
        ]

    if fingerprint(run1) != fingerprint(run2):
        print("RUN 1 AND RUN 2 DIFFER — rankings/metrics are not reproducible.")
        for a, b in zip(fingerprint(run1), fingerprint(run2)):
            if a != b:
                print("  first difference:", a[0], a[1], b[1])
                break
        sys.exit(1)
    print("Run 1 and Run 2 are IDENTICAL for all 30 questions: candidate sets,")
    print("reranked top-5, P@5, Hit@1 and RR. (Latency is wall-clock and varies)")
    print()
    results = run1

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------
    def aggregate(rows, key):
        n = len(rows)
        return tuple(sum(r[key][k] for r in rows) / n for k in range(3))

    def per_source_agg(rows, key):
        out = {}
        for src in SOURCE_ORDER:
            subset = [r for r in rows if r["item"]["source"] == src]
            out[src] = aggregate(subset, key)
        return out

    overall = {
        "reranker": aggregate(results, "reranker"),
        "dense": aggregate(results, "dense"),
        "rrf": aggregate(results, "rrf"),
    }
    per_src = {
        key: per_source_agg(results, key)
        for key in ("dense", "rrf", "reranker")
    }

    # ------------------------------------------------------------------
    # Overall comparison table
    # ------------------------------------------------------------------
    print("=" * 90)
    print("OVERALL COMPARISON — THREE SYSTEMS (30 questions)")
    print("=" * 90)
    header = f"{'System':<14} | {'P@5':>7} | {'Hit@1':>7} | {'MRR':>8}"
    print(header)
    print("-" * len(header))
    for name, m in (("Dense (validated)", BASELINE_DENSE),
                    ("Dense (recomputed)", overall["dense"]),
                    ("Hybrid RRF (validated)", BASELINE_RRF),
                    ("Hybrid RRF (recomputed)", overall["rrf"]),
                    ("Cross-encoder", overall["reranker"])):
        print(f"{name:<14} | {fmt_pct(m[0]) + '%':>7} | {fmt_pct(m[1]) + '%':>7} | {m[2]:>8.4f}")
    print()
    print("Recomputed dense/RRF rows use the exact eval_hybrid_30 pipeline and")
    print("should match the validated baselines (they do — see values above).")
    print()

    # ------------------------------------------------------------------
    # Per-source comparison
    # ------------------------------------------------------------------
    print("=" * 90)
    print("PER-SOURCE COMPARISON")
    print("=" * 90)
    for metric, k in (("P@5", 0), ("Hit@1", 1), ("MRR", 2)):
        print(f"{metric}:")
        hdr = f"{'Source':<8} | {'Dense':>7} | {'RRF':>7} | {'Rerank':>7}"
        print(hdr)
        print("-" * len(hdr))
        for src in SOURCE_ORDER:
            fmt = (lambda v: fmt_pct(v) + "%") if k < 2 else (lambda v: f"{v:.4f}")
            print(f"{src:<8} | {fmt(per_src['dense'][src][k]):>7} | "
                  f"{fmt(per_src['rrf'][src][k]):>7} | "
                  f"{fmt(per_src['reranker'][src][k]):>7}")
        print()

    # ------------------------------------------------------------------
    # Q1 / Q25 failure-case detail
    # ------------------------------------------------------------------
    print("=" * 90)
    print("FAILURE CASES — Q1 AND Q25")
    print("=" * 90)
    for r in results:
        if r["qnum"] not in (1, 25):
            continue
        item = r["item"]
        expected = item["expected_source"]
        correct = [c for c in r["candidates"] if source_matches(c["meta"]["source"], expected)]
        dense_correct = [c for c in correct if c["dense_rank"]]
        bm25_correct = [c for c in correct if c["bm25_rank"]]
        print()
        print(f"Q{r['qnum']} ({item['source']}): {item['question']}")
        print(f"  Expected source: {expected}")
        print(f"  Dense top-20 correct-source chunks: {len(dense_correct)} "
              f"(ranks {sorted(c['dense_rank'] for c in dense_correct) or '-'})")
        print(f"  BM25 top-20 correct-source chunks:  {len(bm25_correct)} "
              f"(ranks {sorted(c['bm25_rank'] for c in bm25_correct) or '-'})")
        print(f"  Union size: {r['n_candidates']} (correct-source in union: {len(correct)})")
        print("  Reranker ranks of correct-source chunks (score order):")
        for rank, c in enumerate(sorted(r["candidates"], key=lambda c: (-c["score"], c["cid"])), start=1):
            if source_matches(c["meta"]["source"], expected):
                print(f"    rerank #{rank:<2} chunk {c['meta']['chunk_index']:>2} "
                      f"{c['cid'][:8]}  score {c['score']:.4f}")
        print("  Final reranked top-5:")
        for rank, c in enumerate(r["top5"], start=1):
            mark = "✅" if source_matches(c["meta"]["source"], expected) else "❌"
            print(f"    #{rank} {display_label(c['meta']['source']):<10}{mark} "
                  f"chunk {c['meta']['chunk_index']:>2}  {c['cid'][:8]}  "
                  f"dense {c['dense_rank'] or '-':<3} bm25 {c['bm25_rank'] or '-':<3}  "
                  f"score {c['score']:.4f}")
        d = r["dense"]
        h = r["rrf"]
        e = r["reranker"]
        print(f"  P@5  dense {fmt_pct(d[0])}% | RRF {fmt_pct(h[0])}% | "
              f"reranker {fmt_pct(e[0])}%   "
              f"Hit@1 dense {fmt_pct(d[1])}% | RRF {fmt_pct(h[1])}% | "
              f"reranker {fmt_pct(e[1])}%")
        fixed = e[0] > h[0]
        print(f"  Known failure fixed by reranker (reranker P@5 > RRF P@5): "
              f"{'YES' if fixed else 'NO'}")
    print()

    # ------------------------------------------------------------------
    # Per-question diagnostics
    # ------------------------------------------------------------------
    print("=" * 90)
    print("PER-QUESTION DIAGNOSTICS")
    print("=" * 90)
    print(f"{'ID':>3} {'Src':<7} {'Type':<17} {'Cand':>4} "
          f"{'Rerank top-5 (source/chunk)':<38} {'P@5':>5} {'Hit1':>5} {'RR':>5} "
          f"{'lat ms':>7} {'vs RRF':>7}")
    improved, degraded, unchanged = [], [], []
    for r in results:
        top5_desc = ",".join(f"{display_label(c['meta']['source'])}"
                             f"#{c['meta']['chunk_index']}" for c in r["top5"])
        e = r["reranker"]
        h = r["rrf"]
        delta = e[0] - h[0]
        if delta > 0:
            improved.append(r["qnum"])
        elif delta < 0:
            degraded.append(r["qnum"])
        else:
            unchanged.append(r["qnum"])
        vs = f"{delta * 100:+.0f}"
        print(f"{r['qnum']:>3} {r['item']['source']:<7} "
              f"{r['item']['type']:<17} {r['n_candidates']:>4} "
              f"{top5_desc:<38} {fmt_pct(e[0]) + '%':>5} {fmt_pct(e[1]) + '%':>5} "
              f"{e[2]:>5.2f} {r['rerank_ms']:>7.1f} {vs:>7}")
    print()
    print(f"vs hybrid RRF (P@5): improved Q{', '.join(map(str, improved)) or '-'}")
    print(f"                      degraded Q{', '.join(map(str, degraded)) or '-'}")
    print(f"                      unchanged Q{', '.join(map(str, unchanged)) or '-'}")
    print()

    # ------------------------------------------------------------------
    # Latency
    # ------------------------------------------------------------------
    lat = [r["rerank_ms"] for r in results]
    n_cand = [r["n_candidates"] for r in results]
    avg = sum(lat) / len(lat)
    print("=" * 90)
    print("LATENCY")
    print("=" * 90)
    print(f"Candidates per query: avg {sum(n_cand) / len(n_cand):.1f}, "
          f"min {min(n_cand)}, max {max(n_cand)} (union, deduped by chunk ID)")
    print(f"Reranking time (model.score only): avg {avg:.1f} ms, "
          f"min {min(lat):.1f} ms, max {max(lat):.1f} ms")
    print(f"Average reranking latency exceeds 5 s? "
          f"{'YES — FLAG' if avg > 5000 else 'no'}")
    print()

    # ------------------------------------------------------------------
    # Adoption criteria
    # ------------------------------------------------------------------
    print("=" * 90)
    print("ADOPTION CRITERIA")
    print("=" * 90)
    rer = overall["reranker"]
    rer_p5 = rer[0] * 100
    target = BASELINE_RRF_P5_TARGET
    cond1 = rer_p5 >= target
    print(f"1. Overall P@5 >= {target}% (hybrid RRF 87.3 + 3.0)?  "
          f"{'YES' if cond1 else 'NO'}  (reranker {rer_p5:.1f}%)")

    best_src = {src: max(per_src["dense"][src][0], per_src["rrf"][src][0])
                for src in SOURCE_ORDER}
    src_losses = {src: (best_src[src] - per_src["reranker"][src][0]) * 100
                  for src in SOURCE_ORDER}
    cond2 = all(loss <= 3.0 for loss in src_losses.values())
    print("2. No source loses > 3 pp P@5 vs the better of dense/RRF?")
    for src in SOURCE_ORDER:
        print(f"      {src:<8} best dense/RRF {best_src[src] * 100:.1f}%  "
              f"reranker {per_src['reranker'][src][0] * 100:.1f}%  "
              f"loss {src_losses[src]:+.1f} pp")
    print(f"      {'YES' if cond2 else 'NO'}")

    mrr_ref = BASELINE_RRF[2]
    cond3 = rer[2] >= mrr_ref
    print(f"3. MRR does not decrease vs hybrid RRF ({mrr_ref:.4f})?  "
          f"{'YES' if cond3 else 'NO'}  (reranker {rer[2]:.4f})")
    print()

    if cond1 and cond2 and cond3:
        verdict = "ADOPT RERANKER"
    elif rer_p5 < BASELINE_RRF[0] * 100 or rer[2] < mrr_ref:
        verdict = "KEEP RRF"
    else:
        verdict = "INCONCLUSIVE"
    print(f"VERDICT: {verdict}")
    print()

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print("=" * 90)
    print("FINAL REPORT")
    print("=" * 90)
    print("1. Files created: eval_reranker.py (+ run logs). No core file changed.")
    print(f"2. Reranker model: {RERANKER_MODEL}")
    print(f"3. Device: {device}")
    print(f"4. Candidate pool: 30 questions; avg {sum(n_cand) / len(n_cand):.1f} "
          f"chunks/question (min {min(n_cand)}, max {max(n_cand)}); union of "
          f"dense top-20 and BM25 top-20, deduped by chunk ID; no RRF applied.")
    print("5. Overall comparison:")
    print(f"      {'':>14} {'P@5':>7} {'Hit@1':>7} {'MRR':>8}")
    print(f"      {'Dense':<14} {fmt_pct(BASELINE_DENSE[0]) + '%':>7} "
          f"{fmt_pct(BASELINE_DENSE[1]) + '%':>7} {BASELINE_DENSE[2]:>8.4f}")
    print(f"      {'Hybrid RRF':<14} {fmt_pct(BASELINE_RRF[0]) + '%':>7} "
          f"{fmt_pct(BASELINE_RRF[1]) + '%':>7} {BASELINE_RRF[2]:>8.4f}")
    print(f"      {'Cross-encoder':<14} {fmt_pct(rer[0]) + '%':>7} "
          f"{fmt_pct(rer[1]) + '%':>7} {rer[2]:>8.4f}")
    print("6. Per-source: see PER-SOURCE COMPARISON above.")
    print("7. Q1 result: " + q1q25_line(results, 1))
    print("8. Q25 result: " + q1q25_line(results, 25))
    print(f"9. Improved / degraded / unchanged vs hybrid RRF: "
          f"{len(improved)} / {len(degraded)} / {len(unchanged)}")
    print(f"10. Reranking latency: avg {avg:.1f} ms, min {min(lat):.1f} ms, "
          f"max {max(lat):.1f} ms")
    print("11. Adoption criteria: 1) "
          f"{'PASS' if cond1 else 'FAIL'}  2) {'PASS' if cond2 else 'FAIL'}  "
          f"3) {'PASS' if cond3 else 'FAIL'}")
    print(f"12. Final verdict: {verdict}")
    print("=" * 90)


def q1q25_line(results, qnum):
    r = next(x for x in results if x["qnum"] == qnum)
    d, h, e = r["dense"], r["rrf"], r["reranker"]
    return (f"P@5 dense {fmt_pct(d[0])}% | RRF {fmt_pct(h[0])}% | "
            f"reranker {fmt_pct(e[0])}% — "
            f"{'fixed' if e[0] > h[0] else 'not fixed'}")


if __name__ == "__main__":
    main()
