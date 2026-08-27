"""
ALTTRNET — Step 1B.6a: Verify Reranker Candidate Recall
=======================================================
Verification ONLY. Nothing is modified or implemented:
  * no reranker, no retrieval change, no tuning
  * ingest.py, eval_retrieval.py, eval_hybrid.py, eval_hybrid_30.py,
    BM25, RRF, embeddings, chunking, ChromaDB, questions — untouched

Objective: confirm that the correct source chunks for the known hybrid
failure cases (Q1, Q25) are present in the candidate pool that a future
reranker would reorder:

    Dense top-20  UNION  BM25 top-20

Relevance is judged ONLY by source metadata (the same source-level
definition used by the evaluation harness). No LLM is used. RRF is not
needed for this check; it is only used to identify which questions'
hybrid P@5 regressed relative to dense-only (using the exact same
pipeline as eval_hybrid_30.py).

The full run is executed twice in-process and must be identical; the
script is also byte-identical across separate invocations.
"""

import sys

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
from eval_retrieval import display_label, fmt_pct, source_matches
from ingest import get_collection, get_embedding

MANDATORY = [1, 25]


def preview(text, n=70):
    return (text or "").replace("\n", " ").strip()[:n]


def run_question(qnum, item, collection, bm25, ids, metas, docs, id_index):
    expected = item["expected_source"]
    qemb = get_embedding(item["question"])
    if qemb is None:
        sys.exit(f"Embedding failed for Q{qnum}. Stopping.")

    res = collection.query(query_embeddings=[qemb],
                           n_results=min(DENSE_TOP_K, len(ids)))
    dense_ids = res["ids"][0]

    bm25_scores = bm25.get_scores(tokenize(item["question"]))
    order = sorted(range(len(ids)), key=lambda i: (-bm25_scores[i], ids[i]))
    bm25_ids = [ids[i] for i in order[:BM25_TOP_K]]

    fused_ids = rrf_fuse(dense_ids, bm25_ids)[:FINAL_TOP_K]

    def metas_for(id_list):
        return [metas[id_index[cid]] for cid in id_list]

    # union preserving first-appearance order, keyed by chunk ID
    union_ids = list(dict.fromkeys(dense_ids + bm25_ids))

    dense_rank = {cid: r for r, cid in enumerate(dense_ids, start=1)}
    bm25_rank = {cid: r for r, cid in enumerate(bm25_ids, start=1)}

    correct = [cid for cid in union_ids
               if source_matches(metas[id_index[cid]]["source"], expected)]

    return {
        "qnum": qnum,
        "item": item,
        "dense_ids": dense_ids,
        "bm25_ids": bm25_ids,
        "union_ids": union_ids,
        "correct_ids": correct,
        "dense_rank": dense_rank,
        "bm25_rank": bm25_rank,
        "dense_p5": compute_metrics(metas_for(dense_ids[:FINAL_TOP_K]), expected)[0],
        "hybrid_p5": compute_metrics(metas_for(fused_ids), expected)[0],
    }


def report_question(r, metas, docs, id_index):
    item = r["item"]
    expected = item["expected_source"]
    print("=" * 88)
    print(f"Q{r['qnum']} ({item['source']}, {item['type']})")
    print(f"Question: {item['question']}")
    print(f"Expected source: {expected}")
    print()

    n_dense = sum(1 for c in r["correct_ids"] if c in r["dense_rank"])
    n_bm25 = sum(1 for c in r["correct_ids"] if c in r["bm25_rank"])
    n_union = len(r["correct_ids"])

    dense_ranks = sorted(r["dense_rank"][c] for c in r["correct_ids"] if c in r["dense_rank"])
    bm25_ranks = sorted(r["bm25_rank"][c] for c in r["correct_ids"] if c in r["bm25_rank"])

    print(f"Correct-source chunks in DENSE top-20: {n_dense}  (ranks {dense_ranks or '-'})")
    print(f"Correct-source chunks in BM25  top-20: {n_bm25}  (ranks {bm25_ranks or '-'})")
    print(f"Correct-source chunks in UNION:        {n_union}  "
          f"(union size {len(r['union_ids'])})")
    print()
    if n_union == 0:
        print("  CORRECT CANDIDATE ABSENT")
    else:
        print("  Correct candidates (union rank | dense rank | bm25 rank | "
              "chunk index | chunk id | preview):")
        for ur, cid in enumerate(r["union_ids"], start=1):
            if cid not in r["correct_ids"]:
                continue
            meta = metas[id_index[cid]]
            dr = r["dense_rank"].get(cid, "-")
            br = r["bm25_rank"].get(cid, "-")
            print(f"    #{ur:<2} dense {dr!s:<3} bm25 {br!s:<3} "
                  f"chunk {meta['chunk_index']:>2}  {cid[:8]}  "
                  f"{preview(docs[id_index[cid]])}")
        print()
        print("  CORRECT CANDIDATE PRESENT")
    print()


def main():
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    print("=" * 88)
    print("ALTTRNET — STEP 1B.6a: VERIFY RERANKER CANDIDATE RECALL")
    print("=" * 88)
    print("Verification only. Dense top-20 + BM25 top-20 candidate pool;")
    print("source-level relevance; no LLM, no reranker, no RRF reordering.")
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
            results.append(run_question(qnum, item, collection, bm25, ids,
                                        metas, docs, id_index))
        return results

    print("Running the verification — pass 1 of 2 (reproducibility)...")
    run1 = run_all()
    print("Running the verification — pass 2 of 2 (reproducibility)...")
    run2 = run_all()
    print()
    print("=" * 88)
    print("REPRODUCIBILITY CHECK")
    print("=" * 88)
    if run1 != run2:
        print("RUN 1 AND RUN 2 DIFFER — verification is not reproducible.")
        for a, b in zip(run1, run2):
            if a != b:
                print(f"First difference at Q{a['qnum']}:")
                print("  run 1:", a)
                print("  run 2:", b)
                break
        sys.exit(1)
    print("Run 1 and Run 2 are IDENTICAL. Verification is deterministic.")
    print()
    results = run1

    # Identify every question where hybrid RRF P@5 < dense-only P@5.
    regressions = [r for r in results if r["hybrid_p5"] < r["dense_p5"]]
    scope_qnums = sorted(set(MANDATORY) | {r["qnum"] for r in regressions})

    print("=" * 88)
    print("HYBRID REGRESSIONS (hybrid RRF P@5 < dense-only P@5), all 30 questions")
    print("=" * 88)
    if regressions:
        for r in regressions:
            print(f"  Q{r['qnum']}: dense P@5 {fmt_pct(r['dense_p5'])}% -> "
                  f"hybrid P@5 {fmt_pct(r['hybrid_p5'])}%")
    else:
        print("  none")
    print()
    print(f"Questions checked for candidate recall: {scope_qnums}")
    print()

    for r in results:
        if r["qnum"] in scope_qnums:
            report_question(r, metas, docs, id_index)

    # ------------------------------------------------------------------
    # Final conclusion
    # ------------------------------------------------------------------
    print("=" * 88)
    print("FINAL CONCLUSION")
    print("=" * 88)
    by_q = {r["qnum"]: r for r in results}
    ok = []
    for q in MANDATORY:
        n = len(by_q[q]["correct_ids"])
        status = "PASS" if n > 0 else "FAIL"
        ok.append(n > 0)
        print(f"Q{q}: {n} correct-source candidate(s) in Dense top-20 UNION BM25 "
              f"top-20 -> candidate recall {status}")
    for r in regressions:
        if r["qnum"] not in MANDATORY:
            n = len(r["correct_ids"])
            status = "PASS" if n > 0 else "FAIL"
            print(f"Q{r['qnum']} (other regression): {n} correct-source candidate(s) "
                  f"in the union -> candidate recall {status}")
    print()
    if all(ok):
        print("Reranking experiment is justified: the observed problem is "
              "ranking/fusion rather than candidate recall.")
    else:
        print("Reranking alone may not solve the failure because candidate "
              "recall is insufficient.")
    print("=" * 88)


if __name__ == "__main__":
    main()
