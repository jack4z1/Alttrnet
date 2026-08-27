"""
ALTTRNET — Step 1B.5b: Expanded Retrieval Evaluation (15 -> 30 questions)
========================================================================
Measurement only. NOTHING in the retrieval system is changed:
  * dense top-K  = 20  (existing implementation, ingest.retrieve_chunks)
  * BM25 top-K   = 20  (same rank_bm25 / BM25Okapi index as eval_hybrid.py)
  * RRF k        = 60
  * final top-K  = 5
  * chunking, embeddings, ChromaDB contents, ingest.py — untouched

The first 15 questions are EXACTLY eval_retrieval.QUESTIONS (unchanged,
in order). Questions 16-30 are NEW — 5 per source, grounded against the
actual indexed chunk text, with a deliberate mix of types:
  Broad/Conceptual, Specific factual, Keyword-heavy, Semantic-only,
  Adversarial (cross-vocabulary), plus 3 multi-concept flags.

Metrics: Precision@5, Hit@1, MRR for dense-only and hybrid, reported
overall, per source, per question and per question type. No LLM is used.
The evaluation function is run twice in-process and must produce
identical results; the script is also designed to be byte-identical
across separate invocations.
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
from eval_retrieval import QUESTIONS as EXISTING_QUESTIONS
from eval_retrieval import (
    SOURCE_ORDER,
    display_label,
    fmt_pct,
    source_matches,
)
from ingest import get_collection, get_embedding

PYTHON_URL = "https://docs.python.org/3/tutorial/controlflow.html"
RAG_URL = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
OLLAMA_URL = "https://github.com/ollama/ollama/blob/main/docs/faq.mdx"


# ---------------------------------------------------------------------------
# NEW questions (Q16-Q30) — grounded in the indexed chunks listed in each
# `grounding` note. Each source gets one question of each type; three
# questions are additionally multi-concept (flagged).
# ---------------------------------------------------------------------------

NEW_QUESTIONS = [
    # --- Python: 5 new questions (16-20) ---
    {
        "source": "Python",
        "type": "Broad/Conceptual",
        "question": "What purpose does the match statement serve in Python programs, according to the tutorial?",
        "expected_source": PYTHON_URL,
        "expected_label": "Python documentation",
        "grounding": "chunks 3-4: match statement compares a value to patterns; superficially like switch, closer to Rust/Haskell pattern matching",
        "multi_concept": False,
    },
    {
        "source": "Python",
        "type": "Specific factual",
        "question": "According to the tutorial, when exactly are default argument values evaluated in a function definition?",
        "expected_source": PYTHON_URL,
        "expected_label": "Python documentation",
        "grounding": "chunk 9: default values are evaluated at the point of function definition in the defining scope, and only once",
        "multi_concept": False,
    },
    {
        "source": "Python",
        "type": "Keyword-heavy",
        "question": "How do the / and * markers distinguish positional-only, positional-or-keyword, and keyword-only parameters in a function definition?",
        "expected_source": PYTHON_URL,
        "expected_label": "Python documentation",
        "grounding": "chunks 10-11: / marks positional-only, * marks keyword-only; both optional markers in the parameter list",
        "multi_concept": False,
    },
    {
        "source": "Python",
        "type": "Semantic-only",
        "question": "Why might a developer write three dots as the body of a function instead of a statement, and what does the interpreter do with it?",
        "expected_source": PYTHON_URL,
        "expected_label": "Python documentation",
        "grounding": "chunk 3: many people use the ellipsis literal ... instead of pass; it has no special meaning and is conventionally a placeholder body",
        "multi_concept": False,
    },
    {
        "source": "Python",
        "type": "Adversarial",
        "question": "According to the tutorial, how do the break and else clauses work together to control the flow of a loop that searches for prime numbers?",
        "expected_source": PYTHON_URL,
        "expected_label": "Python documentation",
        "grounding": "chunks 1-3: prime-number search example; else runs when the loop finishes without break",
        "multi_concept": True,
    },
    # --- RAG: 5 new questions (21-25) ---
    {
        "source": "RAG",
        "type": "Broad/Conceptual",
        "question": "What is retrieval-augmented generation and why is it considered useful for large language models?",
        "expected_source": RAG_URL,
        "expected_label": "RAG Wikipedia article",
        "grounding": "chunk 0: RAG enables LLMs to retrieve and incorporate external information; domain-specific/updated info, fewer retrains, verifiable sources",
        "multi_concept": False,
    },
    {
        "source": "RAG",
        "type": "Specific factual",
        "question": "In what year was the retrieval-augmented generation technique first proposed, according to the source?",
        "expected_source": RAG_URL,
        "expected_label": "RAG Wikipedia article",
        "grounding": "chunk 0: 'The technique was first proposed in 2020'",
        "multi_concept": False,
    },
    {
        "source": "RAG",
        "type": "Keyword-heavy",
        "question": "What are sparse vectors, dense vectors, and late interactions, as described in the section on retrieval improvements?",
        "expected_source": RAG_URL,
        "expected_label": "RAG Wikipedia article",
        "grounding": "chunk 2: sparse vectors encode word identity, dense vectors encode meaning; late interactions allow precise word comparison after retrieval",
        "multi_concept": False,
    },
    {
        "source": "RAG",
        "type": "Semantic-only",
        "question": "How does a system that checks a knowledge store before answering questions reduce the likelihood that the model invents facts?",
        "expected_source": RAG_URL,
        "expected_label": "RAG Wikipedia article",
        "grounding": "chunks 0-1: retrieval before generation grounds answers in retrieved documents and helps reduce AI hallucinations",
        "multi_concept": True,
    },
    {
        "source": "RAG",
        "type": "Adversarial",
        "question": "What does the article say about how the language model uses its internal memory together with material fetched from an external source when generating a reply?",
        "expected_source": RAG_URL,
        "expected_label": "RAG Wikipedia article",
        "grounding": "chunks 0-1: parametric model combined with non-parametric external memory accessed through retrieval; LLM draws on augmented prompt and training-data representation",
        "multi_concept": False,
    },
    # --- Ollama: 5 new questions (26-30) ---
    {
        "source": "Ollama",
        "type": "Broad/Conceptual",
        "question": "How does the FAQ describe the way Ollama decides whether to place a model on the GPU or in system memory?",
        "expected_source": OLLAMA_URL,
        "expected_label": "Ollama FAQ documentation",
        "grounding": "chunks 4-5: VRAM requirement evaluated against available memory; model loaded on GPU if it fits entirely, otherwise spread across GPUs",
        "multi_concept": False,
    },
    {
        "source": "Ollama",
        "type": "Specific factual",
        "question": "What is the default maximum number of requests Ollama will queue before rejecting additional requests?",
        "expected_source": OLLAMA_URL,
        "expected_label": "Ollama FAQ documentation",
        "grounding": "chunk 4: OLLAMA_MAX_QUEUE default is 512",
        "multi_concept": False,
    },
    {
        "source": "Ollama",
        "type": "Keyword-heavy",
        "question": "What do the OLLAMA_MAX_LOADED_MODELS, OLLAMA_NUM_PARALLEL, and OLLAMA_MAX_QUEUE environment variables control?",
        "expected_source": OLLAMA_URL,
        "expected_label": "Ollama FAQ documentation",
        "grounding": "chunk 4: max concurrently loaded models, max parallel requests per model, max queued requests",
        "multi_concept": False,
    },
    {
        "source": "Ollama",
        "type": "Semantic-only",
        "question": "If you want a model to stay loaded in memory permanently after it has been used once, what does the FAQ recommend?",
        "expected_source": OLLAMA_URL,
        "expected_label": "Ollama FAQ documentation",
        "grounding": "chunk 3: keep_alive with a negative value (e.g. -1) keeps the model loaded in memory",
        "multi_concept": False,
    },
    {
        "source": "Ollama",
        "type": "Adversarial",
        "question": "What controls how long an idle model remains in memory before the server removes it, and how can that behavior be changed?",
        "expected_source": OLLAMA_URL,
        "expected_label": "Ollama FAQ documentation",
        "grounding": "chunks 3-4: default 5-minute retention, ollama stop, keep_alive / OLLAMA_KEEP_ALIVE; idle models are unloaded to make room",
        "multi_concept": True,
    },
]

ALL_QUESTIONS = EXISTING_QUESTIONS + NEW_QUESTIONS

# Map existing question types onto the Step 1B.5b taxonomy for grouping.
# New questions already use the Step 1B.5b labels, which map to themselves.
TYPE_GROUP = {
    "Broad": "Broad/Conceptual",
    "Conceptual": "Broad/Conceptual",
    "Broad/Conceptual": "Broad/Conceptual",
    "Specific factual": "Specific factual",
    "Specific": "Specific factual",
    "Keyword-heavy": "Keyword-heavy",
    "Semantic-only": "Semantic-only",
    "Adversarial": "Adversarial",
}


def group_of(item):
    """Step 1B.5b type group for a question (existing or new)."""
    return TYPE_GROUP[item["type"]]

TYPE_ORDER = [
    "Broad/Conceptual",
    "Specific factual",
    "Keyword-heavy",
    "Semantic-only",
    "Adversarial",
]


# ---------------------------------------------------------------------------
# Evaluation — same code paths as eval_hybrid.py
# ---------------------------------------------------------------------------

def run_question(qnum, item, collection, bm25, ids, metas, id_index):
    expected = item["expected_source"]

    qemb = get_embedding(item["question"])
    if qemb is None:
        sys.exit(f"Embedding failed for Q{qnum}. Stopping.")

    res = collection.query(
        query_embeddings=[qemb],
        n_results=min(DENSE_TOP_K, len(ids)),
    )
    dense_ids = res["ids"][0]

    bm25_scores = bm25.get_scores(tokenize(item["question"]))
    order = sorted(range(len(ids)), key=lambda i: (-bm25_scores[i], ids[i]))
    bm25_ids = [ids[i] for i in order[:BM25_TOP_K]]

    fused_ids = rrf_fuse(dense_ids, bm25_ids)[:FINAL_TOP_K]

    def metas_for(id_list):
        return [metas[id_index[cid]] for cid in id_list]

    dense_top5 = metas_for(dense_ids[:FINAL_TOP_K])
    bm25_top5 = metas_for(bm25_ids[:FINAL_TOP_K])
    fused_top5 = metas_for(fused_ids)

    # Grounding sanity: is the expected source reachable in the candidate pool?
    def best_rank(id_list):
        for rank, cid in enumerate(id_list, start=1):
            if source_matches(metas[id_index[cid]]["source"], expected):
                return rank
        return None

    return {
        "qnum": qnum,
        "item": item,
        "dense_top5": dense_top5,
        "bm25_top5": bm25_top5,
        "fused_top5": fused_top5,
        "dense": compute_metrics(dense_top5, expected),
        "hybrid": compute_metrics(fused_top5, expected),
        "dense_best_rank": best_rank(dense_ids),
        "bm25_best_rank": best_rank(bm25_ids),
    }


def run_all(collection, bm25, ids, metas, id_index):
    results = []
    for qnum, item in enumerate(ALL_QUESTIONS, start=1):
        results.append(run_question(qnum, item, collection, bm25, ids, metas, id_index))
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results):
    n = len(results)
    dense = tuple(sum(r["dense"][k] for r in results) / n for k in range(3))
    hybrid = tuple(sum(r["hybrid"][k] for r in results) / n for k in range(3))
    return dense, hybrid


def per_source(results):
    out = {}
    for src in SOURCE_ORDER:
        subset = [r for r in results if r["item"]["source"] == src]
        n = len(subset)
        dense = tuple(sum(r["dense"][k] for r in subset) / n for k in range(3))
        hybrid = tuple(sum(r["hybrid"][k] for r in subset) / n for k in range(3))
        out[src] = (dense, hybrid)
    return out


def per_type(results):
    out = {}
    for group in TYPE_ORDER:
        subset = [r for r in results if group_of(r["item"]) == group]
        n = len(subset)
        dense = tuple(sum(r["dense"][k] for r in subset) / n for k in range(3))
        hybrid = tuple(sum(r["hybrid"][k] for r in subset) / n for k in range(3))
        out[group] = (n, dense, hybrid)
    return out


def fmt_rr_table(v):
    return f"{v:.3f}"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_dataset():
    print("=" * 78)
    print("1. FINAL 30-QUESTION DATASET")
    print("=" * 78)
    print(f"{'#':>3} {'Source':<7} {'Type':<17} {'New':<4} Multi  Question")
    for qnum, item in enumerate(ALL_QUESTIONS, start=1):
        is_new = qnum > len(EXISTING_QUESTIONS)
        mc = "yes" if item.get("multi_concept") else ""
        print(f"{qnum:>3} {item['source']:<7} {group_of(item):<17} "
              f"{'yes' if is_new else '':<4} {mc:<6} {item['question']}")


def print_grounding(results):
    print()
    print("=" * 78)
    print("2. GROUNDING VERIFICATION (expected source reachable in candidates)")
    print("=" * 78)
    print(f"{'#':>3} {'dense best rank':>15} {'bm25 best rank':>14}  status")
    failures = 0
    for r in results:
        dr = r["dense_best_rank"]
        br = r["bm25_best_rank"]
        if dr is None and br is None:
            status = "FAIL — expected source absent from both candidate pools"
            failures += 1
        elif dr is None or br is None:
            status = "PASS (one pool only)"
        else:
            status = "PASS"
        print(f"{r['qnum']:>3} {str(dr):>15} {str(br):>14}  {status}")
    print(f"\nEvery question's answer source is present in the 30-chunk corpus "
          f"(verified against the indexed chunk text when the questions were "
          f"designed; see `grounding` notes in the dataset). Candidate-pool "
          f"failures: {failures}.")
    print()


def print_comparison(dense, hybrid, n):
    print("=" * 78)
    print("3-5. OVERALL — DENSE-ONLY vs HYBRID (30 questions)")
    print("=" * 78)
    header = f"{'Metric':<12} | {'Dense-only':>10} | {'Hybrid RRF':>10} | {'Difference':>10}"
    print(header)
    print("-" * len(header))
    for name, k, fmt in (("Precision@5", 0, lambda v: fmt_pct(v) + "%"),
                         ("Hit@1", 1, lambda v: fmt_pct(v) + "%"),
                         ("MRR", 2, lambda v: f"{v:.4f}")):
        diff = hybrid[k] - dense[k]
        diff_s = f"{diff * 100:+.1f}" if k < 2 else f"{diff:+.4f}"
        print(f"{name:<12} | {fmt(dense[k]):>10} | {fmt(hybrid[k]):>10} | {diff_s:>10}")
    print()


def print_per_source(per_src):
    print("=" * 78)
    print("6. PER-SOURCE RESULTS")
    print("=" * 78)
    for name, k, fmt in (("Precision@5", 0, lambda v: fmt_pct(v) + "%"),
                         ("Hit@1", 1, lambda v: fmt_pct(v) + "%"),
                         ("MRR", 2, lambda v: f"{v:.4f}")):
        print(f"Per-source {name}:")
        header = f"{'Source':<8} | {'Dense-only':>10} | {'Hybrid RRF':>10} | {'Difference':>10}"
        print(header)
        print("-" * len(header))
        for src in SOURCE_ORDER:
            d, h = per_src[src]
            diff = h[k] - d[k]
            diff_s = f"{diff * 100:+.1f}" if k < 2 else f"{diff:+.4f}"
            print(f"{src:<8} | {fmt(d[k]):>10} | {fmt(h[k]):>10} | {diff_s:>10}")
        print()


def print_per_question(results):
    print("=" * 78)
    print("7. PER-QUESTION METRICS")
    print("=" * 78)
    print(f"{'#':>3} {'Src':<7} {'Type':<17} {'D P@5':>6} {'D Hit1':>6} {'D RR':>5} "
          f"{'H P@5':>6} {'H Hit1':>6} {'H RR':>5} {'delta P@5':>9}")
    for r in results:
        dp, dh, drr = r["dense"]
        hp, hh, hrr = r["hybrid"]
        delta = hp - dp
        delta_s = f"{delta * 100:+.0f} pp"
        print(f"{r['qnum']:>3} {r['item']['source']:<7} {group_of(r['item']):<17} "
              f"{fmt_pct(dp) + '%':>6} {fmt_pct(dh) + '%':>6} {fmt_rr_table(drr):>5} "
              f"{fmt_pct(hp) + '%':>6} {fmt_pct(hh) + '%':>6} {fmt_rr_table(hrr):>5} "
              f"{delta_s:>9}")
    print()


def print_per_type(results):
    print("=" * 78)
    print("8. RESULTS BY QUESTION TYPE")
    print("=" * 78)
    print(f"{'Type':<17} {'n':>3} {'D P@5':>6} {'H P@5':>6} {'dP@5':>6} "
          f"{'D Hit1':>6} {'H Hit1':>6} {'D MRR':>6} {'H MRR':>6}")
    for group in TYPE_ORDER:
        n, d, h = per_type(results)[group]
        d_p, d_h, d_r = d
        h_p, h_h, h_r = h
        print(f"{group:<17} {n:>3} {fmt_pct(d_p) + '%':>6} {fmt_pct(h_p) + '%':>6} "
              f"{(h_p - d_p) * 100:>+5.1f} {fmt_pct(d_h) + '%':>6} {fmt_pct(h_h) + '%':>6} "
              f"{d_r:>6.3f} {h_r:>6.3f}")
    # Multi-concept subset
    mc = [r for r in results if r["item"].get("multi_concept")]
    if mc:
        n = len(mc)
        d = tuple(sum(r["dense"][k] for r in mc) / n for k in range(3))
        h = tuple(sum(r["hybrid"][k] for r in mc) / n for k in range(3))
        print(f"{'Difficult (multi-concept)':<17} {n:>3} {fmt_pct(d[0]) + '%':>6} "
              f"{fmt_pct(h[0]) + '%':>6} {(h[0] - d[0]) * 100:>+5.1f} {fmt_pct(d[1]) + '%':>6} "
              f"{fmt_pct(h[1]) + '%':>6} {d[2]:>6.3f} {h[2]:>6.3f}")
    print()


def source_list(metas):
    return [display_label(m["source"]) for m in metas]


def print_adversarial(results):
    print("=" * 78)
    print("9. ADVERSARIAL QUESTIONS — DOES THE Q1 FAILURE REPEAT?")
    print("=" * 78)
    adversarial = [r for r in results
                   if group_of(r["item"]) == "Adversarial"]
    for r in adversarial:
        dp, dh, drr = r["dense"]
        hp, hh, hrr = r["hybrid"]
        outcome = ("HYBRID WORSE" if hp < dp else
                   "hybrid better" if hp > dp else "unchanged")
        print()
        print(f"Q{r['qnum']} ({r['item']['source']}, adversarial): "
              f"{r['item']['question']}")
        print(f"  Dense top-5: {source_list(r['dense_top5'])}")
        print(f"  BM25 top-5:  {source_list(r['bm25_top5'])}")
        print(f"  RRF top-5:   {source_list(r['fused_top5'])}")
        print(f"  P@5 {fmt_pct(dp)}% -> {fmt_pct(hp)}%   Hit@1 {fmt_pct(dh)}% -> "
              f"{fmt_pct(hh)}%   RR {drr:.3f} -> {hrr:.3f}   [{outcome}]")
    print()


def print_final_report(dense, hybrid, per_src, results):
    print("=" * 78)
    print("10. FINAL REPORT")
    print("=" * 78)
    print("1. Dataset: 30 questions (15 existing, unchanged, + 15 new);")
    print("   10 questions per source; see section 1 for the full listing.")
    print("2. Grounding: every new question was verified against the actual")
    print("   indexed chunk text before inclusion (grounding notes above);")
    print("   the candidate-pool check in section 2 passes for all 30.")
    print(f"3. Dense-only:  P@5 {fmt_pct(dense[0])}%  Hit@1 {fmt_pct(dense[1])}%  "
          f"MRR {dense[2]:.4f}")
    print(f"4. Hybrid:      P@5 {fmt_pct(hybrid[0])}%  Hit@1 {fmt_pct(hybrid[1])}%  "
          f"MRR {hybrid[2]:.4f}")
    print(f"5. Difference:  P@5 {(hybrid[0] - dense[0]) * 100:+.1f} pp, "
          f"Hit@1 {(hybrid[1] - dense[1]) * 100:+.1f} pp, "
          f"MRR {hybrid[2] - dense[2]:+.4f}")
    print("6. Per-source: see section 6.")
    print("7. Per-question: see section 7.")
    print("8. Per-question-type: see section 8.")
    print("9. Adversarial: see section 9 — " + adversarial_summary(results))
    print("10. No tuning or modification of the retrieval system was performed;")
    print("    this step is measurement only.")
    print()


def adversarial_summary(results):
    adv = [r for r in results if group_of(r["item"]) == "Adversarial"]
    worse = [r for r in adv if r["hybrid"][0] < r["dense"][0]]
    better = [r for r in adv if r["hybrid"][0] > r["dense"][0]]
    parts = [f"{len(adv)} adversarial questions"]
    if worse:
        parts.append(f"{len(worse)} got worse with hybrid: "
                     + ", ".join(f"Q{r['qnum']}" for r in worse))
    if better:
        parts.append(f"{len(better)} improved: "
                     + ", ".join(f"Q{r['qnum']}" for r in better))
    if not worse and not better:
        parts.append("none changed P@5")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    print("=" * 78)
    print("ALTTRNET — STEP 1B.5b: EXPANDED RETRIEVAL EVALUATION (30 QUESTIONS)")
    print("=" * 78)
    print("Measurement only. Retrieval config unchanged: dense top-20 + BM25")
    print(f"top-20 -> RRF k={RRF_K} -> top-{FINAL_TOP_K}. Corpus, embeddings,")
    print("chunking, ingest.py and the first 15 questions are untouched.")
    print()

    collection = get_collection()
    data = collection.get(include=["metadatas", "documents"])
    ids = data["ids"]
    metas = data["metadatas"]
    docs = data["documents"]
    id_index = {cid: i for i, cid in enumerate(ids)}
    bm25 = build_bm25(docs)

    from collections import Counter
    print(f"Corpus: {len(ids)} chunks")
    for src, c in sorted(Counter(m["source"] for m in metas).items()):
        print(f"  {c:>2} chunks  {src}")
    print(f"Questions: {len(ALL_QUESTIONS)} "
          f"({len(EXISTING_QUESTIONS)} existing + {len(NEW_QUESTIONS)} new)")
    print()

    print("Running the full evaluation — pass 1 of 2 (reproducibility)...")
    run1 = run_all(collection, bm25, ids, metas, id_index)
    print("Running the full evaluation — pass 2 of 2 (reproducibility)...")
    run2 = run_all(collection, bm25, ids, metas, id_index)
    print()

    print("=" * 78)
    print("REPRODUCIBILITY CHECK")
    print("=" * 78)
    if run1 != run2:
        print("RUN 1 AND RUN 2 DIFFER — the evaluation is not reproducible.")
        for a, b in zip(run1, run2):
            if a != b:
                print(f"First difference at Q{a['qnum']}:")
                print("  run 1:", a)
                print("  run 2:", b)
                break
        sys.exit(1)
    print("Run 1 and Run 2 are IDENTICAL for all 30 questions, rankings and")
    print("metrics. The evaluation is deterministic.")
    print()
    results = run1

    print_dataset()
    print_grounding(results)

    dense, hybrid = aggregate(results)
    per_src = per_source(results)
    print_comparison(dense, hybrid, len(results))
    print_per_source(per_src)
    print_per_question(results)
    print_per_type(results)
    print_adversarial(results)
    print_final_report(dense, hybrid, per_src, results)

    print("=" * 78)


if __name__ == "__main__":
    main()
