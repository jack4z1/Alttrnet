"""
ALTTRNET — Step 1B.5c: Query-Adaptive Retrieval Diagnostic
=========================================================
DIAGNOSTIC ONLY. Nothing is modified:
  * retrieval system, BM25, RRF, chunking, embeddings — unchanged
  * ingest.py, eval_hybrid.py, eval_hybrid_30.py, chroma_db — untouched
  * NO query router, no adaptive/weighted RRF, no thresholding, no fallback

Purpose: measure deterministic, corpus-derived properties of the 30
evaluation questions and check whether they relate to whether hybrid
(dense + BM25 + RRF) helped, was neutral, or hurt relative to dense-only.

All analysis is deterministic text/statistical computation. No LLM is
used anywhere. The evaluation itself reuses the exact pipeline from
eval_hybrid_30.py (same tokenizer, BM25 index, RRF k=60, dense query)
and is run twice to verify reproducibility.

Analysis IDF (documented, separate from BM25's internal IDF):
    idf(t) = ln( (N + 1) / (df(t) + 0.5) ),   N = 30 chunks
    -> df=0 terms get the maximum idf (ln 62 = 4.127)
    -> df=30 terms get ~0.016
"""

import math
import sys

import numpy as np

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
from eval_hybrid_30 import ALL_QUESTIONS, TYPE_GROUP, group_of
from eval_retrieval import SOURCE_ORDER, display_label, fmt_pct, source_matches
from ingest import get_collection, get_embedding

N_CHUNKS = 30
DISTINCTIVE_DF = 3   # a term is "distinctive" if it occurs in <= 3 chunks

SOURCE_NAMES = {
    "https://docs.python.org/3/tutorial/controlflow.html": "Python",
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation": "RAG",
    "https://github.com/ollama/ollama/blob/main/docs/faq.mdx": "Ollama",
}


def analysis_idf(df):
    """Smooth corpus IDF for analysis (NOT BM25's internal IDF)."""
    return math.log((N_CHUNKS + 1.0) / (df + 0.5))


# ---------------------------------------------------------------------------
# Corpus token statistics
# ---------------------------------------------------------------------------

class CorpusStats:
    def __init__(self, metas, docs):
        self.metas = metas
        self.docs = docs
        self.tok_docs = [tokenize(d) for d in docs]
        self.src_of = [SOURCE_NAMES.get(m["source"], "Other") for m in metas]

        self.df = {}
        self.by_source = {}
        self.in_source = {}
        for tok_set, src in zip(self.tok_docs, self.src_of):
            for tok in set(tok_set):
                self.df[tok] = self.df.get(tok, 0) + 1
                self.by_source.setdefault(tok, {}).setdefault(src, 0)
                self.by_source[tok][src] += 1
                self.in_source.setdefault(tok, set()).add(src)

    def term_stats(self, tok):
        df = self.df.get(tok, 0)
        by_src = self.by_source.get(tok, {})
        return df, by_src


# ---------------------------------------------------------------------------
# Per-question features
# ---------------------------------------------------------------------------

def question_features(qnum, item, cs):
    """Compute deterministic query/corpus features for one question."""
    expected = item["expected_source"]
    exp_name = SOURCE_NAMES.get(expected, "?")
    toks = tokenize(item["question"])
    unique = sorted(set(toks))

    rows = []
    for t in unique:
        df, by_src = cs.term_stats(t)
        idf = analysis_idf(df)
        rows.append({
            "tok": t, "df": df, "idf": idf,
            "by_source": by_src,
            "in_expected": by_src.get(exp_name, 0) > 0,
            "sources": len(by_src),
        })

    n = len(unique)
    in_exp = [r for r in rows if r["in_expected"]]
    in_other = [r for r in rows if not r["in_expected"] and r["df"] > 0]
    only_exp = [r for r in rows if r["in_expected"] and r["sources"] == 1]
    xsrc = [r for r in rows if r["sources"] >= 2]
    distinctive = [r for r in rows if 0 < r["df"] <= DISTINCTIVE_DF]
    df0 = [r for r in rows if r["df"] == 0]
    # source-exclusive anchor: appears in the expected source and nowhere else
    anchors = [r for r in rows if r["in_expected"] and r["sources"] == 1]

    idfs = [r["idf"] for r in rows]
    avg_idf = sum(idfs) / len(idfs) if idfs else 0.0

    # FEATURE 3 — cross-source profile of lexically matchable terms (df > 0)
    matched = [r for r in rows if r["df"] > 0]
    nA = sum(1 for r in matched if r["sources"] == 1 and r["in_expected"])
    nB = sum(1 for r in matched if r["sources"] == 2 and r["in_expected"])
    nC = sum(1 for r in matched if r["sources"] == 3 and r["in_expected"])
    nD = sum(1 for r in matched if not r["in_expected"])
    counts = {"A": nA, "B": nB, "C": nC, "D": nD}
    if not matched:
        profile = "NONE"
    else:
        profile = max(("A", "B", "C", "D"), key=lambda k: counts[k])

    return {
        "qnum": qnum, "item": item,
        "total": len(toks), "unique": n,
        "in_expected": len(in_exp), "in_other": len(in_other),
        "only_expected": len(only_exp),
        "pct_expected": len(in_exp) / n if n else 0.0,
        "pct_multi": len(xsrc) / n if n else 0.0,
        "avg_idf": avg_idf,
        "max_idf": max(idfs) if idfs else 0.0,
        "min_idf": min(idfs) if idfs else 0.0,
        "distinctive_count": len(distinctive),
        "df0_count": len(df0),
        "xsrc_count": len(xsrc),
        "anchor_count": len(anchors),
        "has_anchor": len(anchors) > 0,
        "profile": profile,
        "profile_counts": counts,
        "terms": rows,
    }


# ---------------------------------------------------------------------------
# Retrieval outcome (same pipeline as eval_hybrid_30.py)
# ---------------------------------------------------------------------------

def retrieval_outcome(qnum, item, collection, bm25, ids, metas, id_index):
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

    dense_top5 = metas_for(dense_ids[:FINAL_TOP_K])
    bm25_top5 = metas_for(bm25_ids[:FINAL_TOP_K])
    fused_top5 = metas_for(fused_ids)

    dense = compute_metrics(dense_top5, expected)
    hybrid = compute_metrics(fused_top5, expected)
    delta = hybrid[0] - dense[0]
    if delta > 0:
        outcome = "HELPED"
    elif delta < 0:
        outcome = "HURT"
    else:
        outcome = "NEUTRAL"
    return {
        "dense_top5": [display_label(m["source"]) for m in dense_top5],
        "bm25_top5": [display_label(m["source"]) for m in bm25_top5],
        "fused_top5": [display_label(m["source"]) for m in fused_top5],
        "dense": dense, "hybrid": hybrid, "delta": delta, "outcome": outcome,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def fmt_idf(v):
    return f"{v:.2f}"


def print_main_table(rows):
    print("=" * 100)
    print("FINAL TABLE — ALL 30 QUESTIONS (features vs outcome)")
    print("=" * 100)
    header = (f"{'ID':>3} {'Src':<7} {'Type':<17} {'avgIDF':>7} {'maxIDF':>7} "
              f"{'dist':>4} {'xsrc':>4} {'D P@5':>6} {'H P@5':>6} {'dP@5':>6} {'Outcome':<8}")
    print(header)
    print("-" * len(header))
    for f in rows:
        o = f["outcome"]
        print(f"{f['qnum']:>3} {f['item']['source']:<7} {group_of(f['item']):<17} "
              f"{fmt_idf(f['avg_idf']):>7} {fmt_idf(f['max_idf']):>7} "
              f"{f['distinctive_count']:>4} {f['xsrc_count']:>4} "
              f"{fmt_pct(o['dense'][0]) + '%':>6} {fmt_pct(o['hybrid'][0]) + '%':>6} "
              f"{o['delta'] * 100:>+5.0f} {o['outcome']:<8}")
    print()


def print_feature1_table(rows):
    print("=" * 100)
    print("FEATURE 1/3/4 DETAIL — vocabulary overlap, profile, anchors")
    print("=" * 100)
    print(f"{'ID':>3} {'tot':>4} {'uniq':>4} {'inExp':>5} {'other':>5} {'onlyExp':>7} "
          f"{'%exp':>5} {'%multi':>6} {'df0':>4} {'anch':>5} profile  (A/B/C/D counts)")
    for f in rows:
        pc = f["profile_counts"]
        print(f"{f['qnum']:>3} {f['total']:>4} {f['unique']:>4} {f['in_expected']:>5} "
              f"{f['in_other']:>5} {f['only_expected']:>7} "
              f"{f['pct_expected'] * 100:>4.0f}% {f['pct_multi'] * 100:>5.0f}% "
              f"{f['df0_count']:>4} {f['anchor_count']:>5} {f['profile']:<8} "
              f"(A:{pc['A']} B:{pc['B']} C:{pc['C']} D:{pc['D']})")
    print()
    print("profile: A=matched terms only in expected source; B=expected + 1 other;")
    print("         C=expected + both others; D=only in unrelated sources; NONE=no match")
    print("anch   : # source-exclusive query terms (present only in the expected source)")
    print()


def print_correlations(rows):
    print("=" * 100)
    print("FEATURE 6 — CORRELATION WITH HYBRID MINUS DENSE P@5 (delta)")
    print("=" * 100)
    deltas = np.array([f["outcome"]["delta"] for f in rows])
    features = {
        "avg query IDF": [f["avg_idf"] for f in rows],
        "max query IDF": [f["max_idf"] for f in rows],
        "distinctive-term count (df<=3)": [f["distinctive_count"] for f in rows],
        "cross-source-term count": [f["xsrc_count"] for f in rows],
        "anchor count (source-exclusive)": [f["anchor_count"] for f in rows],
        "pct terms in expected source": [f["pct_expected"] for f in rows],
        "pct terms across multiple sources": [f["pct_multi"] for f in rows],
        "min query IDF": [f["min_idf"] for f in rows],
    }
    for name, vals in features.items():
        r = np.corrcoef(vals, deltas)[0, 1]
        print(f"  {name:<38} r = {r:+.3f}")
    print()
    print("Group means of delta P@5:")
    def split_mean(cond_true, label):
        a = [f["outcome"]["delta"] for f in rows if cond_true(f)]
        b = [f["outcome"]["delta"] for f in rows if not cond_true(f)]
        ma = np.mean(a) * 100 if a else float("nan")
        mb = np.mean(b) * 100 if b else float("nan")
        print(f"  {label:<55} yes: {ma:+5.1f} pp (n={len(a)})   no: {mb:+5.1f} pp (n={len(b)})")
    split_mean(lambda f: f["has_anchor"], "has source-exclusive anchor term")
    split_mean(lambda f: f["distinctive_count"] >= 1, "has >=1 distinctive term (df<=3)")
    med_idf = np.median([f["avg_idf"] for f in rows])
    split_mean(lambda f: f["avg_idf"] >= med_idf, f"avg IDF >= median ({med_idf:.2f})")
    med_x = np.median([f["xsrc_count"] for f in rows])
    split_mean(lambda f: f["xsrc_count"] <= med_x, f"cross-source count <= median ({med_x:.0f})")
    for g in ["Broad/Conceptual", "Specific factual", "Keyword-heavy", "Semantic-only", "Adversarial"]:
        a = [f["outcome"]["delta"] for f in rows if group_of(f["item"]) == g]
        print(f"  type {g:<17} mean delta: {np.mean(a) * 100:+.1f} pp (n={len(a)})")
    print()


def print_criteria(rows):
    print("=" * 100)
    print("CRITERION EVALUATION — simple deterministic rules (analysis only)")
    print("=" * 100)
    non_neutral = [f for f in rows if f["outcome"]["outcome"] != "NEUTRAL"]
    helped = [f for f in non_neutral if f["outcome"]["outcome"] == "HELPED"]
    hurt = [f for f in non_neutral if f["outcome"]["outcome"] == "HURT"]
    print(f"Non-neutral questions: {len(non_neutral)} "
          f"({len(helped)} HELPED, {len(hurt)} HURT). "
          f"NEUTRAL questions count as correct for either choice.")
    print()

    med_idf = np.median([f["avg_idf"] for f in rows])
    med_x = np.median([f["xsrc_count"] for f in rows])

    def evaluate(name, predict_hybrid):
        correct_all = 0
        correct_nn = 0
        wrong = []
        for f in rows:
            o = f["outcome"]
            pred_hybrid = predict_hybrid(f)
            if o["outcome"] == "NEUTRAL":
                correct_all += 1
                continue
            if (o["outcome"] == "HELPED") == pred_hybrid:
                correct_all += 1
                correct_nn += 1
            else:
                wrong.append(f["qnum"])
        acc_all = correct_all / len(rows) * 100
        acc_nn = correct_nn / len(non_neutral) * 100 if non_neutral else 0.0
        print(f"  {name:<58} non-neutral {correct_nn:>2}/{len(non_neutral):>2} "
              f"({acc_nn:>4.0f}%)   all-30 {acc_all:>4.0f}%   misses: "
              f"{', '.join(f'Q{q}' for q in wrong) or '-'}")
        return correct_nn

    print(f"Baselines (n={len(non_neutral)} non-neutral):")
    evaluate("always hybrid", lambda f: True)
    evaluate("always dense-only", lambda f: False)
    print("Candidate criteria:")
    evaluate("hybrid iff has source-exclusive anchor", lambda f: f["has_anchor"])
    evaluate("hybrid iff >=1 distinctive term (df<=3)", lambda f: f["distinctive_count"] >= 1)
    evaluate("hybrid iff cross-source count <= 1", lambda f: f["xsrc_count"] <= 1)
    evaluate("hybrid iff avg IDF >= median", lambda f: f["avg_idf"] >= med_idf)
    evaluate("hybrid iff anchor AND cross-source <= 2", lambda f: f["has_anchor"] and f["xsrc_count"] <= 2)
    evaluate("hybrid iff avg IDF >= median AND xsrc <= median",
             lambda f: f["avg_idf"] >= med_idf and f["xsrc_count"] <= med_x)
    evaluate("hybrid iff (distinctive>=1) AND (xsrc<=2)", lambda f: f["distinctive_count"] >= 1 and f["xsrc_count"] <= 2)
    print()


def term_line(r):
    """Compact per-term detail: tok (df, source:count, ...)."""
    by = ", ".join(f"{s}:{c}" for s, c in sorted(r["by_source"].items()))
    where = "exp-only" if r["in_expected"] and r["sources"] == 1 else \
            ("expected" if r["in_expected"] else "other")
    return f"      {r['tok']:<14} df={r['df']:>2}/30  idf={r['idf']:.2f}  [{by:<22}] {where}"


def print_deep_dive(rows, qnums, title):
    print("=" * 100)
    print(title)
    print("=" * 100)
    for f in rows:
        if f["qnum"] not in qnums:
            continue
        o = f["outcome"]
        print(f"Q{f['qnum']} ({f['item']['source']}, {group_of(f['item'])}): "
              f"{f['item']['question']}")
        print(f"  avgIDF {f['avg_idf']:.2f}  maxIDF {f['max_idf']:.2f}  "
              f"distinctive {f['distinctive_count']}  cross-source {f['xsrc_count']}  "
              f"anchors {f['anchor_count']}  profile {f['profile']}")
        for r in f["terms"]:
            if r["df"] > 0 or r["tok"] in ("ollama", "gpu", "model", "memory"):
                print(term_line(r))
        print(f"  Dense top-5: {o['dense_top5']}")
        print(f"  BM25 top-5:  {o['bm25_top5']}")
        print(f"  RRF top-5:   {o['fused_top5']}")
        d, h = o["dense"], o["hybrid"]
        print(f"  Dense P@5 {fmt_pct(d[0])}%/Hit@1 {fmt_pct(d[1])}%/RR {d[2]:.2f}  "
              f"Hybrid P@5 {fmt_pct(h[0])}%/Hit@1 {fmt_pct(h[1])}%/RR {h[2]:.2f}  "
              f"[{o['outcome']}]")
        print()


def print_final_answers(rows):
    print("=" * 100)
    print("FINAL REPORT — ANSWERS")
    print("=" * 100)
    deltas = [f["outcome"]["delta"] for f in rows]
    helped = [f for f in rows if f["outcome"]["outcome"] == "HELPED"]
    hurt = [f for f in rows if f["outcome"]["outcome"] == "HURT"]
    neutral = [f for f in rows if f["outcome"]["outcome"] == "NEUTRAL"]
    r_avg = np.corrcoef([f["avg_idf"] for f in rows], deltas)[0, 1]
    r_x = np.corrcoef([f["xsrc_count"] for f in rows], deltas)[0, 1]
    r_anch = np.corrcoef([f["anchor_count"] for f in rows], deltas)[0, 1]
    r_dist = np.corrcoef([f["distinctive_count"] for f in rows], deltas)[0, 1]

    print("1. Does query distinctiveness correlate with hybrid improvement?")
    print(f"   NO, and the direction is slightly NEGATIVE: avg IDF r={r_avg:+.3f},")
    print(f"   distinctive-count r={r_dist:+.3f}, anchor-count r={r_anch:+.3f}.")
    print("   The 7 helped questions carry distinctive OLLAMA vocabulary, but the")
    print("   2 hurt questions also carry source-exclusive anchors (Q1: tutorial;")
    print("   Q25: external/internal/material/source), so distinctiveness alone")
    print("   does not separate helped from hurt.")
    print()
    print("2. Does cross-source vocabulary correlate with hybrid regression?")
    print(f"   Essentially no: cross-source count r={r_x:+.3f}. The hurt questions")
    print("   have cross-source terms, but so do helped questions (Q12/Q15/Q26")
    print("   have 12 cross-source terms each). Cross-source overlap is common to")
    print("   nearly all 30 questions and does not discriminate.")
    print()
    print("3. Do adversarial questions have a measurable signature?")
    print("   No single query-text property separates Q1/Q25 (HURT) from Q20/Q30")
    print("   (NEUTRAL). The shared property of the failures is a RETRIEVAL-")
    print("   OUTCOME signature: the wrong source (Ollama) was already in the")
    print("   dense top-5 (Q1 ranks 1-3, Q25 rank 5) AND BM25 also ranked it")
    print("   (Q1 ranks 3-4, Q25 ranks 4-5), so RRF double-counted it. That is")
    print("   overlap between the two candidate lists, not a property of the")
    print("   query text.")
    print()
    print("4. Do keyword-heavy questions have a measurable signature?")
    print("   Weak yes: they carry source-exclusive anchors (OLLAMA_*, sparse/")
    print("   dense vectors) and profile A. BM25 promoted correct chunks when")
    print("   dense was weak (Q13/Q28 80->100); where dense was already 100%")
    print("   (Q3/Q8/Q18) hybrid added nothing. Outcome depends on dense's")
    print("   starting point, not on the anchor presence itself.")
    print()
    print("5. Do semantic-only questions have a measurable signature?")
    print("   Yes: dense already retrieves the source (P@5 80-100%); BM25 adds")
    print("   no P@5 gain (delta 0.0 on all three; Q24's order changed but P@5")
    print("   held). There is no lexical gain available for BM25 to contribute.")
    print()
    print("6. Is there a simple deterministic criterion that predicts hybrid vs")
    print("   dense-only? See the criterion table above. NO query-only rule")
    print("   beats the 'always hybrid' baseline (7/9 on the non-neutral subset).")
    print("   Every one of the 30 questions has >=1 source-exclusive anchor, so")
    print("   anchor-based rules cannot discriminate at all; IDF/cross-source")
    print("   thresholds do worse (avg-IDF rule: 3/9).")
    print()
    print("7. Criterion in plain language:")
    print("   None is justified. The only defensible description is the")
    print("   retrieval-outcome observation: hybrid hurt exactly when dense and")
    print("   BM25 already shared the same wrong chunks, and helped when BM25")
    print("   added correct chunks dense lacked. No query-text feature measured")
    print("   here predicted either condition.")
    print()
    print("8. Verdict on justification:")
    print("   EVIDENCE IS INSUFFICIENT to justify a query router. With 2 HURT /")
    print("   7 HELPED / 21 NEUTRAL and all query-side correlations |r| <= 0.15,")
    print("   no deterministic query criterion can be validated. 'Always hybrid'")
    print("   remains the best measured policy (93% all-30 vs 77% always-dense).")
    print("   A router would add complexity to fix 2 cases while risking the 7")
    print("   gains. More adversarial and generic-vocabulary questions are needed")
    print("   before any adaptive mechanism is justified.")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    print("=" * 100)
    print("ALTTRNET — STEP 1B.5c: QUERY-ADAPTIVE RETRIEVAL DIAGNOSTIC")
    print("=" * 100)
    print("Diagnostic only. No retrieval change, no router, no adaptive logic.")
    print("Deterministic corpus statistics + the exact eval_hybrid_30 pipeline.")
    print()

    collection = get_collection()
    data = collection.get(include=["metadatas", "documents"])
    ids = data["ids"]
    metas = data["metadatas"]
    docs = data["documents"]
    id_index = {cid: i for i, cid in enumerate(ids)}
    bm25 = build_bm25(docs)
    cs = CorpusStats(metas, docs)

    def run_all():
        rows = []
        for qnum, item in enumerate(ALL_QUESTIONS, start=1):
            f = question_features(qnum, item, cs)
            f["outcome"] = retrieval_outcome(qnum, item, collection, bm25,
                                             ids, metas, id_index)
            rows.append(f)
        return rows

    print("Running the full diagnostic — pass 1 of 2 (reproducibility)...")
    run1 = run_all()
    print("Running the full diagnostic — pass 2 of 2 (reproducibility)...")
    run2 = run_all()
    print()
    print("=" * 100)
    print("REPRODUCIBILITY CHECK")
    print("=" * 100)
    if run1 != run2:
        print("RUN 1 AND RUN 2 DIFFER — diagnostic is not reproducible.")
        for a, b in zip(run1, run2):
            if a != b:
                print(f"First difference at Q{a['qnum']}:")
                print("  run 1:", a)
                print("  run 2:", b)
                break
        sys.exit(1)
    print("Run 1 and Run 2 are IDENTICAL. The diagnostic is deterministic.")
    print()
    rows = run1

    print_main_table(rows)
    print_feature1_table(rows)
    print_correlations(rows)
    print_criteria(rows)
    print_deep_dive(rows, [1, 20, 25, 30],
                    "FEATURE 7 — ADVERSARIAL QUESTIONS (Q1, Q20, Q25, Q30)")
    print_deep_dive(rows, [3, 8, 13, 18, 23, 28],
                    "FEATURE 8 — KEYWORD-HEAVY QUESTIONS (Q3, Q8, Q13, Q18, Q23, Q28)")
    print_deep_dive(rows, [19, 24, 29],
                    "FEATURE 9 — SEMANTIC-ONLY QUESTIONS (Q19, Q24, Q29)")
    print_final_answers(rows)


if __name__ == "__main__":
    main()
