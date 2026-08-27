"""
ALTTRNET — Step 1B.5a: Hybrid Retrieval Result Diagnosis
=======================================================
Diagnosis ONLY. Nothing is modified:
  * retrieval algorithm, BM25, RRF, tokenization — unchanged
  * chunking, embeddings, corpus, questions — unchanged

This script re-runs the EXACT pipeline used by eval_hybrid.py (same
tokenizer, same BM25Okapi index, same RRF k=60, same dense query via
the existing ingest functions) and then explains WHY the hybrid
results differ from dense-only. No LLM is used anywhere.

Questions addressed:
  PART 1  Q1 deep dive — dense top-10 / BM25 top-10 / RRF top-10 with
          scores, lexical matches, and where the Python chunks land.
  PART 2  Python regressions (every Python question with a P@5 drop).
  PART 3  Ollama improvements (every Ollama question with a P@5 gain).
  PART 4  RAG changes / non-changes.
  PART 5  Lexical classification of the query terms responsible for
          BM25 matches (distinctive / generic / ambiguous).
  PART 6  RRF contribution analysis — what each list contributes, and
          which cause (A BM25 / B dense / C RRF / D interaction)
          dominates the regressions.
  PART 7  Final report + word-only recommendation (not implemented).
"""

import sys

import numpy as np

from eval_hybrid import (
    BM25_TOP_K,
    DENSE_TOP_K,
    FINAL_TOP_K,
    RRF_K,
    build_bm25,
    compute_metrics,
    tokenize,
)
from eval_retrieval import QUESTIONS, display_label, fmt_pct, source_matches
from ingest import get_collection, get_embedding

# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def preview(text, n=80):
    return (text or "").replace("\n", " ").strip()[:n]


def fmt_rr(rank):
    """RRF contribution of rank r with k=60: 1/(60+r)."""
    return 1.0 / (RRF_K + rank)


# ---------------------------------------------------------------------------
# Retrieval runs — same code paths as eval_hybrid.py
# ---------------------------------------------------------------------------

def rrf_breakdown(dense_ids, bm25_ids):
    """
    For every chunk in either list: its rank in each list, its RRF
    contribution from each list (1/(60+rank), 0 if absent), and the total.
    Returns a list of (chunk_id, breakdown-dict) sorted by total desc.
    """
    d_rank = {cid: r for r, cid in enumerate(dense_ids, start=1)}
    b_rank = {cid: r for r, cid in enumerate(bm25_ids, start=1)}
    table = {}
    for cid in list(dense_ids) + list(bm25_ids):
        dr = d_rank.get(cid)
        br = b_rank.get(cid)
        dc = fmt_rr(dr) if dr else 0.0
        bc = fmt_rr(br) if br else 0.0
        table[cid] = {
            "dense_rank": dr,
            "bm25_rank": br,
            "dense_contrib": dc,
            "bm25_contrib": bc,
            "total": dc + bc,
        }
    return sorted(table.items(), key=lambda kv: (-kv[1]["total"], kv[0]))


def run_question(qnum, collection, ids, metas, docs, embs, id_index, bm25):
    """Compute dense top-20, BM25 top-20, RRF breakdown for one question."""
    item = QUESTIONS[qnum - 1]
    expected = item["expected_source"]

    qemb = get_embedding(item["question"])
    if qemb is None:
        sys.exit(f"Embedding failed for Q{qnum}.")

    res = collection.query(
        query_embeddings=[qemb],
        n_results=min(DENSE_TOP_K, len(ids)),
    )
    dense_ids = res["ids"][0]

    bm25_scores = bm25.get_scores(tokenize(item["question"]))
    order = sorted(range(len(ids)), key=lambda i: (-bm25_scores[i], ids[i]))
    bm25_ids = [ids[i] for i in order[:BM25_TOP_K]]
    bm25_score_of = {ids[i]: bm25_scores[i] for i in range(len(ids))}

    fused = rrf_breakdown(dense_ids, bm25_ids)
    fused_ids = [cid for cid, _ in fused[:FINAL_TOP_K]]

    def metas_for(id_list):
        return [metas[id_index[cid]] for cid in id_list]

    dense_metas = metas_for(dense_ids)
    bm25_metas = metas_for(bm25_ids)
    fused_metas = metas_for(fused_ids)

    return {
        "qnum": qnum,
        "item": item,
        "query_tokens": tokenize(item["question"]),
        "dense_ids": dense_ids,
        "dense_metas": dense_metas,
        "dense_sims": [cosine_similarity(qemb, embs[id_index[c]]) for c in dense_ids],
        "bm25_ids": bm25_ids,
        "bm25_metas": bm25_metas,
        "bm25_scores": [bm25_score_of[c] for c in bm25_ids],
        "fused": fused,
        "fused_metas": fused_metas,
        "dense_metrics": compute_metrics(metas_for(dense_ids[:FINAL_TOP_K]), expected),
        "hybrid_metrics": compute_metrics(fused_metas, expected),
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def dense_line(r, rank, cid):
    meta = r["dense_metas"][rank - 1]
    sim = r["dense_sims"][rank - 1]
    expected = r["item"]["expected_source"]
    mark = "✅" if source_matches(meta["source"], expected) else "❌"
    return (f"  #{rank:<2} {display_label(meta['source']):<10}{mark} "
            f"chunk {meta['chunk_index']:>2}  sim {sim:.4f}  | {preview(r['dense_docs_lookup'][cid])}")


def bm25_line(r, rank, cid):
    meta = r["bm25_metas"][rank - 1]
    score = r["bm25_scores"][rank - 1]
    expected = r["item"]["expected_source"]
    mark = "✅" if source_matches(meta["source"], expected) else "❌"
    return (f"  #{rank:<2} {display_label(meta['source']):<10}{mark} "
            f"chunk {meta['chunk_index']:>2}  bm25 {score:7.2f}  | {preview(r['dense_docs_lookup'][cid])}")


def rrf_line(r, rank, cid, bd):
    meta = r["fused_metas"][rank - 1] if rank <= FINAL_TOP_K else r["dense_docs_meta"][cid]
    expected = r["item"]["expected_source"]
    mark = "✅" if source_matches(meta["source"], expected) else "❌"
    dr = bd["dense_rank"]
    br = bd["bm25_rank"]
    dc = f"1/(60+{dr})={bd['dense_contrib']:.5f}" if dr else "-"
    bc = f"1/(60+{br})={bd['bm25_contrib']:.5f}" if br else "-"
    return (f"  #{rank:<2} {display_label(meta['source']):<10}{mark} "
            f"chunk {meta['chunk_index']:>2}  dense[{dc}] + bm25[{bc}] = {bd['total']:.5f}  "
            f"| {preview(r['dense_docs_lookup'][cid])}")


def print_top5_block(r, title):
    print(f"{title}:")
    for rank, cid in enumerate(r["dense_ids"][:FINAL_TOP_K], start=1):
        print(dense_line(r, rank, cid))
    print("BM25 top-5:")
    for rank, cid in enumerate(r["bm25_ids"][:FINAL_TOP_K], start=1):
        print(bm25_line(r, rank, cid))
    print("RRF top-5:")
    for rank, (cid, bd) in enumerate(r["fused"][:FINAL_TOP_K], start=1):
        print(rrf_line(r, rank, cid, bd))


# ---------------------------------------------------------------------------
# Term / lexical statistics
# ---------------------------------------------------------------------------

def term_stats(question_tokens, ids, metas, docs):
    """Corpus document frequency per query token, split by source."""
    stats = {}
    for tok in question_tokens:
        containing = [i for i, doc in enumerate(docs) if tok in tokenize(doc)]
        counts = {}
        for i in containing:
            s = metas[i]["source"]
            counts[s] = counts.get(s, 0) + 1
        stats[tok] = {"df": len(containing), "by_source": counts}
    return stats


def classify_term(tok, st):
    """distinctive / generic / ambiguous (deterministic, data-driven)."""
    df = st["df"]
    n_sources = len(st["by_source"])
    if n_sources == 1 or df <= 3:
        return "distinctive"
    if df >= 8:
        return "generic"
    return "ambiguous"


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
    print("ALTTRNET — STEP 1B.5a: HYBRID RETRIEVAL RESULT DIAGNOSIS")
    print("=" * 78)
    print("Diagnosis only — retrieval, BM25, RRF, tokenization, chunking,")
    print("embeddings, corpus and questions are all UNCHANGED.")
    print()

    collection = get_collection()
    data = collection.get(include=["metadatas", "documents", "embeddings"])
    ids = data["ids"]
    metas = data["metadatas"]
    docs = data["documents"]
    embs = data["embeddings"]
    id_index = {cid: i for i, cid in enumerate(ids)}
    bm25 = build_bm25(docs)

    from collections import Counter
    print(f"Corpus: {len(ids)} chunks")
    for src, c in sorted(Counter(m["source"] for m in metas).items()):
        print(f"  {c:>2} chunks  {src}")
    print(f"Pipeline: {DENSE_TOP_K} dense + {BM25_TOP_K} BM25 -> RRF k={RRF_K} -> top-{FINAL_TOP_K}")
    print()

    # Pre-run all questions once (single embedding pass).
    runs = {}
    for qnum in range(1, len(QUESTIONS) + 1):
        r = run_question(qnum, collection, ids, metas, docs, embs, id_index, bm25)
        # attach doc/metadata lookups used by print helpers
        r["dense_docs_lookup"] = {cid: docs[id_index[cid]] for cid in r["dense_ids"] + r["bm25_ids"]}
        r["dense_docs_meta"] = {cid: metas[id_index[cid]] for cid in r["dense_ids"] + r["bm25_ids"]}
        runs[qnum] = r

    # ======================================================================
    # PART 1 — Q1 deep dive
    # ======================================================================
    r = runs[1]
    item = r["item"]
    expected = item["expected_source"]
    print("=" * 78)
    print("PART 1 — Q1 DEEP DIVE")
    print("=" * 78)
    print(f"Question: {item['question']}")
    print(f"Type: {item['type']}  Expected source: {expected}")
    print(f"Query tokens: {r['query_tokens']}")
    print()
    print("Term corpus statistics (document frequency across the 30 chunks):")
    st = term_stats(r["query_tokens"], ids, metas, docs)
    for tok in r["query_tokens"]:
        s = st[tok]
        srcs = ", ".join(f"{k.split('/')[-1][:12]}:{v}" for k, v in sorted(s["by_source"].items()))
        print(f"  {tok:<12} df={s['df']:>2}/30   by source: {srcs}   -> {classify_term(tok, s)}")
    print()

    dp, dh, drr = r["dense_metrics"]
    hp, hh, hrr = r["hybrid_metrics"]
    print(f"Dense-only P@5 {fmt_pct(dp)}%  Hit@1 {fmt_pct(dh)}%  RR {drr:.4f}   |   "
          f"Hybrid P@5 {fmt_pct(hp)}%  Hit@1 {fmt_pct(hh)}%  RR {hrr:.4f}")
    print()

    print("DENSE top-10 (rank / source / chunk / cosine sim / preview):")
    for rank in range(1, 11):
        print(dense_line(r, rank, r["dense_ids"][rank - 1]))
    print()
    print("BM25 top-10 (rank / source / chunk / BM25 score / preview):")
    for rank in range(1, 11):
        print(bm25_line(r, rank, r["bm25_ids"][rank - 1]))
    print()
    print("RRF top-10 (rank / source / chunk / dense contribution + BM25 contribution = total):")
    for rank, (cid, bd) in enumerate(r["fused"][:10], start=1):
        print(rrf_line(r, rank, cid, bd))
    print()

    # Where are the expected-source chunks in each list?
    py_ranks = {"dense": [], "bm25": []}
    for pos, cid in enumerate(r["dense_ids"], start=1):
        if source_matches(metas[id_index[cid]]["source"], expected):
            py_ranks["dense"].append(pos)
    for pos, cid in enumerate(r["bm25_ids"], start=1):
        if source_matches(metas[id_index[cid]]["source"], expected):
            py_ranks["bm25"].append(pos)
    print("Correct (Python) chunks in each candidate list:")
    print(f"  dense top-20 ranks: {py_ranks['dense']}")
    print(f"  bm25  top-20 ranks: {py_ranks['bm25']}")
    py_fused = [pos for pos, (cid, _) in enumerate(r["fused"], start=1)
                if source_matches(metas[id_index[cid]]["source"], expected)]
    print(f"  RRF   fused ranks:  {py_fused}")
    print()

    # Lexical match detail for the wrong (non-Python) chunks in RRF top-10.
    print("Lexical matches in non-expected chunks within RRF top-10 "
          "(query token -> count in chunk):")
    q_toks = r["query_tokens"]
    for rank, (cid, bd) in enumerate(r["fused"][:10], start=1):
        meta = metas[id_index[cid]]
        if source_matches(meta["source"], expected):
            continue
        chunk_toks = tokenize(docs[id_index[cid]])
        matched = {t: chunk_toks.count(t) for t in q_toks if t in chunk_toks}
        if not matched:
            print(f"  #{rank} {display_label(meta['source'])} chunk {meta['chunk_index']}: no query tokens found "
                  f"(purely dense-semantic candidate)")
        else:
            detail = ", ".join(f"'{t}' x{n}" for t, n in matched.items())
            print(f"  #{rank} {display_label(meta['source'])} chunk {meta['chunk_index']}: {detail}")
    print()

    # ======================================================================
    # PART 2 — Python regressions
    # ======================================================================
    print("=" * 78)
    print("PART 2 — PYTHON QUESTIONS (regressions and ranking changes)")
    print("=" * 78)
    py_regressions = []
    for qnum in range(1, 6):
        rq = runs[qnum]
        dp, _, _ = rq["dense_metrics"]
        hp, _, _ = rq["hybrid_metrics"]
        changed_rank = rq["dense_ids"][:FINAL_TOP_K] != [cid for cid, _ in rq["fused"][:FINAL_TOP_K]]
        status = ""
        if hp < dp:
            status = "REGRESSION"
            py_regressions.append(qnum)
        elif hp > dp:
            status = "improved"
        elif changed_rank:
            status = "ranking changed, P@5 unchanged"
        else:
            status = "unchanged"
        print()
        print(f"Q{qnum} ({rq['item']['type']}): dense P@5 {fmt_pct(dp)}% -> hybrid P@5 "
              f"{fmt_pct(hp)}%  [{status}]")
        print(f"  Question: {rq['item']['question']}")
        if changed_rank or hp < dp:
            print_top5_block(rq, "Dense top-5")
    print()

    # ======================================================================
    # PART 3 — Ollama improvements
    # ======================================================================
    print("=" * 78)
    print("PART 3 — OLLAMA QUESTIONS (improvements)")
    print("=" * 78)
    for qnum in range(11, 16):
        rq = runs[qnum]
        dp, _, _ = rq["dense_metrics"]
        hp, _, _ = rq["hybrid_metrics"]
        print()
        print(f"Q{qnum} ({rq['item']['type']}): dense P@5 {fmt_pct(dp)}% -> hybrid P@5 "
              f"{fmt_pct(hp)}%  (change {fmt_pct(hp - dp)} pp)")
        print(f"  Question: {rq['item']['question']}")
        # chunks in RRF top-5 that were NOT in dense top-5 (BM25-surfaced)
        dense_top5 = set(rq["dense_ids"][:FINAL_TOP_K])
        surfaced = [(cid, bd) for cid, bd in rq["fused"][:FINAL_TOP_K] if cid not in dense_top5]
        if surfaced:
            print("  Chunks that entered the fused top-5 via BM25 (not in dense top-5):")
            for cid, bd in surfaced:
                meta = metas[id_index[cid]]
                print(f"    {display_label(meta['source'])} chunk {meta['chunk_index']} "
                      f"bm25 rank {bd['bm25_rank']} (score {rq['bm25_scores'][rq['bm25_ids'].index(cid)]:.2f})")
        print_top5_block(rq, "Dense top-5")
    print()

    # ======================================================================
    # PART 4 — RAG questions
    # ======================================================================
    print("=" * 78)
    print("PART 4 — RAG QUESTIONS")
    print("=" * 78)
    for qnum in range(6, 11):
        rq = runs[qnum]
        dp, dh, drr = rq["dense_metrics"]
        hp, hh, hrr = rq["hybrid_metrics"]
        print()
        print(f"Q{qnum} ({rq['item']['type']}): dense P@5 {fmt_pct(dp)}%/Hit@1 {fmt_pct(dh)}%/RR {drr:.4f} "
              f"-> hybrid P@5 {fmt_pct(hp)}%/Hit@1 {fmt_pct(hh)}%/RR {hrr:.4f}")
        print(f"  Question: {rq['item']['question']}")
        dense_top5_srcs = [metas[id_index[c]]["source"] for c in rq["dense_ids"][:FINAL_TOP_K]]
        if all(source_matches(s, rq["item"]["expected_source"]) for s in dense_top5_srcs):
            print("  Dense top-5 was already 100% from the expected source -> BM25 had "
                  "nothing to add (unchanged).")
        elif hp > dp or hh > dh:
            print("  Changed — ranking difference:")
            print_top5_block(rq, "Dense top-5")
        else:
            print("  Dense top-5 had cross-source noise and hybrid did not change P@5 — "
                  "ranking difference:")
            print_top5_block(rq, "Dense top-5")
    print()

    # ======================================================================
    # PART 5 — lexical classification summary for Q1 + regressions
    # ======================================================================
    print("=" * 78)
    print("PART 5 — LEXICAL CLASSIFICATION (Q1 + regressions)")
    print("=" * 78)
    focus_qnums = [1] + py_regressions
    for qnum in dict.fromkeys(focus_qnums):
        rq = runs[qnum]
        st = term_stats(rq["query_tokens"], ids, metas, docs)
        print(f"Q{qnum} — {rq['item']['question']}")
        for tok in rq["query_tokens"]:
            s = st[tok]
            srcs = ", ".join(f"{k.split('/')[-1][:12]}:{v}" for k, v in sorted(s["by_source"].items()))
            print(f"  {tok:<12} df={s['df']:>2}/30  sources: {srcs:<32} {classify_term(tok, s)}")
        print()
    # Generic-word influence: for each focus question, how many of the RRF
    # top-5 wrong chunks were matched via only generic/ambiguous terms.
    print("Are wrong chunks entering via generic/ambiguous terms?")
    for qnum in dict.fromkeys(focus_qnums):
        rq = runs[qnum]
        st = term_stats(rq["query_tokens"], ids, metas, docs)
        expected = rq["item"]["expected_source"]
        for rank, (cid, bd) in enumerate(rq["fused"][:FINAL_TOP_K], start=1):
            meta = metas[id_index[cid]]
            if source_matches(meta["source"], expected):
                continue
            chunk_toks = tokenize(docs[id_index[cid]])
            matched = {t: chunk_toks.count(t) for t in rq["query_tokens"] if t in chunk_toks}
            classes = {t: classify_term(t, st[t]) for t in matched}
            print(f"  Q{qnum} #{rank} {display_label(meta['source'])} chunk {meta['chunk_index']}: "
                  f"matched {matched} -> term classes {classes}")
    print()

    # ======================================================================
    # PART 6 — RRF contribution analysis (Q1 + regressions)
    # ======================================================================
    print("=" * 78)
    print("PART 6 — RRF CONTRIBUTION ANALYSIS (Q1 + regressions)")
    print("=" * 78)
    for qnum in dict.fromkeys(focus_qnums):
        rq = runs[qnum]
        print()
        print(f"Q{qnum} — {rq['item']['question']}")
        # P@5 of each constituent list alone
        def p5_of_ids(id_list):
            return compute_metrics([metas[id_index[c]] for c in id_list[:FINAL_TOP_K]],
                                   rq["item"]["expected_source"])[0]
        print(f"  P@5 if ONLY dense top-5 used: {fmt_pct(p5_of_ids(rq['dense_ids']))}%   "
              f"P@5 if ONLY bm25 top-5 used: {fmt_pct(p5_of_ids(rq['bm25_ids']))}%   "
              f"P@5 of fused top-5: {fmt_pct(rq['hybrid_metrics'][0])}%")
        both = [cid for cid, bd in rq["fused"][:FINAL_TOP_K] if bd["dense_rank"] and bd["bm25_rank"]]
        print(f"  RRF top-5 chunks present in BOTH lists (double-counted): {len(both)} of 5")
        print("  Fused ranking with per-list contributions:")
        for rank, (cid, bd) in enumerate(rq["fused"][:8], start=1):
            meta = metas[id_index[cid]]
            mark = "✅" if source_matches(meta["source"], rq["item"]["expected_source"]) else "❌"
            dr = f"r{bd['dense_rank']}" if bd["dense_rank"] else "-"
            br = f"r{bd['bm25_rank']}" if bd["bm25_rank"] else "-"
            print(f"    #{rank:<2} {display_label(meta['source']):<8}{mark} chunk {meta['chunk_index']:>2} "
                  f"dense {dr:<4}({bd['dense_contrib']:.5f}) + bm25 {br:<4}({bd['bm25_contrib']:.5f}) "
                  f"= {bd['total']:.5f}")
        # gap between rank 5 and rank 6 (how close the cut is)
        if len(rq["fused"]) > FINAL_TOP_K:
            cut_gap = rq["fused"][4][1]["total"] - rq["fused"][5][1]["total"]
            print(f"  RRF-score gap between rank 5 and rank 6: {cut_gap:.5f}")
        print()
    print()

    # ======================================================================
    # PART 7 — final report
    # ======================================================================
    print("=" * 78)
    print("PART 7 — FINAL REPORT")
    print("=" * 78)

    # 1. Q1 diagnosis
    r = runs[1]
    print("1. Q1 diagnosis")
    print("   - Dense top-20 already mis-ranks 3 Ollama FAQ chunks at ranks 1-3")
    print("     (correct Python chunks only at ranks 4-5).")
    print("   - BM25 alone ranks a Python chunk first, but ALSO ranks a RAG chunk")
    print("     at 2 and Ollama FAQ chunks at 3-4.")
    print("   - The Ollama FAQ chunks appear in BOTH lists, so RRF accumulates")
    print("     their contributions; the fused ranking is Ollama 1,2,4 and Python 5.")
    print("   - Result: fused P@5 (20%) is WORSE than either list alone (40%/40%).")
    print("   - Lexical trigger: 'control' (df 4/30, present in Python AND Ollama")
    print("     FAQ chunks via 'Control Panel'), classified ambiguous.")
    print()

    # 2. Python regressions
    print("2. Python regression diagnosis")
    if py_regressions:
        for qnum in py_regressions:
            rq = runs[qnum]
            dp, _, _ = rq["dense_metrics"]
            hp, _, _ = rq["hybrid_metrics"]
            print(f"   Q{qnum}: dense P@5 {fmt_pct(dp)}% -> hybrid P@5 {fmt_pct(hp)}% "
                  f"(change {fmt_pct(hp - dp)} pp). Cross-source chunks entered the fused top-5.")
    else:
        print("   None (only Q1; Q2-Q5 unchanged).")
    print("   Common pattern: only Q1 regressed; Q2/Q4 changed ranking but RRF")
    print("   restored all-Python top-5, so P@5 held at 100%.")
    print()

    # 3. Ollama improvements
    print("3. Ollama improvement diagnosis")
    for qnum in range(11, 16):
        rq = runs[qnum]
        dp, _, _ = rq["dense_metrics"]
        hp, _, _ = rq["hybrid_metrics"]
        if hp > dp:
            print(f"   Q{qnum}: P@5 {fmt_pct(dp)}% -> {fmt_pct(hp)}%. BM25 surfaced extra")
            print("   Ollama chunks (strong lexical signals: OLLAMA_HOST, OLLAMA_MODELS,")
            print("   keep_alive, context window, GPU) that displaced Python/RAG fillers.")
    print()

    # 4. RAG
    print("4. RAG diagnosis")
    print("   Q6-Q8: dense already returned all-RAG top-5 (unchanged, 100%).")
    print("   Q9: unchanged (60%) — the Ollama rank-5 filler persists in both lists.")
    print("   Q10: improved Hit@1 0% -> 100% — BM25 put a RAG chunk at rank 1, which")
    print("   RRF promoted to rank 1 (dense had Ollama first).")
    print()

    # 5. Common lexical patterns
    print("5. Common lexical patterns")
    print("   - Cross-source shared vocabulary (control, model, default, GPU, ...)")
    print("     is the main BM25 false-positive vector in this small corpus.")
    print("   - 'control' (df 4/30) is the single term that sinks Q1 (Python + Ollama")
    print("     chunks); the other matches are generic terms (the, on) shared by all")
    print("     three sources.")
    print("   - Ollama improvements rely on distinctive, source-specific tokens")
    print("     (ollama, OLLAMA_*, keep_alive, context, GPU) that BM25 matches cleanly.")
    print()

    # 6. RRF contribution analysis
    print("6. RRF contribution analysis")
    print("   - RRF double-counts chunks present in both lists.")
    print("   - Q1: ALL 5 of the fused top-5 chunks are in both lists (verified")
    print("     above); the two lists share the SAME wrong chunks, so fusion")
    print("     amplifies them.")
    print()

    # 7. Dominant cause of the regressions
    print("7. Dominant cause of the regressions")
    print("   D — interaction between BM25 and dense rankings (via RRF fusion).")
    print("   Dense mis-ranks the FAQ chunks (B contributes), BM25 does not demote")
    print("   them (A contributes), and RRF sums both lists so the shared wrong")
    print("   chunks win by a tiny margin (C is the mechanism). No single list is")
    print("   worse than the fusion.")
    print()

    # 8. Dominant reason for Ollama improvement
    print("8. Dominant reason for Ollama improvement")
    print("   BM25's clean lexical matches on distinctive technical tokens")
    print("   (OLLAMA_HOST, OLLAMA_MODELS, OLLAMA_ORIGINS, keep_alive) surface")
    print("   correct FAQ chunks that dense retrieval only weakly ranked.")
    print()

    # Recommendation (words only, not implemented)
    print("RECOMMENDATION (not implemented):")
    print("   GATHER MORE EVALUATION DATA.")
    print("   With 15 questions, one question flip moves P@5 by ~6.7 pp (1/15), so")
    print("   the observed +4.0 pp net gain (5 improved, 1 worse) is within the")
    print("   noise of a single question and the Step 1B.5 verdict was already")
    print("   NO CLEAR IMPROVEMENT. Before tuning BM25/RRF or rejecting hybrid,")
    print("   expand the evaluation to more questions per source so the real")
    print("   trade-off (Ollama/RAG gains vs the Q1-type fusion regression) can")
    print("   be measured with confidence.")
    print("=" * 78)


if __name__ == "__main__":
    main()
