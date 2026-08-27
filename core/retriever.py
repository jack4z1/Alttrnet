"""
core/retriever.py — ALTTRNET reusable production retrieval module
=================================================================
Implements the FROZEN, experimentally-validated retrieval architecture
(Step 1B.6b result: P@5 96.0%, Hit@1 100%, MRR 1.0000):

    QUERY
      -> dense top-20  (nomic-embed-text via Ollama + ChromaDB cosine)
      + BM25 top-20    (BM25Okapi over the collection's chunk documents)
      -> UNION + deduplicate by chunk ID
      -> cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
      -> final top-5

RRF is NOT part of this architecture. Reranking is done by jointly
scoring (query, chunk) pairs with a cross-encoder; ties in the reranker
score are broken by chunk ID so the final ranking is deterministic.

This is the reusable PRODUCTION path, separate from the historical
evaluation harnesses (eval_*.py), which are intentionally left untouched
so they can keep reproducing the validated results.

GPU/CPU behavior: the cross-encoder uses CUDA when torch reports it
available, otherwise falls back to CPU. GPU is never a hard requirement.
The sentence-transformers import is deferred until the reranker is first
used, so this module imports cleanly even where the reranker cannot load.

Reusable by: ingest.py (answering path), future repository ingestion and
any future RAG/answering system.
"""

import re

import torch
from rank_bm25 import BM25Okapi

from core.db import get_collection
from core.embedder import get_embedding

# ---------------------------------------------------------------------------
# Frozen retrieval constants (do NOT tune casually — validated by Step 1B.6b)
# ---------------------------------------------------------------------------

DENSE_TOP_K = 20     # dense candidates per query
BM25_TOP_K = 20      # BM25 candidates per query
FINAL_TOP_K = 5      # final top-5 returned

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Deterministic tokenizer shared by chunks and queries (identical to the
# validated experiment): lowercase -> word-like tokens -> no punctuation.
# No stop words, no stemming, no LLM, no semantic preprocessing.
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase, keep word-like tokens, drop punctuation. Deterministic."""
    return TOKEN_RE.findall((text or "").lower())


def build_bm25(chunk_docs):
    """BM25Okapi index over the chunk documents (k1=1.5, b=0.75 defaults)."""
    return BM25Okapi([tokenize(doc) for doc in chunk_docs])


class Retriever:
    """
    Reusable production retriever implementing the frozen architecture.

    Builds lazily (on first use) the BM25 index and the cross-encoder
    model, so constructing a Retriever is cheap and importable even when
    the reranker backend is unavailable. Call `refresh()` after the
    collection has changed (e.g. new ingestion) to rebuild the index.
    """

    def __init__(self, collection=None, dense_top_k=DENSE_TOP_K,
                 bm25_top_k=BM25_TOP_K, final_top_k=FINAL_TOP_K,
                 reranker_device=None):
        self.collection = collection if collection is not None else get_collection()
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        self.device = reranker_device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._ids = []
        self._metas = []
        self._docs = []
        self._id_index = {}
        self._bm25 = None
        self._reranker_model = None
        self._load_corpus()

    # ------------------------------------------------------------------
    # Corpus / index management
    # ------------------------------------------------------------------

    def _load_corpus(self):
        """Snapshot the collection's chunks (ids, metadata, documents)."""
        data = self.collection.get(include=["metadatas", "documents"])
        self._ids = data["ids"]
        self._metas = data["metadatas"]
        self._docs = data["documents"]
        self._id_index = {cid: i for i, cid in enumerate(self._ids)}

    def refresh(self):
        """Reload the corpus and rebuild the BM25 index after ingestion."""
        self._load_corpus()
        self._bm25 = None
        self._bm25 = build_bm25(self._docs)

    @property
    def count(self):
        """Number of chunks currently indexed."""
        return len(self._ids)

    def _get_bm25(self):
        """BM25 index over the collection chunks, built once and cached."""
        if self._bm25 is None:
            self._bm25 = build_bm25(self._docs)
        return self._bm25

    def _get_reranker(self):
        """Cross-encoder, loaded lazily on first reranking use."""
        if self._reranker_model is None:
            try:
                # Deferred import: keeps this module importable even when the
                # sentence-transformers backend is unavailable.
                from sentence_transformers import CrossEncoder

                print(f"Loading reranker {RERANKER_MODEL} on {self.device} "
                      "(first use)...")
                self._reranker_model = CrossEncoder(RERANKER_MODEL, device=self.device)
            except Exception as exc:
                raise RuntimeError(
                    f"The cross-encoder reranker could not be loaded "
                    f"({RERANKER_MODEL} on {self.device}). The frozen "
                    f"architecture requires it. Underlying error: {exc}"
                ) from exc
        return self._reranker_model

    # ------------------------------------------------------------------
    # Retrieval pipeline
    # ------------------------------------------------------------------

    def search(self, query, rerank=True):
        """
        Run the frozen retrieval pipeline for `query`.

        Returns a list of at most `final_top_k` result records, each:
            {"id": chunk_id,
             "document": chunk text,
             "metadata": chunk metadata dict,
             "score": cross-encoder score or None if rerank=False}
        Ranked by reranker score (highest first), ties broken by chunk ID.
        Returns [] for an empty/invalid query or an empty collection.
        """
        if not query or not query.strip():
            return []
        if self.count == 0:
            print("The knowledge base is empty. Ingest content before retrieving.")
            return []

        # 1) Dense top-k (nomic-embed-text -> ChromaDB cosine).
        qemb = get_embedding(query)
        if qemb is None:
            print("Embedding failed; cannot retrieve.")
            return []

        res = self.collection.query(
            query_embeddings=[qemb],
            n_results=min(self.dense_top_k, self.count),
        )
        dense_ids = res["ids"][0]

        # 2) BM25 top-k (chunk-level index, deterministic tie-break by ID).
        bm25 = self._get_bm25()
        bm25_scores = bm25.get_scores(tokenize(query))
        order = sorted(range(self.count), key=lambda i: (-bm25_scores[i], self._ids[i]))
        bm25_ids = [self._ids[i] for i in order[:self.bm25_top_k]]

        # 3) Union + deduplicate by chunk ID (preserves first-appearance order).
        union_ids = list(dict.fromkeys(dense_ids + bm25_ids))

        candidates = [
            {
                "id": cid,
                "document": self._docs[self._id_index[cid]],
                "metadata": self._metas[self._id_index[cid]],
            }
            for cid in union_ids
        ]

        # 4) Cross-encoder reranking (joint (query, chunk) scoring).
        if rerank:
            model = self._get_reranker()
            pairs = [(query, c["document"]) for c in candidates]
            scores = model.predict(pairs)
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            for c, s in zip(candidates, scores):
                c["score"] = float(s)
            candidates.sort(key=lambda c: (-c["score"], c["id"]))
        else:
            for c in candidates:
                c["score"] = None

        # 5) Final top-5.
        return candidates[:self.final_top_k]