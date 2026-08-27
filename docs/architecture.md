# ALTTRNET Architecture — Current State

## Overview

The current repository implements a **RAG (Retrieval-Augmented Generation) prototype**
with validated retrieval experiments. This is the foundation for future model-development work.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│                                                         │
│  ingest.py (interactive)    ingest_markdown.py (batch)  │
└─────────┬────────────────────────────┬──────────────────┘
          │                            │
          ▼                            ▼
┌─────────────────┐          ┌─────────────────────┐
│  Content Source  │          │  Knowledge Root      │
│  (Web URL)       │          │  (Markdown files)    │
└────────┬────────┘          └──────────┬──────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌─────────────────────┐
│  trafilatura     │          │  core.chunker       │
│  (text extract)  │          │  (400/50 chunking)  │
└────────┬────────┘          └──────────┬──────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌─────────────────────┐
│  core.chunker    │          │  core.embedder      │
│  (400/50 words)  │          │  (nomic-embed-text) │
└────────┬────────┘          └──────────┬──────────┘
         │                              │
         ▼                              ▼
┌─────────────────────────────────────────────────┐
│              core.embedder                       │
│         (nomic-embed-text via Ollama)            │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│              core.db                             │
│         (ChromaDB collection: knowledge_base)    │
│         Space: cosine                            │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│              core.retriever                      │
│                                                 │
│  QUERY → Dense top-20 (ChromaDB)                │
│        + BM25 top-20 (rank_bm25)                │
│        → UNION + deduplicate by chunk ID         │
│        → Cross-encoder reranking                 │
│           (ms-marco-MiniLM-L-6-v2)              │
│        → Final top-5                             │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│              LLM (qwen3:14b via Ollama)          │
│         Answer from retrieved context            │
└─────────────────────────────────────────────────┘
```

## Frozen Components

These components are experimentally validated and MUST NOT be modified without
re-running the full evaluation suite:

| Component | Constant | Value | Validated By |
|-----------|----------|-------|--------------|
| Chunk size | `CHUNK_SIZE` | 400 words | Step 1A |
| Chunk overlap | `CHUNK_OVERLAP` | 50 words | Step 1A |
| Embedding model | `EMBED_MODEL` | nomic-embed-text | Step 1A |
| Dense top-K | `DENSE_TOP_K` | 20 | Step 1B.5 |
| BM25 top-K | `BM25_TOP_K` | 20 | Step 1B.5 |
| Final top-K | `FINAL_TOP_K` | 5 | Step 1B.5 |
| Reranker model | `RERANKER_MODEL` | ms-marco-MiniLM-L-6-v2 | Step 1B.6b |
| Collection space | - | cosine | Step 1A |

## Validated Retrieval Architecture

```
Query
  │
  ├──► Dense top-20 (nomic-embed-text → ChromaDB cosine)
  │
  ├──► BM25 top-20 (BM25Okapi over chunk documents)
  │
  └──► UNION + deduplicate by chunk ID
         │
         └──► Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
                │
                └──► Final top-5
```

**Validated results (30 questions):**
- P@5: 96.0%
- Hit@1: 100%
- MRR: 1.0000

## Experiment History

| Step | Experiment | Result |
|------|-----------|--------|
| 1A | Chunking 400/50 + nomic-embed-text | Baseline established |
| 1B.3 | 15-question retrieval evaluation | Dense P@5 83.3% |
| 1B.5 | BM25 + Dense + RRF | P@5 87.3% (+4.0pp) |
| 1B.5b | 30-question expanded evaluation | Confirmed improvements |
| 1B.6a | Reranker candidate recall verification | Correct candidates present |
| 1B.6b | Cross-encoder reranker experiment | P@5 96.0% (ADOPTED) |

## What is NOT Decided

The following are explicitly deferred to the research/architecture layer:

- Attention architecture (MHA, GQA, MQA, etc.)
- MoE architecture
- State-space architecture
- Tokenizer algorithm
- Training loss function
- Optimizer choice
- Data mixture proportions
- Reasoning-training method
- Quantization strategy
- Model depth/width allocation within the 17B cap
