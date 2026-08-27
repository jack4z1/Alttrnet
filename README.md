# ALTTRNET

**Efficient coding-focused AI with system/model co-design**

Hard constraint: **≤ 17B total parameters**.

## Project Goal

Build an efficient coding AI that punches above its parameter class using:
- Reasoning & chain-of-thought
- Tool use & web research
- External memory (RAG)
- Verification & self-correction
- Inference-time scaling
- Knowledge distillation
- System/model co-design

## Current Status

The repository currently contains a **RAG/retrieval prototype** and validated
experimentation infrastructure. This is the foundation for future model-development work.

### What exists today

| Component | Status | Location |
|-----------|--------|----------|
| Core modules | ✅ Frozen | `core/` |
| Chunking (400/50) | ✅ Frozen | `core/chunker.py` |
| Embedding (nomic-embed-text) | ✅ Frozen | `core/embedder.py` |
| ChromaDB access | ✅ Frozen | `core/db.py` |
| Deterministic IDs | ✅ Frozen | `core/ids.py` |
| Retrieval (dense+BM25+reranker) | ✅ Frozen | `core/retriever.py` |
| Web ingestion | ✅ Working | `ingest.py` |
| Markdown ingestion | ✅ Working | `ingest_markdown.py` |
| Evaluation harnesses | ✅ Working | `eval_*.py` |
| Diagnostics | ✅ Working | `diagnose_*.py` |
| Config system | ✅ New | `core/config.py` |
| Seed management | ✅ New | `core/seeds.py` |
| Logging | ✅ New | `core/logging.py` |
| Experiment metadata | ✅ New | `core/experiment.py` |
| Eval result format | ✅ New | `core/eval_format.py` |
| Test infrastructure | ✅ New | `tests/` |

### What is NOT here yet

- Model architecture (no attention/MoE/SSA choices made)
- Tokenizer
- Training pipeline
- Data mixture configuration
- Training loss / optimizer selection
- Quantization strategy
- These decisions are deferred to the research/architecture layer.

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Pull required Ollama models
ollama pull nomic-embed-text
ollama pull qwen3:14b

# Ingest markdown knowledge base
python ingest_markdown.py

# Ingest a web page
python ingest.py "https://example.com/article"

# Run tests
pytest tests/ -v
```

## Repository Structure

```
alttrnet/
├── core/                  # Core infrastructure (frozen + new foundation)
│   ├── config.py          # Central configuration
│   ├── chunker.py         # Frozen chunking (400/50)
│   ├── db.py              # ChromaDB access
│   ├── embedder.py        # Embedding interface
│   ├── ids.py             # Deterministic IDs
│   ├── retriever.py       # Production retrieval
│   ├── seeds.py           # Reproducibility utilities
│   ├── logging.py         # Structured logging
│   ├── experiment.py      # Experiment metadata
│   └── eval_format.py     # Evaluation result format
├── ingest.py              # Web URL ingestion + interactive Q&A
├── ingest_markdown.py     # Markdown knowledge base ingestion
├── knowledge/             # Markdown documents for ingestion
├── chroma_db/             # Persistent vector database
├── tests/                 # Test suite
├── configs/               # Configuration files (future)
├── experiments/           # Experiment scripts (future)
├── scripts/               # Utility scripts (future)
├── evaluation/            # Evaluation harnesses (future)
├── docs/                  # Documentation
├── checkpoints/           # Model checkpoints (future)
├── artifacts/             # Experiment outputs
├── pyproject.toml         # Project metadata
└── requirements.txt       # Dependencies
```

## Configuration

All project constants are centralized in `core/config.py`:

```python
from core.config import PROJECT, PATHS, MODELS, CHUNKING, RETRIEVAL

print(PROJECT.max_parameters)  # 17B
print(MODELS.embed)            # "nomic-embed-text"
print(CHUNKING.size)           # 400
print(RETRIEVAL.final_top_k)   # 5
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_chunker.py -v

# Run with coverage
pytest tests/ --cov=core --cov-report=term-missing
```

## Design Principles

1. **Frozen components are frozen.** The chunking, embedding, and retrieval
   parameters have been experimentally validated. Do not modify without
   re-running the full evaluation suite.

2. **Metadata-only evaluation.** Retrieval quality is judged by chunk source
   metadata, not by LLM output. This prevents good LLMs from masking bad retrieval.

3. **Deterministic by default.** IDs, chunk ordering, and evaluation results
   are all deterministic. Seeds are managed centrally.

4. **Measure, then change.** Every retrieval change is measured against
   validated baselines before adoption.
