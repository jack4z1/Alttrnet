# ALTTRNET knowledge/ — Markdown knowledge base layout

This directory holds the Markdown documents ingested by
`python ingest_markdown.py`.

Layout is free-form; documents are discovered recursively, so
sub-directories can organize documents by topic or source:

    python/     Python documentation
    ollama/     Ollama documentation
    rag/        Retrieval-augmented generation notes
    skills/     Skill/guide documents

Rules:

- Only `*.md` files are ingested.
- Files whose name starts with `_` (like this one) or `.` are skipped.
- Documents are treated as text: headings, code blocks, URLs and
  terminology are preserved by ingestion.
- Re-ingesting a file replaces its chunks; if a file changes, its old
  chunks are removed automatically.

The files currently present are SAMPLE documents used to test the
pipeline. Replace them with the real documents when ready.
