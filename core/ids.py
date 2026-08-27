"""
core/ids.py — ALTTRNET deterministic identifiers & path normalization
=====================================================================
All knowledge-base identities are derived here so that no two scripts
can disagree on how paths are normalized or IDs are generated.

Conventions (Step 2A, complete spec):

    * canonical file key = POSIX-style RELATIVE path from the knowledge
      root, e.g. "python/control_flow.md"  (portable — never absolute)
    * chunk ID (exact spec formula):
          make_chunk_id(file_path, chunk_index) =
              md5(f"{normalized}::{chunk_index}")        # hex digest
      where normalized = file_path.replace("\\", "/").lower()
    * document ID / source ID (file sources: source == document == file):
          make_document_id(file_path) = md5("doc:" + normalized)
    * url for a file source = "file://" + file_path

Two different files containing identical text therefore always receive
different chunk IDs (the path is part of the hash). MD5 is used ONLY as
a deterministic ID generator, never for security or integrity.
"""

import hashlib
import re
from pathlib import Path, PurePosixPath

FILE_SCHEME = "file://"


def normalize_doc_path(doc_path, root=None):
    """
    Return the canonical POSIX RELATIVE path for a document.

    When `root` is given and the document lives under it, the result is
    the relative path from `root` (e.g. "python/control_flow.md").
    Backslashes become forward slashes and leading "./" or "/" are
    stripped, so the same file always maps to the same key.
    """
    p = Path(doc_path)
    if root is not None:
        try:
            rel = p.absolute().relative_to(Path(root).absolute())
        except ValueError:
            rel = p
    else:
        rel = p

    norm = PurePosixPath(rel).as_posix().lstrip("./").lstrip("/")
    return norm if norm else "unknown"


def make_chunk_id(file_path, chunk_index):
    """
    Deterministic, stable chunk ID (spec-mandated formula).

    Uses the normalized (backslash->slash, lowercased) relative file
    path plus the chunk index, so IDs survive re-ingestion and differ
    between files with identical content.
    """
    normalized = file_path.replace("\\", "/").lower()
    return hashlib.md5(f"{normalized}::{chunk_index}".encode()).hexdigest()


def make_document_id(file_path):
    """Deterministic document ID for a file (file sources: document id
    and source id are the same file)."""
    normalized = file_path.replace("\\", "/").lower()
    return hashlib.md5(("doc:" + normalized).encode()).hexdigest()


def url_for_file(file_path):
    """url metadata value for a file-based source."""
    return FILE_SCHEME + file_path


_TITLE_RE = re.compile(r"^\s*#\s+(.+?)\s*$")


def title_from_markdown(text, fallback):
    """
    Derive a document title from the first level-1 Markdown heading
    (the first line that starts with '# '). Returns `fallback` (e.g.
    the file stem) when no heading exists. Heading-aware CHUNKING is
    NOT implemented; this is only a metadata convenience.
    """
    for line in (text or "").splitlines():
        m = _TITLE_RE.match(line)
        if m:
            return m.group(1).strip() or fallback
    return fallback
