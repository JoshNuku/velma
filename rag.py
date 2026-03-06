"""
RAG (Retrieval-Augmented Generation) module for Velma.

Performs semantic search over local notes (~/Documents/Notes) using
Ollama embeddings.  If ChromaDB is available it uses a persistent vector
store; otherwise it falls back to a lightweight in-memory cosine-similarity
search so Velma still works on Python versions where ChromaDB isn't
supported yet (e.g. 3.14).
"""

import json
import math
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NOTES_DIR = Path(os.path.expanduser("~/Documents/Notes"))
CACHE_DIR = Path(__file__).parent / "data"
CACHE_FILE = CACHE_DIR / "note_embeddings.json"
EMBEDDING_MODEL = "nomic-embed-text"  # pull with `ollama pull nomic-embed-text`
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50
TOP_K = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def _embed(texts: list[str]) -> list[list[float]]:
    """Get embeddings from Ollama for a batch of texts."""
    import ollama
    results = []
    for text in texts:
        resp = ollama.embed(model=EMBEDDING_MODEL, input=text)
        # ollama.embed returns {"embeddings": [[...]]}
        results.append(resp["embeddings"][0])
    return results


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# In-memory vector store (JSON-backed)
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    """Load the cached embeddings from disk."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"chunks": [], "embeddings": [], "metadata": []}


def _save_cache(data: dict) -> None:
    """Persist embeddings cache to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_notes() -> str:
    """
    Scan ~/Documents/Notes for .txt and .md files, chunk them, embed them,
    and cache the results to disk.  Safe to run repeatedly.

    Returns:
        A status message with the number of files and chunks indexed.
    """
    if not NOTES_DIR.is_dir():
        return f"Notes directory '{NOTES_DIR}' not found. Please create it first."

    all_chunks: list[str] = []
    all_meta: list[dict] = []
    total_files = 0

    for file in sorted(NOTES_DIR.iterdir()):
        if file.suffix.lower() not in (".txt", ".md"):
            continue
        try:
            content = file.read_text(encoding="utf-8")
        except Exception:
            continue

        chunks = _chunk_text(content)
        if not chunks:
            continue

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_meta.append({"source": file.name, "chunk_index": i})
        total_files += 1

    if not all_chunks:
        return f"No note files found in '{NOTES_DIR}'."

    # Embed everything
    try:
        embeddings = _embed(all_chunks)
    except Exception as e:
        return (
            f"Found {total_files} files ({len(all_chunks)} chunks) but embedding "
            f"failed: {e}\nMake sure '{EMBEDDING_MODEL}' is pulled in Ollama."
        )

    _save_cache({
        "chunks": all_chunks,
        "embeddings": embeddings,
        "metadata": all_meta,
    })

    return f"Indexed {total_files} files ({len(all_chunks)} chunks) into the vector store."


def query_notes(query: str, top_k: int = TOP_K) -> str:
    """
    Perform semantic search over indexed notes.

    Args:
        query: Natural-language search query.
        top_k: Number of results to return.

    Returns:
        Formatted results with source filenames and matching excerpts.
    """
    cache = _load_cache()
    if not cache["chunks"]:
        # Auto-index if the store is empty
        status = index_notes()
        cache = _load_cache()
        if not cache["chunks"]:
            return f"No notes indexed yet. {status}"

    # Embed the query
    try:
        query_emb = _embed([query])[0]
    except Exception as e:
        return f"Failed to embed query: {e}"

    # Compute similarities
    scored = []
    for i, emb in enumerate(cache["embeddings"]):
        sim = _cosine_similarity(query_emb, emb)
        scored.append((sim, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    if not top:
        return "No semantically matching notes found."

    lines = []
    for rank, (sim, idx) in enumerate(top, 1):
        meta = cache["metadata"][idx]
        source = meta.get("source", "unknown")
        chunk = cache["chunks"][idx]
        preview = chunk[:300].replace("\n", " ")
        if len(chunk) > 300:
            preview += "..."
        lines.append(f"{rank}. **{source}** (score: {sim:.2f})\n   {preview}")

    return "\n\n".join(lines)


def reindex_notes() -> str:
    """Delete the existing index and rebuild from scratch."""
    if CACHE_FILE.exists():
        try:
            CACHE_FILE.unlink()
        except Exception as e:
            return f"Failed to clear cache: {e}"
    return index_notes()
