"""
Persistent conversation memory for Velma.

Stores chat history in a JSON file so Velma remembers across restarts.
Also handles trimming to avoid blowing the model's context window.
"""

import json
import os
from pathlib import Path

MEMORY_DIR = Path(__file__).parent / "data"
MEMORY_FILE = MEMORY_DIR / "conversation_history.json"

# Maximum number of messages to keep before summarising/trimming.
# Each user+assistant pair ≈ 2 messages, so 60 → ~30 turns of context.
MAX_MESSAGES = 60


def _ensure_dir() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def load_history() -> list[dict]:
    """Load conversation history from disk. Returns [] if none exists."""
    _ensure_dir()
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass  # corrupt file — start fresh
    return []


def save_history(history: list[dict]) -> None:
    """Persist the current conversation history to disk."""
    _ensure_dir()
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def trim_history(history: list[dict], *, max_messages: int = MAX_MESSAGES) -> list[dict]:
    """
    Keep only the most recent *max_messages* entries.

    Always preserves at least the last user message so the model has context.
    Tool-role messages are kept together with the assistant call that
    triggered them (we never orphan a tool result).
    """
    if len(history) <= max_messages:
        return history

    # Simple strategy: keep the tail.  A smarter version could summarise
    # the dropped segment and prepend a summary message.
    trimmed = history[-max_messages:]

    # Make sure we don't start on a tool or assistant message that references
    # a prior tool_calls message we just dropped.
    while trimmed and trimmed[0].get("role") in ("tool", "assistant") and len(trimmed) > 1:
        trimmed = trimmed[1:]

    return trimmed


def clear_history() -> None:
    """Wipe the saved history (start a fresh conversation)."""
    _ensure_dir()
    if MEMORY_FILE.exists():
        MEMORY_FILE.unlink()
