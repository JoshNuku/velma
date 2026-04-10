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


def _sanitize(history: list[dict]) -> list[dict]:
    """
    Remove messages that would cause API errors:
    - tool-role messages (only valid transiently inside the agentic loop)
    - assistant messages with empty content (incomplete turns from old code)
    """
    clean = []
    for msg in history:
        role = msg.get("role", "")
        if role == "tool":
            continue
        if role == "assistant" and not (msg.get("content") or "").strip():
            continue
        clean.append(msg)
    return clean


def load_history() -> list[dict]:
    """Load conversation history from disk. Returns [] if none exists."""
    _ensure_dir()
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return _sanitize(data)
        except (json.JSONDecodeError, OSError):
            pass  # corrupt file — start fresh
    return []


def save_history(history: list[dict]) -> None:
    """Persist the current conversation history to disk."""
    _ensure_dir()
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(_sanitize(history), f, indent=2, ensure_ascii=False)


def trim_history(history: list[dict], *, max_messages: int = MAX_MESSAGES) -> list[dict]:
    """
    Keep only the most recent *max_messages* entries.

    History only contains user/assistant text messages (tool call context
    is handled transiently in main.py and not persisted).
    """
    if len(history) <= max_messages:
        return history

    trimmed = history[-max_messages:]

    # Always start on a user message so the model has proper context
    while trimmed and trimmed[0].get("role") != "user" and len(trimmed) > 1:
        trimmed = trimmed[1:]

    return trimmed


def clear_history() -> None:
    """Wipe the saved history (start a fresh conversation)."""
    _ensure_dir()
    if MEMORY_FILE.exists():
        MEMORY_FILE.unlink()
