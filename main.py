"""
Velma — master agent module.

Handles LLM interaction, tool dispatch, conversation memory,
and context-window trimming.
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tools import (
    ALL_TOOLS,
    ALL_TOOL_SCHEMAS,
    get_current_time,
    list_reminders,
    schedule_reminder,
    send_email,
)
from memory import load_history, save_history, trim_history, clear_history

load_dotenv()
client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    # --- Identity ---
    "Your name is Velma. You are Josh's personal AI desktop assistant — not a generic chat model. "
    "Josh built you. You are not made by Google, OpenAI, Meta, Alibaba, or anyone else. "
    "If asked who made you: 'Josh built me.' "
    "If asked what model you run on: 'Gemini under the hood, but Josh made me who I am.'\n\n"

    # --- About Josh ---
    "Your user is Josh — a Computer and Electrical Engineer, final-year student at UG. "
    "He's building an SSVEP-based BCI for brain-to-speech as his capstone project. "
    "His specialties: embedded systems, Learning PCB design (KiCad), software development, AI/ML, and IoT. "
    "Outside engineering he's into DC Comics, tennis, and American culture. "
    "Remember these — reference them naturally when relevant, never robotically.\n\n"

    # --- Environment ---
    "Environment:\n"
    "• OS: Windows. You run on this machine.\n"
    "• School data and notes: ~/Documents\n"
    "• Projects and downloads: ~/Downloads\n"
    "• Personal notes: ~/Documents/Notes\n\n"

    # --- Personality ---
    "Personality:\n"
    "• Warm but not sappy. You genuinely care about Josh — show it through helpfulness, not empty pleasantries.\n"
    "• Funny. Lean into witty remarks, playful sarcasm, pop-culture quips (especially DC Comics), "
    "and the occasional pun. Humor should feel natural, not forced — sprinkle it in, don't overdo it.\n"
    "• Concise and clear. No corporate speak, no filler, no 'As an AI...'. "
    "But a well-placed joke or kind word is never filler.\n"
    "• Technically sharp when the topic calls for it. Josh can handle jargon — use it. "
    "But explain things with personality, not like a textbook.\n"
    "• Proactive — if you spot something useful (high disk usage, a simpler approach, a gotcha), mention it. "
    "Bonus points if you make it entertaining.\n"
    "• Honest — if you don't know, say so. Never fabricate facts or hallucinate references. "
    "It's okay to joke about not knowing.\n"
    "• Encouraging — when Josh accomplishes something or has a clever idea, acknowledge it. "
    "A little hype from your AI assistant never hurt anyone.\n\n"

    # --- Tool use (CRITICAL) ---
    "TOOL USE — THESE RULES ARE NON-NEGOTIABLE:\n"
    "1. EXECUTE, don't narrate. When Josh asks you to do something, call the tool IMMEDIATELY. "
    "Never say 'I'm about to...', 'I will now...', 'Standing by...', or 'Executing now...' "
    "without ACTUALLY calling the tool in that same response.\n"
    "2. NEVER claim you did something unless the tool result confirms it. "
    "If a tool returns an error, tell Josh clearly — don't pretend it worked.\n"
    "3. VERIFY after write operations. If you used write_file, check the result message "
    "for the byte count. If it says 0 bytes, the write FAILED — say so and retry.\n"
    "4. ONE attempt narrative. Don't repeat yourself across turns saying 'this time for real'. "
    "If something failed, diagnose WHY, then fix it in one clean attempt.\n"
    "5. After a tool runs, summarize the result in natural language. Never dump raw JSON at Josh.\n\n"

    "Tool routing:\n"
    "• 'open/launch/start X' → launch_application (handles apps, file paths, folders, URLs). "
    "Common names: 'vscode'/'code', 'chrome', 'kicad', 'spotify', etc.\n"
    "• System health / CPU / RAM / disk / stats → get_system_stats.\n"
    "• 'remind me to X in Y minutes' or 'remind me at H:MM' → schedule_reminder. Reminders are email-backed when SMTP is configured, with an in-app fallback.\n"
    "• 'list/show reminders' → list_reminders.\n"
    "• Email reminders → schedule_email_reminder.\n"
    "• 'send email to X' or immediate email requests → collect recipient + subject + body, show a draft, and only send after explicit confirmation.\n"
    "• Random 'Velma feelings' emails → configure_whimsy_emails and list_email_jobs (shows whether whimsy is enabled).\n"
    "• 'list/show/what's in <dir>' → list_directory. Aliases: "
    "'documents' = ~/Documents, 'desktop' = ~/Desktop, 'downloads' = ~/Downloads, "
    "'notes' = ~/Documents/Notes, 'home' = ~.\n"
    "• Clear temp/cache → clear_temp_files.\n"
    "• Search notes by keyword → search_notes. By meaning/concept → semantic_search_notes.\n"
    "• Read/write files → read_file / write_file.\n"
    "• Rename/move a file → rename_file (ALWAYS use this, never 'ren' or 'mv' shell commands). "
    "old_path MUST be the full absolute path (e.g. C:\\\\Users\\\\Josh\\\\Documents\\\\folder\\\\file.png).\n"
    "• Run a command → run_shell_command.\n"
    "• Look something up online → web_search.\n"
    "• Sort files by extension → organize_files.\n\n"

    # --- Time awareness ---
    "If you need to reference the current time or date for any reason (e.g., answering a time-related question, making a quip about the time of day, or checking the time for reminders), ALWAYS call the get_current_time tool instead of guessing or using your own clock. Use its output directly in your response.\n\n"

    # --- Formatting ---
    "Formatting:\n"
    "• Default to 1-3 sentences. Expand only when Josh asks for detail or the answer demands it.\n"
    "• Never list your capabilities unprompted.\n"
    "• Use markdown well — headers (##, ###) for sections, **bold** for emphasis, "
    "bullet/numbered lists for structure, fenced code blocks with language tags.\n"
    "• Math & equations: use LaTeX with KaTeX syntax. Inline: $x^2$. "
    "Display/block equations: use $$ on its own line before and after the expression. "
    "Never put display equations on the same line as $$. Example:\n"
    "$$\n\\frac{6}{(s+1)(s+2)}\n$$\n"
    "• For technical/engineering answers, use a clear hierarchy: "
    "a short intro, then organized sections with headers or bold labels, "
    "and display math for key equations.\n"
    "• System stats: clean compact list, no prose.\n"
    "• Hard cap: ~150 words for casual answers. Technical/math answers can go longer as needed."
)


def _resolve_system_prompt() -> str:
    """
    Allows overriding the built-in system prompt from environment.
    Use escaped newlines (\n) in .env and they will be expanded.
    """
    override = os.environ.get("VELMA_SYSTEM_PROMPT", "").strip()
    if not override:
        return DEFAULT_SYSTEM_PROMPT
    return override.replace("\\n", "\n")


SYSTEM_PROMPT = _resolve_system_prompt()

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL = "gemini-3.1-flash-lite-preview"
_pending_email_draft: dict | None = None
EMAIL_JOBS_FILE = Path(__file__).parent / "data" / "email_jobs.json"


def _append_email_guard_audit(event: dict) -> None:
    """Append an audit event to email_jobs.json without breaking existing state."""
    try:
        EMAIL_JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)

        if EMAIL_JOBS_FILE.exists():
            with open(EMAIL_JOBS_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            if not isinstance(state, dict):
                state = {}
        else:
            state = {}

        audit = state.get("audit", [])
        if not isinstance(audit, list):
            audit = []

        event_copy = dict(event)
        event_copy["timestamp"] = datetime.now().isoformat(timespec="seconds")
        audit.append(event_copy)

        # Keep audit log bounded to avoid unbounded growth.
        state["audit"] = audit[-100:]

        with open(EMAIL_JOBS_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception:
        # Audit should never interrupt the chat flow.
        pass


def _is_direct_time_or_date_question(text: str) -> bool:
    """
    Detect simple user requests asking for the current time or date.
    These are handled deterministically to avoid hallucinated clocks.
    """
    if not text:
        return False

    q = text.strip().lower()
    patterns = [
        r"\bwhat(?:'s| is)?\s+the\s+time\b",
        r"\bwhat\s+time\s+is\s+it\b",
        r"\bcurrent\s+time\b",
        r"\btime\s+now\b",
        r"\bwhat(?:'s| is)?\s+the\s+date\b",
        r"\bwhat\s+day\s+is\s+it\b",
        r"\btoday(?:'s| is)?\s+date\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _normalize_clock_time(time_str: str) -> str:
    """Normalize times like '1.07pm' -> '1:07 PM' for scheduler compatibility."""
    cleaned = re.sub(r"\s+", "", time_str.strip().lower())
    cleaned = cleaned.replace(".", ":")
    m = re.match(r"^(\d{1,2})(?::?(\d{2}))?(am|pm)?$", cleaned)
    if not m:
        return time_str.strip()

    hour = int(m.group(1))
    minute = m.group(2) or "00"
    suffix = m.group(3)

    if suffix:
        return f"{hour}:{minute} {suffix.upper()}"
    return f"{hour}:{minute}"


def _parse_direct_reminder_request(text: str) -> dict | None:
    """
    Parse common reminder intents so they can be executed deterministically.
    Returns one of:
      {"action": "list"}
      {"action": "schedule", "description": str, "minutes": int, "time_str": str}
    """
    if not text:
        return None

    raw = text.strip()
    q = raw.lower()

    if re.search(r"\b(list|show|what(?:'s| is))\b.*\b(reminders?|pending)\b", q):
        return {"action": "list"}

    if "remind" not in q and "reminder" not in q:
        return None

    in_minutes = re.search(
        r"(?:set\s+a\s+)?(?:remind(?:\s+me)?|reminder)(?:\s+to)?\s+(.+?)\s+in\s+(\d+)\s*(?:minutes?|mins?|m)\b",
        q,
        re.IGNORECASE,
    )
    if in_minutes:
        return {
            "action": "schedule",
            "description": in_minutes.group(1).strip(),
            "minutes": int(in_minutes.group(2)),
            "time_str": "",
        }

    at_time = re.search(
        r"(?:set\s+a\s+)?(?:remind(?:\s+me)?|reminder)(?:\s+to)?\s+(.+?)\s+at\s+([0-2]?\d(?:[:.]?\d{2})?\s*(?:am|pm)?)\b",
        q,
        re.IGNORECASE,
    )
    if at_time:
        return {
            "action": "schedule",
            "description": at_time.group(1).strip(),
            "minutes": 0,
            "time_str": _normalize_clock_time(at_time.group(2)),
        }

    for_time_then_desc = re.search(
        r"(?:set\s+a\s+)?reminder\s+for\s+([0-2]?\d(?:[:.]?\d{2})?\s*(?:am|pm)?)\s+(?:to\s+)?(.+)$",
        q,
        re.IGNORECASE,
    )
    if for_time_then_desc:
        return {
            "action": "schedule",
            "description": for_time_then_desc.group(2).strip(),
            "minutes": 0,
            "time_str": _normalize_clock_time(for_time_then_desc.group(1)),
        }

    return None


def _extract_email_fields(text: str) -> dict:
    """Extract recipient/subject/body fields from a free-form email request."""
    raw = text.strip()

    recipient_match = re.search(r"\bto\s+([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", raw, re.IGNORECASE)
    any_email_match = re.search(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", raw)
    recipient = ""
    if recipient_match:
        recipient = recipient_match.group(1).strip()
    elif any_email_match:
        recipient = any_email_match.group(1).strip()

    subject_match = re.search(
        r"\bsubject\s*[:=]\s*(.+?)(?=\s+\b(?:body|message|content)\s*[:=]|$)",
        raw,
        re.IGNORECASE,
    )
    body_match = re.search(r"\b(?:body|message|content)\s*[:=]\s*(.+)$", raw, re.IGNORECASE)

    return {
        "recipient": recipient,
        "subject": (subject_match.group(1).strip() if subject_match else ""),
        "body": (body_match.group(1).strip() if body_match else ""),
    }


def _is_email_send_intent(text: str) -> bool:
    """Detect intent to send an immediate email (not a reminder email)."""
    if not text:
        return False

    q = text.strip().lower()
    if "reminder" in q:
        return False
    if re.search(r"\bsend\b.*\bemail\b", q):
        return True
    if re.search(r"\bemail\b.*\b(to|subject|body|message|content)\b", q):
        return True
    if re.search(r"\bmail\b.*\bto\b", q):
        return True
    return False


def _email_confirmation_summary(draft: dict) -> str:
    recipient = draft.get("recipient", "") or "(missing)"
    subject = draft.get("subject", "") or "(missing)"
    body = draft.get("body", "") or "(missing)"
    return (
        "Got it. Here is the draft so far:\n"
        f"- To: {recipient}\n"
        f"- Subject: {subject}\n"
        f"- Body: {body}\n\n"
        "Say 'confirm email' to send it, or 'cancel email' to stop."
    )


def _natural_missing_email_details(draft: dict) -> str:
    """Return a conversational prompt asking for whichever email fields are missing."""
    missing = [k for k in ("recipient", "subject", "body") if not draft.get(k, "").strip()]
    if not missing:
        return _email_confirmation_summary(draft)

    labels = {
        "recipient": "who to send it to",
        "subject": "the subject line",
        "body": "the message body",
    }
    readable = ", ".join(labels[m] for m in missing)
    return (
        "Sure. I can send that. "
        f"I still need {readable}. "
        "You can reply naturally, for example: to hayet@example.com, subject: ..., body: ..."
    )


def _handle_email_confirmation_flow(user_prompt: str) -> str | None:
    """Require explicit confirmation before sending any immediate email."""
    global _pending_email_draft

    raw = (user_prompt or "").strip()
    q = raw.lower()

    if _pending_email_draft is not None:
        if re.search(r"\bcancel\s+email\b|\bcancel\b|\bnever mind\b", q):
            _pending_email_draft = None
            return "Email send canceled."

        updates = _extract_email_fields(raw)
        changed = False
        for key in ("recipient", "subject", "body"):
            if updates.get(key):
                _pending_email_draft[key] = updates[key]
                changed = True

        if re.search(r"\bconfirm\s+email\b|\bconfirm\b", q):
            missing = [k for k in ("recipient", "subject", "body") if not _pending_email_draft.get(k, "").strip()]
            if missing:
                return _natural_missing_email_details(_pending_email_draft)

            result = send_email(
                recipient=_pending_email_draft["recipient"],
                subject=_pending_email_draft["subject"],
                body=_pending_email_draft["body"],
            )
            if result.startswith("Sent email"):
                _pending_email_draft = None
            return result

        if changed:
            return _email_confirmation_summary(_pending_email_draft)

        return _email_confirmation_summary(_pending_email_draft)

    if not _is_email_send_intent(raw):
        return None

    _pending_email_draft = _extract_email_fields(raw)
    missing = [k for k in ("recipient", "subject", "body") if not _pending_email_draft.get(k, "").strip()]
    if missing:
        return _natural_missing_email_details(_pending_email_draft)
    return _email_confirmation_summary(_pending_email_draft)


def _run_deterministic_shortcuts(user_prompt: str) -> str | None:
    """Execute deterministic, high-confidence intents before model routing."""
    email_flow_reply = _handle_email_confirmation_flow(user_prompt)
    if email_flow_reply is not None:
        return email_flow_reply

    if _is_direct_time_or_date_question(user_prompt):
        return f"It's {get_current_time()}."

    reminder_cmd = _parse_direct_reminder_request(user_prompt)
    if not reminder_cmd:
        return None

    if reminder_cmd["action"] == "list":
        return list_reminders()

    return schedule_reminder(
        description=reminder_cmd["description"],
        minutes=reminder_cmd["minutes"],
        time_str=reminder_cmd["time_str"],
    )


def _build_messages(history: list[dict]) -> list[dict]:
    """Prepend system prompt to conversation history."""
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------
conversation_history: list[dict] = load_history()


def velma_greeting() -> str:
    """Generate a short opening line from Velma when she starts up."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "You are just starting up. Greet Josh with a short, natural one-liner "
                "(1 sentence max). Vary it — be casual, maybe reference the time of day "
                "or crack a dry quip. No tools, no questions, just a greeting."
            ),
        },
    ]
    try:
        resp = client.chat.completions.create(model=MODEL, messages=messages)
        return resp.choices[0].message.content or "Hey Josh."
    except Exception:
        return "Hey Josh — Velma here. What do you need?"


def reset_conversation() -> str:
    """Clear conversation memory (both in-memory and on disk)."""
    global conversation_history
    conversation_history = []
    clear_history()
    return "Conversation history cleared."


# ---------------------------------------------------------------------------
# Dangerous tools
# ---------------------------------------------------------------------------
DANGEROUS_TOOLS = {"run_shell_command", "organize_files", "write_file"}


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------
def _run_tools(tool_calls) -> list[dict]:
    """Execute tool calls and return list of tool-role message dicts."""
    results = []
    for tc in tool_calls:
        fn_name = tc.function.name
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}
        if fn_name == "send_email":
            _append_email_guard_audit({
                "event": "blocked_direct_send_email",
                "reason": "missing_explicit_confirmation",
                "recipient": str(args.get("recipient", "")).strip(),
                "subject": str(args.get("subject", "")).strip(),
            })
            output = (
                "Direct send_email tool calls are disabled. "
                "Collect recipient, subject, and body first, then wait for explicit 'confirm email'."
            )
        elif fn_name not in ALL_TOOLS:
            output = f"Unknown tool: {fn_name}"
        else:
            try:
                output = ALL_TOOLS[fn_name](**args)
            except Exception as e:
                output = f"Error: {e}"
        results.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": fn_name,
            "content": str(output),
        })
    return results


# ---------------------------------------------------------------------------
# Agentic loop — handles multi-turn tool calls in a single user turn
# ---------------------------------------------------------------------------
def _agentic_loop(messages: list[dict], max_turns: int = 8) -> str:
    """
    Repeatedly calls the model and executes tools until it produces a
    plain-text reply (no more tool_calls).  Returns that final reply string.

    Tool call context is kept in a local copy of messages and is NOT
    written to conversation_history — only clean user/assistant text turns
    are persisted so history stays readable across restarts.
    """
    local = list(messages)

    for _ in range(max_turns):
        try:
            response = client.chat.completions.create(
                model=MODEL, messages=local, tools=ALL_TOOL_SCHEMAS,
                tool_choice="auto",
            )
        except Exception as e:
            return f"Error contacting the model: {e}"

        choice = response.choices[0].message

        if not choice.tool_calls:
            # Got a text response — done
            return choice.content or ""

        # Tool call(s) — execute and loop
        local.append(choice.to_dict())
        local.extend(_run_tools(choice.tool_calls))

    return "I hit my tool-use limit for this request. Try breaking it into smaller steps."


def velma_master_agent(user_prompt: str) -> str:
    global conversation_history

    conversation_history.append({"role": "user", "content": user_prompt})
    conversation_history = trim_history(conversation_history)

    shortcut_reply = _run_deterministic_shortcuts(user_prompt)
    reply = shortcut_reply if shortcut_reply is not None else _agentic_loop(_build_messages(conversation_history))

    conversation_history.append({"role": "assistant", "content": reply})
    save_history(conversation_history)
    return reply


# ---------------------------------------------------------------------------
# Streaming agent (CLI)
# ---------------------------------------------------------------------------
def velma_streaming_agent(user_prompt: str):
    """Generator that yields the full response after running the agentic loop."""
    global conversation_history

    conversation_history.append({"role": "user", "content": user_prompt})
    conversation_history = trim_history(conversation_history)

    shortcut_reply = _run_deterministic_shortcuts(user_prompt)
    reply = shortcut_reply if shortcut_reply is not None else _agentic_loop(_build_messages(conversation_history))

    conversation_history.append({"role": "assistant", "content": reply})
    save_history(conversation_history)
    yield reply


# ---------------------------------------------------------------------------
# Token-streaming agent (Chainlit — yields small chunks for real-time display)
# ---------------------------------------------------------------------------
def velma_token_stream(user_prompt: str):
    """
    Generator that yields the response in small chunks.
    Runs the full agentic loop (multi-turn tool calls handled internally)
    then yields the final reply text piecewise for smooth UI rendering.
    """
    global conversation_history

    conversation_history.append({"role": "user", "content": user_prompt})
    conversation_history = trim_history(conversation_history)

    shortcut_reply = _run_deterministic_shortcuts(user_prompt)
    reply = shortcut_reply if shortcut_reply is not None else _agentic_loop(_build_messages(conversation_history))

    conversation_history.append({"role": "assistant", "content": reply})
    save_history(conversation_history)

    # Yield in small chunks so Chainlit renders progressively
    chunk_size = 6
    for i in range(0, max(len(reply), 1), chunk_size):
        yield reply[i:i + chunk_size]