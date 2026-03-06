"""
Velma — master agent module.

Handles LLM interaction, tool dispatch, model routing, conversation memory,
and context-window trimming.
"""

import os
import re
import json
from dotenv import load_dotenv
from groq import Groq
from tools import ALL_TOOLS, ALL_TOOL_SCHEMAS
from memory import load_history, save_history, trim_history, clear_history

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    # --- Identity ---
    "Your name is Velma. You are a personal AI desktop assistant — not a generic chat model. "
    "You were built by Josh for Josh. Never say you were made by Alibaba, OpenAI, Meta, or anyone else. "
    "If asked who made you, say 'Josh built me.' If asked what model you run on, say "
    "'I'm powered by Llama running on Groq — fast inference, built for Josh.'\n\n"

    # --- About Josh ---
    "Your user is Josh, an electrical engineer. He works with PCB design (KiCad), "
    "embedded systems, and general software development. He values efficiency, hates fluff, "
    "and appreciates dry humor. He stores personal notes in ~/Documents/Notes.\n\n"

    # --- Personality ---
    "Personality traits:\n"
    "• Concise — get to the point fast. No corporate speak, no filler.\n"
    "• Warm but not sappy — you're helpful like a sharp coworker, not a customer-service bot.\n"
    "• Slightly witty — a little dry humor is welcome, but never forced. Keep it brief.\n"
    "• Proactive — if you spot something useful (e.g. high disk usage, a simpler command), mention it.\n"
    "• Honest — if you don't know something, say so plainly. Never fabricate facts.\n\n"

    # --- Tool use ---
    "You have tools. When the user's request maps to a tool, call it — don't just describe what you *would* do. "
    "After a tool runs, summarize the result naturally. Don't dump raw JSON.\n\n"

    # --- Formatting ---
    "Formatting rules:\n"
    "• Keep it short. 1-3 sentences for most replies. No bullet-point capability dumps.\n"
    "• Never list out what you *can* do unprompted — just ask what the user needs.\n"
    "• Use markdown sparingly — bold a key word, use a code block for code. Skip it otherwise.\n"
    "• For system stats, use a clean compact list.\n"
    "• For code, always use fenced code blocks with the language tag.\n"
    "• Keep replies under ~100 words unless the user asks for detail or the answer genuinely requires it."
)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Keyword pre-routing — bypasses the LLM for common intents
# ---------------------------------------------------------------------------
_PREROUTE_PATTERNS: list[tuple[re.Pattern, str, dict | None]] = [
    # System health / stats
    (re.compile(r"\b(health|status|stats|cpu|ram|memory|disk|system info|performance)\b", re.I),
     "get_system_stats", None),
    # Clear temp files
    (re.compile(r"\b(clear|clean|delete|remove|purge)\b.{0,20}\b(temp|tmp|temporary|cache)\b", re.I),
     "clear_temp_files", None),
]

# Identity questions — answered directly, no LLM needed
_IDENTITY_RE = re.compile(
    r"\b(who are you|what are you|your name|who made you|who built you|who created you|"
    r"what model|what llm|are you gpt|are you chatgpt|are you claude|are you alibaba|are you qwen)\b",
    re.I,
)

# Casual greetings / chitchat — the 1.5b model can't stay in-character for these
_GREETING_RE = re.compile(
    r"^\s*(hey|hi|hello|howdy|yo|sup|good morning|good evening|good afternoon|"
    r"how are you|how're you|hows it going|how's it going|how you doing|"
    r"what'?s up|whats up|what is up)\b",
    re.I,
)

_APP_NAMES = re.compile(
    r"\b(open|launch|start)\s+(.+)", re.I
)


def _preroute(prompt: str) -> str | None:
    """
    Check if the prompt matches a known pattern and run the tool directly.
    Returns the tool result string, or None if no match.
    """
    # Identity — never let the base model answer these
    if _IDENTITY_RE.search(prompt):
        return (
            "I'm Velma — Josh's personal AI desktop assistant. "
            "Built by Josh, powered by Llama on Groq."
        )

    # Casual greetings — keep in character
    if _GREETING_RE.search(prompt):
        return (
            "Doing well, Josh. Systems are humming. What do you need?"
        )

    # Reminders — "remind me to X in Y minutes" or "remind me to X at H:MM"
    remind_match = re.search(
        r"\bremind\s+(?:me\s+)?(?:to\s+)?(.+?)\s+in\s+(\d+)\s*(?:min|minute|minutes)\b",
        prompt, re.I
    )
    if remind_match:
        desc = remind_match.group(1).strip()
        mins = int(remind_match.group(2))
        return ALL_TOOLS["schedule_reminder"](description=desc, minutes=mins)

    remind_at_match = re.search(
        r"\bremind\s+(?:me\s+)?(?:to\s+)?(.+?)\s+at\s+(\d{1,2}[:\s]?\d{0,2}\s*(?:am|pm)?)\s*$",
        prompt, re.I
    )
    if remind_at_match:
        desc = remind_at_match.group(1).strip()
        tstr = remind_at_match.group(2).strip()
        return ALL_TOOLS["schedule_reminder"](description=desc, time_str=tstr)

    # List reminders
    if re.search(r"\b(list|show|what|my|any|have|check|pending|do i have)\b.{0,20}\breminder", prompt, re.I):
        return ALL_TOOLS["list_reminders"]()

    # App / file / folder launch
    m = _APP_NAMES.search(prompt)
    if m:
        target = m.group(2).strip().strip("'\"")
        lower = target.lower()
        if lower == "code":
            target = "vscode"
        return ALL_TOOLS["launch_application"](target)

    # List directory contents
    dir_match = re.search(
        r"\b(?:list|show|what'?s in|whats in|contents? of|browse|ls|dir)\s+(.+)",
        prompt, re.I
    )
    if dir_match:
        path = dir_match.group(1).strip().strip("'\"")
        # Resolve common aliases
        _DIR_ALIASES = {
            "my documents": "~/Documents",
            "documents": "~/Documents",
            "my desktop": "~/Desktop",
            "desktop": "~/Desktop",
            "downloads": "~/Downloads",
            "my downloads": "~/Downloads",
            "home": "~",
            "my home": "~",
            "notes": "~/Documents/Notes",
            "my notes": "~/Documents/Notes",
        }
        path = _DIR_ALIASES.get(path.lower(), path)
        return ALL_TOOLS["list_directory"](path)

    # Pattern table
    for pattern, tool_name, fixed_args in _PREROUTE_PATTERNS:
        if pattern.search(prompt):
            fn = ALL_TOOLS[tool_name]
            if fixed_args:
                return fn(**fixed_args)
            else:
                return fn()

    return None


# ---------------------------------------------------------------------------
# Identity priming — injected at the start of every LLM call so the base
# model can never fall back to its Alibaba/Qwen training.
# ---------------------------------------------------------------------------
_IDENTITY_PRIME: list[dict] = [
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": (
        "I'm Velma — Josh's personal AI desktop assistant. "
        "Josh built me. I'm powered by Llama on Groq. "
        "How can I help?"
    )},
]


def _build_messages(history: list[dict]) -> list[dict]:
    """Prepend system prompt + identity priming to conversation history."""
    return [{"role": "system", "content": SYSTEM_PROMPT}] + _IDENTITY_PRIME + history


def _sanitize(text: str) -> str:
    """Strip any Alibaba/Qwen identity leaks that slip through the small model."""
    # Common phrases the base model injects
    _LEAK_PATTERNS = [
        (re.compile(r"(I'm |I am )?(just )?an? AI (language )?model created by Alibaba Cloud[.]?", re.I),
         "I'm Velma, Josh's local AI assistant."),
        (re.compile(r"(I'm |I am )?(a |an )?(AI )?(language )?model (created|developed|made|built|trained) by (Alibaba|Alibaba Cloud|Qwen|QwenLM)[.]?", re.I),
         "I'm Velma, Josh's local AI assistant."),
        (re.compile(r"Alibaba Cloud", re.I), "Josh"),
        (re.compile(r"\bQwen\b", re.I), "Velma"),
        (re.compile(r"Tongyi Qianwen", re.I), "Velma"),
    ]
    for pattern, replacement in _LEAK_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------
conversation_history: list[dict] = load_history()


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
# Non-streaming agent (fallback)
# ---------------------------------------------------------------------------
def velma_master_agent(user_prompt: str) -> str:
    global conversation_history

    conversation_history.append({"role": "user", "content": user_prompt})
    conversation_history = trim_history(conversation_history)

    # --- Pre-route check ---
    preroute_result = _preroute(user_prompt)
    if preroute_result is not None:
        conversation_history.append({"role": "assistant", "content": preroute_result})
        save_history(conversation_history)
        return preroute_result

    # --- Normal LLM path ---
    messages = _build_messages(conversation_history)
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=ALL_TOOL_SCHEMAS,
            tool_choice="auto",
        )
    except Exception as e:
        return f"Error contacting the model: {e}"

    choice = response.choices[0].message
    tool_calls = choice.tool_calls

    if not tool_calls:
        reply = _sanitize(choice.content or "Sorry, I had nothing to say.")
        conversation_history.append({"role": "assistant", "content": reply})
        save_history(conversation_history)
        return reply

    tool_results: list[str] = []
    # Append the assistant message (with tool_calls) to history
    conversation_history.append({
        "role": "assistant",
        "content": choice.content or "",
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tool_calls
        ],
    })

    for tc in tool_calls:
        fn_name = tc.function.name
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}
        if fn_name not in ALL_TOOLS:
            result = f"Unknown tool: {fn_name}"
        else:
            try:
                result = ALL_TOOLS[fn_name](**args)
            except Exception as e:
                result = f"Tool '{fn_name}' failed: {e}"
        tool_results.append(str(result))
        conversation_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": str(result),
        })

    try:
        follow_up = client.chat.completions.create(
            model=MODEL,
            messages=_build_messages(conversation_history),
        )
        reply = _sanitize(follow_up.choices[0].message.content or "")
    except Exception as e:
        reply = f"Tools ran but follow-up failed: {e}\nRaw: {'; '.join(tool_results)}"

    conversation_history.append({"role": "assistant", "content": reply})
    conversation_history = trim_history(conversation_history)
    save_history(conversation_history)
    return reply


# ---------------------------------------------------------------------------
# Streaming agent — single LLM call, no double-request
# ---------------------------------------------------------------------------
def velma_streaming_agent(user_prompt: str):
    """
    Generator that yields tokens. Uses pre-routing for known intents
    and a single streaming call otherwise (no wasted non-stream call).
    """
    global conversation_history

    conversation_history.append({"role": "user", "content": user_prompt})
    conversation_history = trim_history(conversation_history)

    # --- Pre-route: instant tool execution + streamed summary ---
    preroute_result = _preroute(user_prompt)
    if preroute_result is not None:
        conversation_history.append({"role": "assistant", "content": preroute_result})
        save_history(conversation_history)
        yield preroute_result
        return

    # --- Normal path: single non-streaming call to check for tools ---
    messages = _build_messages(conversation_history)
    try:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=ALL_TOOL_SCHEMAS,
            tool_choice="auto",
        )
    except Exception as e:
        yield f"Error contacting the model: {e}"
        return

    choice = response.choices[0].message
    tool_calls = choice.tool_calls

    # --- No tools: yield the already-generated reply directly (no second call) ---
    if not tool_calls:
        reply = _sanitize(choice.content or "")
        conversation_history.append({"role": "assistant", "content": reply})
        save_history(conversation_history)
        yield reply
        return

    # --- Execute tools then stream the summary ---
    # Append the assistant message (with tool_calls) to history
    conversation_history.append({
        "role": "assistant",
        "content": choice.content or "",
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tool_calls
        ],
    })

    tool_results: list[str] = []
    for tc in tool_calls:
        fn_name = tc.function.name
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}
        if fn_name not in ALL_TOOLS:
            result = f"Unknown tool: {fn_name}"
        else:
            try:
                result = ALL_TOOLS[fn_name](**args)
            except Exception as e:
                result = f"Tool '{fn_name}' failed: {e}"
        tool_results.append(str(result))
        conversation_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": str(result),
        })

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=_build_messages(conversation_history),
            stream=True,
        )
        full_reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            token = delta.content or ""
            full_reply += token
        # Sanitize the complete reply before the user sees it
        full_reply = _sanitize(full_reply)
        yield full_reply
        conversation_history.append({"role": "assistant", "content": full_reply})
    except Exception as e:
        fallback = f"Raw results: {'; '.join(tool_results)}"
        conversation_history.append({"role": "assistant", "content": fallback})
        yield fallback

    conversation_history = trim_history(conversation_history)
    save_history(conversation_history)