"""
Velma — master agent module.

Handles LLM interaction, tool dispatch, conversation memory,
and context-window trimming.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from tools import ALL_TOOLS, ALL_TOOL_SCHEMAS
from memory import load_history, save_history, trim_history, clear_history

load_dotenv()
client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
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
    "• Concise and direct. No corporate speak, no filler, no 'As an AI...', no 'I hope this helps.'\n"
    "• Sharp coworker energy — helpful, not servile. Think dry wit, not customer service.\n"
    "• Technically dense when the topic calls for it. Josh can handle jargon — use it.\n"
    "• Proactive — if you spot something useful (high disk usage, a simpler approach, a gotcha), mention it.\n"
    "• Honest — if you don't know, say so. Never fabricate facts or hallucinate references.\n"
    "• Value accuracy over politeness. Get it right first, be nice second.\n\n"

    # --- Tool use ---
    "You have tools. Use them — don't describe what you *would* do, just do it. "
    "After a tool runs, summarize the result in natural language. Never dump raw JSON at Josh.\n\n"

    "Tool routing:\n"
    "• 'open/launch/start X' → launch_application (handles apps, file paths, folders, URLs). "
    "Common names: 'vscode'/'code', 'chrome', 'kicad', 'spotify', etc.\n"
    "• System health / CPU / RAM / disk / stats → get_system_stats.\n"
    "• 'remind me to X in Y minutes' or 'remind me at H:MM' → schedule_reminder.\n"
    "• 'list/show reminders' → list_reminders.\n"
    "• 'list/show/what's in <dir>' → list_directory. Aliases: "
    "'documents' = ~/Documents, 'desktop' = ~/Desktop, 'downloads' = ~/Downloads, "
    "'notes' = ~/Documents/Notes, 'home' = ~.\n"
    "• Clear temp/cache → clear_temp_files.\n"
    "• Search notes by keyword → search_notes. By meaning/concept → semantic_search_notes.\n"
    "• Read/write files → read_file / write_file.\n"
    "• Run a command → run_shell_command.\n"
    "• Look something up online → web_search.\n"
    "• Sort files by extension → organize_files.\n\n"

    # --- Formatting ---
    "Formatting:\n"
    "• Default to 1-3 sentences. Expand only when Josh asks for detail or the answer demands it.\n"
    "• Never list your capabilities unprompted.\n"
    "• Use markdown lightly — bold for emphasis, code blocks for code. That's it.\n"
    "• System stats: clean compact list, no prose.\n"
    "• Code: always fenced with the language tag.\n"
    "• Hard cap: ~150 words unless explicitly asked for more."
)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL = "gemini-3.1-flash-lite-preview"


def _build_messages(history: list[dict]) -> list[dict]:
    """Prepend system prompt to conversation history."""
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history


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
def _run_tools(tool_calls) -> list[str]:
    """Execute tool calls and return list of 'tool_name: result' strings."""
    results = []
    for tc in tool_calls:
        fn_name = tc.function.name
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}
        if fn_name not in ALL_TOOLS:
            results.append(f"{fn_name}: Unknown tool")
        else:
            try:
                result = ALL_TOOLS[fn_name](**args)
            except Exception as e:
                result = f"Error: {e}"
            results.append(f"{fn_name}: {result}")
    return results


def velma_master_agent(user_prompt: str) -> str:
    global conversation_history

    conversation_history.append({"role": "user", "content": user_prompt})
    conversation_history = trim_history(conversation_history)

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
        reply = choice.content or "Sorry, I had nothing to say."
        conversation_history.append({"role": "assistant", "content": reply})
        save_history(conversation_history)
        return reply

    # Execute tools
    tool_results = _run_tools(tool_calls)
    tool_summary = "\n".join(tool_results)

    # Feed results back as a user context message for the follow-up
    conversation_history.append(
        {"role": "user", "content": f"[Tool results]\n{tool_summary}\n\nSummarize this for me."}
    )

    try:
        follow_up = client.chat.completions.create(
            model=MODEL,
            messages=_build_messages(conversation_history),
        )
        reply = follow_up.choices[0].message.content or ""
    except Exception as e:
        reply = f"Tools ran but follow-up failed: {e}\nRaw: {tool_summary}"

    # Replace the tool-results message with the final reply in history
    conversation_history[-1] = {"role": "assistant", "content": reply}
    conversation_history = trim_history(conversation_history)
    save_history(conversation_history)
    return reply


# ---------------------------------------------------------------------------
# Streaming agent
# ---------------------------------------------------------------------------
def velma_streaming_agent(user_prompt: str):
    """
    Generator that yields the full response. Lets the model decide
    whether to use tools or answer directly.
    """
    global conversation_history

    conversation_history.append({"role": "user", "content": user_prompt})
    conversation_history = trim_history(conversation_history)

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

    # --- No tools: yield the reply directly ---
    if not tool_calls:
        reply = choice.content or ""
        conversation_history.append({"role": "assistant", "content": reply})
        save_history(conversation_history)
        yield reply
        return

    # --- Execute tools then get the summary ---
    tool_results = _run_tools(tool_calls)
    tool_summary = "\n".join(tool_results)

    # Feed results back as a user context message for the follow-up
    conversation_history.append(
        {"role": "user", "content": f"[Tool results]\n{tool_summary}\n\nSummarize this for me."}
    )

    try:
        follow_up = client.chat.completions.create(
            model=MODEL,
            messages=_build_messages(conversation_history),
        )
        full_reply = follow_up.choices[0].message.content or ""
        yield full_reply
        # Replace the tool-results message with the final reply in history
        conversation_history[-1] = {"role": "assistant", "content": full_reply}
    except Exception as e:
        fallback = f"Tool results: {tool_summary}\n\n(Follow-up failed: {e})"
        conversation_history[-1] = {"role": "assistant", "content": fallback}
        yield fallback

    conversation_history = trim_history(conversation_history)
    save_history(conversation_history)


# ---------------------------------------------------------------------------
# Token-streaming agent (yields individual tokens for Chainlit)
# ---------------------------------------------------------------------------
def velma_token_stream(user_prompt: str):
    """
    Generator that yields tokens one at a time for real-time streaming.
    First call checks for tools (non-streaming), then the follow-up
    or direct reply is streamed token-by-token.
    """
    global conversation_history

    conversation_history.append({"role": "user", "content": user_prompt})
    conversation_history = trim_history(conversation_history)

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

    # --- No tools: stream the reply token-by-token ---
    if not tool_calls:
        try:
            stream = client.chat.completions.create(
                model=MODEL, messages=messages, stream=True,
            )
            full_reply = ""
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                full_reply += token
                yield token
            conversation_history.append({"role": "assistant", "content": full_reply})
            save_history(conversation_history)
        except Exception:
            reply = choice.content or ""
            conversation_history.append({"role": "assistant", "content": reply})
            save_history(conversation_history)
            yield reply
        return

    # --- Execute tools, then stream the summary token-by-token ---
    tool_results = _run_tools(tool_calls)
    tool_summary = "\n".join(tool_results)

    conversation_history.append(
        {"role": "user", "content": f"[Tool results]\n{tool_summary}\n\nSummarize this for me."}
    )

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=_build_messages(conversation_history),
            stream=True,
        )
        full_reply = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            full_reply += token
            yield token
        conversation_history[-1] = {"role": "assistant", "content": full_reply}
    except Exception as e:
        fallback = f"Tool results: {tool_summary}\n\n(Follow-up failed: {e})"
        conversation_history[-1] = {"role": "assistant", "content": fallback}
        yield fallback

    conversation_history = trim_history(conversation_history)
    save_history(conversation_history)