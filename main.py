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
# Non-streaming agent (fallback)
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
        if fn_name not in ALL_TOOLS:
            output = f"Unknown tool: {fn_name}"
        else:
            try:
                output = ALL_TOOLS[fn_name](**args)
            except Exception as e:
                output = f"Error: {e}"
        results.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": str(output),
        })
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

    # Record the assistant's tool-call message in history
    conversation_history.append(choice.to_dict())

    # Execute tools and append each result as a proper tool message
    tool_messages = _run_tools(tool_calls)
    conversation_history.extend(tool_messages)

    # Let the model summarise the results
    try:
        follow_up = client.chat.completions.create(
            model=MODEL,
            messages=_build_messages(conversation_history),
        )
        reply = follow_up.choices[0].message.content or ""
    except Exception as e:
        raw = "\n".join(m["content"] for m in tool_messages)
        reply = f"Tools ran but follow-up failed: {e}\nRaw: {raw}"

    conversation_history.append({"role": "assistant", "content": reply})
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

    # --- Record assistant tool-call message, execute, append results ---
    conversation_history.append(choice.to_dict())
    tool_messages = _run_tools(tool_calls)
    conversation_history.extend(tool_messages)

    try:
        follow_up = client.chat.completions.create(
            model=MODEL,
            messages=_build_messages(conversation_history),
        )
        full_reply = follow_up.choices[0].message.content or ""
        yield full_reply
        conversation_history.append({"role": "assistant", "content": full_reply})
    except Exception as e:
        raw = "\n".join(m["content"] for m in tool_messages)
        fallback = f"Tool results: {raw}\n\n(Follow-up failed: {e})"
        conversation_history.append({"role": "assistant", "content": fallback})
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
    is streamed token-by-token.
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

    # --- No tools: use the already-received reply, stream it token-by-token ---
    if not tool_calls:
        reply = choice.content or ""
        conversation_history.append({"role": "assistant", "content": reply})
        save_history(conversation_history)
        # Yield in small chunks to simulate streaming from the cached response
        chunk_size = 4
        for i in range(0, len(reply), chunk_size):
            yield reply[i:i + chunk_size]
        return

    # --- Record assistant tool-call, execute tools, append results ---
    conversation_history.append(choice.to_dict())
    tool_messages = _run_tools(tool_calls)
    conversation_history.extend(tool_messages)

    # Stream the follow-up summary token-by-token
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
        conversation_history.append({"role": "assistant", "content": full_reply})
    except Exception as e:
        raw = "\n".join(m["content"] for m in tool_messages)
        fallback = f"Tool results: {raw}\n\n(Follow-up failed: {e})"
        conversation_history.append({"role": "assistant", "content": fallback})
        yield fallback

    conversation_history = trim_history(conversation_history)
    save_history(conversation_history)