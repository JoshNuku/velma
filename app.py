"""
Velma — Chainlit UI layer.

Features:
  - Token streaming for responsive UX
  - Confirmation dialogs before dangerous actions
  - Scheduled reminders (shared with CLI via tools.py)
  - New-conversation button
"""

import asyncio
import queue
from datetime import datetime

import chainlit as cl

from main import (
    velma_token_stream,
    reset_conversation,
    DANGEROUS_TOOLS,
)
from tools import ALL_TOOLS, set_reminder_callback


# ---------------------------------------------------------------------------
# Reminder callback for Chainlit
# ---------------------------------------------------------------------------
def _chainlit_reminder(text: str):
    """Send a reminder as a Chainlit message (fire-and-forget from thread)."""
    import asyncio as _aio
    loop = _aio.get_event_loop()
    if loop.is_running():
        _aio.ensure_future(cl.Message(content=f"**⏰ Reminder:** {text}").send())
    else:
        loop.run_until_complete(cl.Message(content=f"**⏰ Reminder:** {text}").send())


set_reminder_callback(_chainlit_reminder)

# ---------------------------------------------------------------------------
# Chainlit Hooks
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_start():
    """Welcome message."""
    await cl.Message(
        content=(
            "Hey Josh — Velma here. What do you need?"
        )
    ).send()


@cl.action_callback("confirm_action")
async def on_confirm(action: cl.Action):
    """User confirmed a dangerous tool action."""
    await cl.Message(content=f"Confirmed — running **{action.value}** now.").send()


@cl.action_callback("cancel_action")
async def on_cancel(action: cl.Action):
    """User cancelled a dangerous tool action."""
    await cl.Message(content="Okay, cancelled.").send()


@cl.on_message
async def on_message(message: cl.Message):
    """Main message handler — streams Velma's response token-by-token."""

    user_text = message.content.strip()

    # Quick meta-commands
    if user_text.lower() in ("/clear", "/reset", "/new"):
        result = reset_conversation()
        await cl.Message(content=result).send()
        return

    # Stream the response token-by-token
    msg = cl.Message(content="")
    await msg.send()

    full_response = ""
    loop = asyncio.get_event_loop()
    token_queue: queue.Queue = queue.Queue()
    sentinel = object()

    def _produce():
        try:
            for token in velma_token_stream(user_text):
                token_queue.put(token)
        finally:
            token_queue.put(sentinel)

    loop.run_in_executor(None, _produce)

    while True:
        token = await loop.run_in_executor(None, token_queue.get)
        if token is sentinel:
            break
        full_response += token
        await msg.stream_token(token)

    # Finalise
    msg.content = full_response
    await msg.update()

