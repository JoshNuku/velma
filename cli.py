"""
Velma — Terminal interface.

Fast, no browser overhead. Run with:  python cli.py
"""

import re
import sys
import threading
import itertools
import time
from main import velma_streaming_agent, reset_conversation
from tools import set_reminder_callback

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
MAGENTA = "\033[95m"
UNDERLINE = "\033[4m"


def clean_markdown(text: str) -> str:
    """Convert markdown formatting to ANSI terminal codes."""
    # Fenced code blocks — keep content, dim it
    text = re.sub(r"```\w*\n(.*?)```", rf"{DIM}\1{RESET}", text, flags=re.S)
    # Inline code
    text = re.sub(r"`([^`]+)`", rf"{DIM}\1{RESET}", text)
    # Bold + italic
    text = re.sub(r"\*\*\*(.+?)\*\*\*", rf"{BOLD}{CYAN}\1{RESET}", text)
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", rf"{BOLD}\1{RESET}", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", rf"{CYAN}\1{RESET}", text)
    # Headings (strip the #'s, bold the text)
    text = re.sub(r"^#{1,6}\s+(.+)$", rf"{BOLD}\1{RESET}", text, flags=re.M)
    # Bullet points — clean up to a simple dash
    text = re.sub(r"^[\s]*[-*+]\s+", "  - ", text, flags=re.M)
    # Links [text](url) → text (url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", rf"\1 ({UNDERLINE}\2{RESET})", text)
    # Horizontal rules
    text = re.sub(r"^-{3,}$", "─" * 40, text, flags=re.M)
    return text


def _cli_reminder(text: str):
    """Display a reminder in the terminal with a bell."""
    sys.stdout.write(f"\n{MAGENTA}⏰ Reminder: {text}{RESET}\n")
    sys.stdout.write(f"{GREEN}Josh ❯{RESET} ")
    sys.stdout.flush()
    sys.stdout.write("\a")  # terminal bell


class _Spinner:
    """Animated thinking indicator that runs in a background thread."""

    def __init__(self):
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        frames = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        while not self._stop.is_set():
            sys.stdout.write(f"\r{YELLOW}{next(frames)} Velma is thinking...{RESET}")
            sys.stdout.flush()
            time.sleep(0.08)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        # Clear the spinner line
        sys.stdout.write("\r" + " " * 30 + "\r")
        sys.stdout.flush()


def main():
    set_reminder_callback(_cli_reminder)
    print(f"\n{CYAN}{BOLD}Velma{RESET} {DIM}(terminal mode — type /quit to exit, /clear to reset){RESET}\n")

    spinner = _Spinner()

    while True:
        try:
            user_input = input(f"{GREEN}Josh ❯{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "/q"):
            print("Bye.")
            break
        if cmd in ("/clear", "/reset", "/new"):
            print(reset_conversation())
            continue
        if cmd == "/help":
            print(
                f"{DIM}Commands: /clear  /quit  /help{RESET}\n"
                f"{DIM}Just type naturally — Velma has all the same tools as the web UI.{RESET}"
            )
            continue

        # Show spinner while waiting for first token
        spinner.start()
        first = True
        try:
            for token in velma_streaming_agent(user_input):
                if first:
                    spinner.stop()
                    sys.stdout.write(f"{CYAN}Velma:{RESET} ")
                    first = False
                sys.stdout.write(clean_markdown(token))
                sys.stdout.flush()
        except KeyboardInterrupt:
            spinner.stop()
            print(f"\n{DIM}(interrupted){RESET}\n")
            continue
        if first:
            # Never got a token — stop spinner anyway
            spinner.stop()
        print("\n")


if __name__ == "__main__":
    main()
