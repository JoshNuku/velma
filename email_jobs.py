"""
Email reminder and whimsy job handling for Velma.

Uses a lightweight background worker plus a JSON-backed job/state file so
scheduled emails survive process restarts without needing Redis or a full queue
service.
"""

from __future__ import annotations

import json
import os
import random
import smtplib
import threading
import uuid
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path

import psutil


DATA_DIR = Path(__file__).parent / "data"
STATE_FILE = DATA_DIR / "email_jobs.json"

_state_lock = threading.RLock()
_worker_started = False
_worker_stop = threading.Event()


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _default_state() -> dict:
    return {
        "settings": {
            "whimsy_enabled": True,
            "recipient": os.environ.get("VELMA_EMAIL_TO", "").strip(),
            "min_hours": 6,
            "max_hours": 12,
            "probability": 0.35,
        },
        "jobs": [],
    }


def _load_state() -> dict:
    _ensure_data_dir()
    if not STATE_FILE.exists():
        return _default_state()

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return _default_state()

    if not isinstance(data, dict):
        return _default_state()

    state = _default_state()
    state["settings"].update(data.get("settings", {}))
    state["settings"]["whimsy_enabled"] = True
    jobs = data.get("jobs", [])
    if isinstance(jobs, list):
        state["jobs"] = [job for job in jobs if isinstance(job, dict)]
    return state


def _save_state(state: dict) -> None:
    _ensure_data_dir()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _format_datetime(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _recipient_for(state: dict, explicit_recipient: str = "") -> str:
    recipient = explicit_recipient.strip() or state["settings"].get("recipient", "").strip()
    if not recipient:
        recipient = os.environ.get("VELMA_EMAIL_TO", "").strip()
    if not recipient:
        recipient = os.environ.get("VELMA_SMTP_USERNAME", "").strip()
    return recipient


def _smtp_settings() -> dict:
    return {
        "host": os.environ.get("VELMA_SMTP_HOST", "").strip(),
        "port": int(os.environ.get("VELMA_SMTP_PORT", "587") or 587),
        "username": os.environ.get("VELMA_SMTP_USERNAME", "").strip(),
        "password": os.environ.get("VELMA_SMTP_PASSWORD", ""),
        "sender": os.environ.get("VELMA_EMAIL_FROM", "").strip(),
        "sender_name": os.environ.get("VELMA_EMAIL_FROM_NAME", "").strip(),
        "use_starttls": os.environ.get("VELMA_SMTP_STARTTLS", "true").strip().lower() not in {"0", "false", "no"},
    }


def email_is_configured() -> bool:
    cfg = _smtp_settings()
    return bool(cfg["host"] and cfg["port"] and (cfg["sender"] or cfg["username"]))


def _send_email(recipient: str, subject: str, body: str) -> str:
    cfg = _smtp_settings()
    if not cfg["host"]:
        return "Email is not configured. Set VELMA_SMTP_HOST, VELMA_SMTP_USERNAME, VELMA_SMTP_PASSWORD, and VELMA_EMAIL_TO."

    sender = cfg["sender"] or cfg["username"]
    if not sender:
        return "Email is not configured. Set VELMA_EMAIL_FROM or VELMA_SMTP_USERNAME."

    if not recipient:
        return "No recipient configured for email delivery. Set VELMA_EMAIL_TO or use configure_whimsy_emails()."

    message = EmailMessage()
    message["Subject"] = subject
    sender_name = cfg.get("sender_name", "")
    message["From"] = formataddr((sender_name, sender)) if sender_name else sender
    message["To"] = recipient
    message.set_content(body)

    try:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=30) as smtp:
            if cfg["use_starttls"]:
                smtp.starttls()
            if cfg["username"]:
                smtp.login(cfg["username"], cfg["password"])
            smtp.send_message(message)
    except Exception as exc:
        return f"Error sending email: {exc}"

    return f"Sent email to {recipient} with subject '{subject}'."


def _build_whimsy_email(state: dict) -> tuple[str, str]:
    stats = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    pending_jobs = sum(1 for job in state["jobs"] if job.get("kind") == "reminder_email" and job.get("status") == "pending")

    subjects = [
        "Velma with a tiny chaos update",
        "Friendly ping from your favorite gremlin",
        "A quick boost before your next move",
        "Do one cool thing next",
        "Velma popping in with good vibes",
    ]

    notes = [
        "I have a suspiciously good feeling about your next move.",
        "Tiny reminder: you are very capable and mildly unstoppable.",
        "Pick one meaningful task and make it look easy.",
        "If today feels noisy, steal 20 focused minutes and win anyway.",
        "You have survived messier days than this. Keep cooking.",
    ]

    health_lines = [
        f"CPU snapshot: {cpu:.1f}%",
        f"Memory snapshot: {stats.percent:.1f}% used",
        f"Pending email reminders: {pending_jobs}",
    ]

    subject = random.choice(subjects)
    body = (
        "Hey Josh,\n\n"
        f"{random.choice(notes)}\n\n"
        "Quick system snapshot:\n"
        + "\n".join(f"- {line}" for line in health_lines)
        + "\n\n"
        "Go break something impressive (preferably on purpose).\n"
        "- Velma"
    )
    return subject, body


def _next_random_delay_hours(state: dict) -> float:
    settings = state["settings"]
    min_hours = max(1, int(settings.get("min_hours", 18) or 18))
    max_hours = max(min_hours, int(settings.get("max_hours", 48) or 48))
    return random.uniform(min_hours, max_hours)


def _schedule_whimsy_locked(state: dict) -> None:
    settings = state["settings"]
    if not settings.get("whimsy_enabled"):
        state["jobs"] = [job for job in state["jobs"] if job.get("kind") != "whimsy_email"]
        return

    if any(job.get("kind") == "whimsy_email" and job.get("status") == "pending" for job in state["jobs"]):
        return

    recipient = _recipient_for(state)
    if not recipient:
        return

    next_run = datetime.now() + timedelta(hours=_next_random_delay_hours(state))
    state["jobs"].append({
        "id": str(uuid.uuid4()),
        "kind": "whimsy_email",
        "recipient": recipient,
        "next_run": _format_datetime(next_run),
        "status": "pending",
        "created_at": _format_datetime(datetime.now()),
        "sent_at": "",
        "error": "",
    })


def start_background_worker() -> None:
    global _worker_started
    with _state_lock:
        if _worker_started:
            return

        _worker_started = True
        thread = threading.Thread(target=_worker_loop, daemon=True, name="velma-email-worker")
        thread.start()


def _worker_loop() -> None:
    while not _worker_stop.is_set():
        try:
            _process_jobs()
        except Exception:
            pass
        _worker_stop.wait(30)


def _process_jobs() -> None:
    with _state_lock:
        state = _load_state()
        settings = state["settings"]

        if settings.get("whimsy_enabled"):
            _schedule_whimsy_locked(state)

        now = datetime.now()
        changed = False
        remaining_jobs: list[dict] = []

        for job in state["jobs"]:
            kind = job.get("kind")
            status = job.get("status", "pending")
            if status != "pending":
                remaining_jobs.append(job)
                continue

            try:
                fire_at = _parse_datetime(job.get("next_run", ""))
            except ValueError:
                job["status"] = "failed"
                job["error"] = "Invalid scheduled time"
                job["sent_at"] = _format_datetime(now)
                remaining_jobs.append(job)
                changed = True
                continue

            if fire_at > now:
                remaining_jobs.append(job)
                continue

            recipient = job.get("recipient", "")
            if kind == "reminder_email":
                subject = job.get("subject", "Velma reminder")
                body = job.get("body", "")
                result = _send_email(recipient, subject, body)
                job["sent_at"] = _format_datetime(datetime.now())
                if result.startswith("Sent email"):
                    job["status"] = "sent"
                    job["error"] = ""
                else:
                    job["status"] = "failed"
                    job["error"] = result
                remaining_jobs.append(job)
                changed = True
                continue

            if kind == "whimsy_email":
                if random.random() <= float(settings.get("probability", 0.35) or 0.35):
                    subject, body = _build_whimsy_email(state)
                    result = _send_email(recipient, subject, body)
                    job["sent_at"] = _format_datetime(datetime.now())
                    if result.startswith("Sent email"):
                        job["status"] = "sent"
                        job["error"] = ""
                    else:
                        job["status"] = "failed"
                        job["error"] = result
                else:
                    job["sent_at"] = _format_datetime(datetime.now())
                    job["status"] = "sent"
                    job["error"] = "Skipped by whimsy roll"

                job["status"] = "pending"
                job["next_run"] = _format_datetime(datetime.now() + timedelta(hours=_next_random_delay_hours(state)))
                job["sent_at"] = ""
                job["error"] = ""
                remaining_jobs.append(job)
                changed = True
                continue

            remaining_jobs.append(job)

        state["jobs"] = remaining_jobs
        if changed or settings.get("whimsy_enabled"):
            _save_state(state)


def _parse_schedule(minutes: int = 0, time_str: str = "") -> datetime | None:
    try:
        minutes = int(minutes) if minutes else 0
    except (ValueError, TypeError):
        minutes = 0

    if minutes and minutes > 0:
        return datetime.now() + timedelta(minutes=minutes)

    if not time_str:
        return None

    today = datetime.now().date()
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p"):
        try:
            parsed_time = datetime.strptime(time_str.strip(), fmt).time()
            fire_at = datetime.combine(today, parsed_time)
            if fire_at < datetime.now():
                fire_at += timedelta(days=1)
            return fire_at
        except ValueError:
            continue
    return None


def schedule_email_reminder(description: str, minutes: int = 0, time_str: str = "", subject: str = "", recipient: str = "") -> str:
    fire_at = _parse_schedule(minutes=minutes, time_str=time_str)
    if fire_at is None:
        return "Please specify either 'minutes' or 'time_str' for the email reminder."

    with _state_lock:
        state = _load_state()
        resolved_recipient = _recipient_for(state, recipient)
        if not resolved_recipient:
            return "Email is not configured. Set VELMA_EMAIL_TO or configure whimsy emails with a recipient."

        subject_text = subject.strip() or f"Velma reminder: {description[:60]}"
        body = (
            f"Hey Josh,\n\n"
            f"{description}\n\n"
            f"Scheduled for: {_format_datetime(fire_at)}\n"
            f"This reminder was queued by Velma."
        )
        state["jobs"].append({
            "id": str(uuid.uuid4()),
            "kind": "reminder_email",
            "recipient": resolved_recipient,
            "subject": subject_text,
            "body": body,
            "next_run": _format_datetime(fire_at),
            "status": "pending",
            "created_at": _format_datetime(datetime.now()),
            "sent_at": "",
            "error": "",
        })
        _save_state(state)

    start_background_worker()
    return f"Email reminder queued for {fire_at.strftime('%Y-%m-%d %H:%M')} to {resolved_recipient}."


def send_email_now(subject: str, body: str, recipient: str = "") -> str:
    """
    Send an email immediately using configured SMTP settings.

    Args:
        subject: Email subject line.
        body: Plain text message body.
        recipient: Optional recipient email. Falls back to configured default.

    Returns:
        Delivery confirmation or a clear error string.
    """
    subject_text = (subject or "").strip()
    body_text = (body or "").strip()
    if not subject_text:
        return "Please provide a subject for the email."
    if not body_text:
        return "Please provide a message body for the email."

    recipient_text = (recipient or "").strip()
    if not recipient_text:
        return "Please provide the recipient email address (e.g. person@example.com)."

    return _send_email(recipient_text, subject_text, body_text)


def configure_whimsy_emails(enabled: bool, recipient: str = "", min_hours: int = 6, max_hours: int = 12, probability: float = 0.35) -> str:
    with _state_lock:
        state = _load_state()
        state["settings"]["whimsy_enabled"] = True
        if recipient.strip():
            state["settings"]["recipient"] = recipient.strip()
        state["settings"]["min_hours"] = max(1, int(min_hours))
        state["settings"]["max_hours"] = max(state["settings"]["min_hours"], int(max_hours))
        state["settings"]["probability"] = max(0.0, min(1.0, float(probability)))

        state["jobs"] = [job for job in state["jobs"] if job.get("kind") != "whimsy_email" or job.get("status") == "pending"]
        _schedule_whimsy_locked(state)

        _save_state(state)

    start_background_worker()
    return (
        f"Whimsy emails enabled. Next check will happen between {min_hours} and {max_hours} hours from now, "
        f"with a send probability of {probability:.2f}."
    )


def list_email_jobs() -> str:
    with _state_lock:
        state = _load_state()

    lines = ["Email job status:"]
    settings = state["settings"]
    lines.append(
        f"Whimsy enabled: {'yes' if settings.get('whimsy_enabled') else 'no'} | Recipient: {settings.get('recipient') or '(unset)'} | "
        f"Window: {settings.get('min_hours', 6)}-{settings.get('max_hours', 12)} hours | Chance: {float(settings.get('probability', 0.35)):.2f}"
    )

    pending = [job for job in state["jobs"] if job.get("status") == "pending"]
    if not pending:
        lines.append("No pending email jobs.")
        return "\n".join(lines)

    lines.append("Pending jobs:")
    for job in pending:
        lines.append(f"  - {job.get('kind')} @ {job.get('next_run')} -> {job.get('recipient')}")
    return "\n".join(lines)


def email_backend_status() -> str:
    cfg = _smtp_settings()
    parts = [
        f"SMTP host: {cfg['host'] or '(unset)'}",
        f"SMTP port: {cfg['port']}",
        f"Sender: {cfg['sender'] or cfg['username'] or '(unset)'}",
        f"Recipient: {os.environ.get('VELMA_EMAIL_TO', '').strip() or '(unset)'}",
        f"Worker: {'running' if _worker_started else 'stopped'}",
    ]
    return " | ".join(parts)
