# Velma

Personal AI desktop assistant built by Josh. Powered by Gemini.

## What it does

- **Launch apps, files, folders, URLs** — "open vscode", "open ~/Documents/Notes"
- **System stats** — CPU, RAM, disk usage, top processes
- **File operations** — read, write, organize, list directories
- **Note search** — keyword and semantic (vector RAG via ChromaDB)
- **Web search** — DuckDuckGo integration
- **Shell commands** — run anything from natural language
- **Reminders** — "remind me to push the commit in 10 minutes"
- **Email reminders** — queue reminder emails and optional random check-in emails
- **Temp file cleanup** — clear Windows temp dirs

## Setup

### 1. Install dependencies

```bash
python -m venv velma_env
velma_env\Scripts\activate
pip install -r requirements.txt  # or install packages manually
```

### 2. Get a Gemini API key

Sign up at [aistudio.google.com](https://aistudio.google.com/apikey) (free tier available).

### 3. Configure `.env`

```
GEMINI_API_KEY=your-key-here
VELMA_SMTP_HOST=smtp.gmail.com
VELMA_SMTP_PORT=587
VELMA_SMTP_USERNAME=you@example.com
VELMA_SMTP_PASSWORD=your-app-password
VELMA_EMAIL_FROM=you@example.com
VELMA_EMAIL_TO=you@example.com
VELMA_SMTP_STARTTLS=true
VELMA_SYSTEM_PROMPT=Optional override for the full Velma system prompt
```

You can override Velma's built-in system prompt with `VELMA_SYSTEM_PROMPT`.
If you use line breaks in `.env`, write them as escaped `\n`.

If you want Velma to send occasional random emails, enable them with the `configure_whimsy_emails` tool after setting the SMTP variables. The feature runs as a background worker inside the app process and persists its job state in `data/email_jobs.json`.

### 4. Run

**Terminal UI:**
```bash
python cli.py
```

**Web UI (Chainlit):**
```bash
chainlit run app.py
```

**Admin mode** (for clearing system temp files):
```bash
velma_admin.bat
```

## CLI Commands

| Command | Action |
|---------|--------|
| `/clear` | Reset conversation history |
| `/quit` | Exit |
| `/help` | Show available commands |

## Project Structure

```
app.py          — Chainlit web UI
cli.py          — Terminal interface
main.py         — Agent logic, LLM calls, tool dispatch, pre-routing
tools.py        — All tool implementations + OpenAI-format schemas
memory.py       — Persistent conversation history (JSON)
rag.py          — Semantic note search (ChromaDB embeddings)
data/           — Conversation history storage
```

