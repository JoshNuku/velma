import os
import subprocess
import shutil
import psutil
import threading
from datetime import datetime, timedelta
from ddgs import DDGS

# ---------------------------------------------------------------------------
# Reminder callback — set by the UI layer (cli or chainlit)
# ---------------------------------------------------------------------------
_reminder_callback = None  # Will be set to a function(text: str) by the UI
_scheduled_reminders: list[dict] = []  # Track active reminders for listing


def set_reminder_callback(fn):
    """Called by the UI layer to register how reminders should be displayed."""
    global _reminder_callback
    _reminder_callback = fn

# ---------------------------------------------------------------------------
# TOOL 1: App / File / URL Opener
# ---------------------------------------------------------------------------
def launch_application(app_name: str) -> str:
    """
    Opens an application, file, folder, or URL.
    Knows common app shortcuts (vscode, chrome, kicad, etc.) and can also
    open arbitrary paths or URLs via the system default handler.

    Args:
        app_name: An app shortcut name, file path, folder path, or URL to open.

    Returns:
        A confirmation message or an error message.
    """
    # Known shortcuts
    shortcuts = {
        "vscode": "code",
        "code": "code",
        "browser": "start chrome",
        "chrome": "start chrome",
        "firefox": "start firefox",
        "edge": "start msedge",
        "kicad": "kicad",
        "spotify": "start spotify",
        "notepad": "notepad",
        "terminal": "start cmd",
        "cmd": "start cmd",
        "powershell": "start powershell",
        "explorer": "explorer",
        "discord": "start discord",
        "steam": "start steam",
        "task manager": "taskmgr",
        "taskmgr": "taskmgr",
        "settings": "start ms-settings:",
        "calc": "calc",
        "calculator": "calc",
    }

    key = app_name.lower().strip()
    cmd = shortcuts.get(key)

    if cmd:
        os.system(cmd)
        return f"Launched {app_name}."

    # Try as a file / folder / URL — os.startfile handles all of these on Windows
    expanded = os.path.expanduser(app_name)
    if os.path.exists(expanded):
        try:
            os.startfile(expanded)
            return f"Opened {expanded}."
        except Exception as e:
            return f"Failed to open '{expanded}': {e}"

    # Last resort: try running it as a command
    try:
        subprocess.Popen(app_name, shell=True)
        return f"Attempted to launch '{app_name}'."
    except Exception as e:
        return f"Couldn't open '{app_name}': {e}"


# ---------------------------------------------------------------------------
# TOOL 2: File Management — organize
# ---------------------------------------------------------------------------
def organize_files(source_dir: str, extension: str, folder_name: str) -> str:
    """
    Moves files of a specific type (e.g., .sch, .pdf) into a subfolder.

    Args:
        source_dir: The directory to scan for files.
        extension: The file extension to filter by (e.g. '.pdf', '.txt').
        folder_name: The name of the subfolder to move matching files into.

    Returns:
        A summary of how many files were moved.
    """
    if not os.path.isdir(source_dir):
        return f"Error: Directory '{source_dir}' does not exist."

    path = os.path.join(source_dir, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)

    moved_count = 0
    for file in os.listdir(source_dir):
        if file.endswith(extension):
            shutil.move(os.path.join(source_dir, file), os.path.join(path, file))
            moved_count += 1
    return f"Moved {moved_count} '{extension}' files into '{folder_name}/'."


# ---------------------------------------------------------------------------
# TOOL 3: Local Note Search (keyword fallback — see rag.py for vector search)
# ---------------------------------------------------------------------------
def search_notes(query: str) -> str:
    """
    Searches through a folder of .txt or .md notes for a keyword or phrase.

    Args:
        query: The search term to look for inside note files.

    Returns:
        A list of matching filenames, or a message if none matched.
    """
    notes_path = os.path.expanduser("~/Documents/Notes")
    if not os.path.isdir(notes_path):
        return f"Notes directory not found at '{notes_path}'. Please create it or update the path."

    results = []
    for file in os.listdir(notes_path):
        if file.endswith(".txt") or file.endswith(".md"):
            try:
                with open(os.path.join(notes_path, file), 'r', encoding='utf-8') as f:
                    if query.lower() in f.read().lower():
                        results.append(file)
            except Exception:
                continue
    return f"Found matches in: {', '.join(results)}" if results else "No matching notes found."


# ---------------------------------------------------------------------------
# TOOL 4: System Stats
# ---------------------------------------------------------------------------
def get_system_stats() -> str:
    """
    Returns current system resource usage including CPU, memory, disk,
    and the top processes by RAM and CPU consumption.

    Returns:
        A formatted string with system stats and top resource-hungry processes.
    """
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    lines = [
        f"CPU: {cpu}% | RAM: {mem.percent}% ({mem.used / (1024**3):.1f}GB / {mem.total / (1024**3):.1f}GB) | Disk: {disk.percent}% ({disk.used / (1024**3):.1f}GB / {disk.total / (1024**3):.1f}GB)",
        "",
    ]

    # Gather per-process info (skip access-denied)
    procs = []
    for p in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
        try:
            info = p.info
            ram_mb = (info['memory_info'].rss / (1024 ** 2)) if info['memory_info'] else 0
            procs.append({
                'name': info['name'] or '?',
                'pid': info['pid'],
                'ram_mb': ram_mb,
                'cpu': info['cpu_percent'] or 0.0,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Top 8 by RAM
    top_ram = sorted(procs, key=lambda p: p['ram_mb'], reverse=True)[:8]
    lines.append("Top RAM consumers:")
    for p in top_ram:
        lines.append(f"  {p['name']:<28} {p['ram_mb']:>7.1f} MB  (PID {p['pid']})")

    # Top 5 by CPU
    top_cpu = sorted(procs, key=lambda p: p['cpu'], reverse=True)[:5]
    lines.append("")
    lines.append("Top CPU consumers:")
    for p in top_cpu:
        lines.append(f"  {p['name']:<28} {p['cpu']:>5.1f}%     (PID {p['pid']})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TOOL 5: Run Shell Command
# ---------------------------------------------------------------------------
def run_shell_command(command: str) -> str:
    """
    Executes a shell command on the local system and returns the output.
    Use this to run scripts, git commands, compilers, simulators, etc.

    Args:
        command: The shell command string to execute (e.g. 'git status', 'python sim.py').

    Returns:
        The stdout/stderr output of the command, or an error message.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.path.expanduser("~"),
        )
        output = result.stdout.strip()
        errors = result.stderr.strip()
        combined = ""
        if output:
            combined += output
        if errors:
            combined += ("\n" + errors) if combined else errors
        if not combined:
            combined = "(command completed with no output)"
        # Truncate very long output to stay within context limits
        if len(combined) > 4000:
            combined = combined[:4000] + "\n... (truncated)"
        return combined
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds."
    except Exception as e:
        return f"Error running command: {e}"


# ---------------------------------------------------------------------------
# TOOL 6: Read File
# ---------------------------------------------------------------------------
def read_file(file_path: str) -> str:
    """
    Reads and returns the contents of a text file.

    Args:
        file_path: The absolute or relative path to the file to read.

    Returns:
        The file contents, or an error message if the file cannot be read.
    """
    expanded = os.path.expanduser(file_path)
    if not os.path.isfile(expanded):
        return f"Error: File '{expanded}' does not exist."
    try:
        with open(expanded, 'r', encoding='utf-8') as f:
            content = f.read()
        if len(content) > 8000:
            content = content[:8000] + "\n... (truncated — file is very large)"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


# ---------------------------------------------------------------------------
# TOOL 7: Write File
# ---------------------------------------------------------------------------
def write_file(file_path: str, content: str) -> str:
    """
    Creates or overwrites a file with the given content.

    Args:
        file_path: The absolute or relative path to the file to write.
        content: The text content to write into the file.

    Returns:
        A confirmation message or an error message.
    """
    expanded = os.path.expanduser(file_path)
    try:
        parent = os.path.dirname(expanded)
        if parent and not os.path.exists(parent):
            os.makedirs(parent)
        with open(expanded, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Wrote {len(content)} characters to '{expanded}'."
    except Exception as e:
        return f"Error writing file: {e}"


# ---------------------------------------------------------------------------
# TOOL 8: Web Search
# ---------------------------------------------------------------------------
def web_search(query: str) -> str:
    """
    Searches the web using DuckDuckGo and returns the top results.
    Use this to look up datasheets, component specs, error messages, etc.

    Args:
        query: The search query string.

    Returns:
        A formatted list of search results with titles, URLs, and snippets.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No search results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r['title']}**\n   {r['href']}\n   {r['body']}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Web search failed: {e}"


# ---------------------------------------------------------------------------
# TOOL 9: Semantic Note Search (vector RAG via ChromaDB)
# ---------------------------------------------------------------------------
def semantic_search_notes(query: str) -> str:
    """
    Searches notes using semantic similarity (embeddings) rather than exact keywords.
    Finds relevant notes even when the exact words don't match.

    Args:
        query: A natural-language description of what you're looking for.

    Returns:
        The most relevant note excerpts, or an error/status message.
    """
    try:
        from rag import query_notes
        return query_notes(query)
    except ImportError:
        return "RAG module not available — falling back to keyword search."
    except Exception as e:
        return f"Semantic search failed: {e}"


# ---------------------------------------------------------------------------
# TOOL 10: Clear Temp Files
# ---------------------------------------------------------------------------
def clear_temp_files() -> str:
    """
    Clears temporary files from common Windows temp directories.
    Removes files from %TEMP%, C:\\Windows\\Temp, and browser caches.

    Returns:
        A summary of how many files/folders were removed and space freed.
    """
    import tempfile

    temp_dirs = [
        tempfile.gettempdir(),                         # %TEMP% / user temp
        os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "Temp"),
    ]

    removed = 0
    freed = 0
    errors = 0
    skipped_dirs = []

    for tdir in temp_dirs:
        if not os.path.isdir(tdir):
            continue
        try:
            entries = os.listdir(tdir)
        except PermissionError:
            skipped_dirs.append(tdir)
            continue
        for entry in entries:
            path = os.path.join(tdir, entry)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    size = os.path.getsize(path)
                    os.unlink(path)
                    freed += size
                    removed += 1
                elif os.path.isdir(path):
                    size = sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, _, fns in os.walk(path)
                        for f in fns
                        if os.path.exists(os.path.join(dp, f))
                    )
                    shutil.rmtree(path, ignore_errors=True)
                    freed += size
                    removed += 1
            except Exception:
                errors += 1

    freed_mb = freed / (1024 * 1024)
    parts = [f"Removed {removed} items, freed {freed_mb:.1f} MB."]
    if errors:
        parts.append(f"{errors} items were locked/skipped.")
    if skipped_dirs:
        parts.append(f"Access denied to: {', '.join(skipped_dirs)} — run Velma as admin (velma_admin.bat) for full access.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# TOOL 11: List Directory Contents
# ---------------------------------------------------------------------------
def list_directory(directory_path: str) -> str:
    """
    Lists the contents of a directory — files, folders, sizes, and modification dates.

    Args:
        directory_path: The absolute or relative path to the directory to list.

    Returns:
        A formatted listing of the directory contents, or an error message.
    """
    from datetime import datetime

    expanded = os.path.expanduser(directory_path)
    if not os.path.isdir(expanded):
        return f"Error: '{expanded}' is not a directory or doesn't exist."

    try:
        entries = os.listdir(expanded)
    except PermissionError:
        return f"Error: Access denied to '{expanded}'. Try running Velma as admin."

    if not entries:
        return f"'{expanded}' is empty."

    lines = [f"Contents of {expanded}  ({len(entries)} items):", ""]

    dirs = []
    files = []
    for name in sorted(entries):
        full = os.path.join(expanded, name)
        try:
            stat = os.stat(full)
            mod = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            if os.path.isdir(full):
                dirs.append(f"  📁 {name + '/':<40} {mod}")
            else:
                size = stat.st_size
                if size < 1024:
                    sz = f"{size} B"
                elif size < 1024 ** 2:
                    sz = f"{size / 1024:.1f} KB"
                elif size < 1024 ** 3:
                    sz = f"{size / (1024**2):.1f} MB"
                else:
                    sz = f"{size / (1024**3):.1f} GB"
                files.append(f"  📄 {name:<40} {sz:>10}  {mod}")
        except (PermissionError, OSError):
            files.append(f"  ⚠️  {name:<40} (access denied)")

    lines.extend(dirs)
    if dirs and files:
        lines.append("")
    lines.extend(files)

    result = "\n".join(lines)
    if len(result) > 6000:
        result = result[:6000] + "\n... (truncated — too many entries)"
    return result


# ---------------------------------------------------------------------------
# TOOL 12: Schedule Reminder
# ---------------------------------------------------------------------------
def schedule_reminder(description: str, minutes: int = 0, time_str: str = "") -> str:
    """
    Schedule a one-shot reminder. Specify either minutes from now, or an
    absolute time string like '15:30' or '3:00 PM'.

    Args:
        description: What to remind the user about.
        minutes: Minutes from now to fire the reminder (e.g. 10). Use 0 if using time_str instead.
        time_str: Absolute time like '15:30' or '3:00 PM'. Ignored if minutes > 0.

    Returns:
        Confirmation message.
    """
    # Coerce in case the model sends minutes as a string
    try:
        minutes = int(minutes) if minutes else 0
    except (ValueError, TypeError):
        minutes = 0

    if minutes and minutes > 0:
        fire_at = datetime.now() + timedelta(minutes=minutes)
    elif time_str:
        today = datetime.now().date()
        for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p"):
            try:
                t = datetime.strptime(time_str.strip(), fmt).time()
                fire_at = datetime.combine(today, t)
                if fire_at < datetime.now():
                    fire_at += timedelta(days=1)  # tomorrow if time already passed
                break
            except ValueError:
                continue
        else:
            return f"Couldn't parse time '{time_str}'. Use formats like '15:30' or '3:00 PM'."
    else:
        return "Please specify either 'minutes' or 'time_str' for the reminder."

    delay = (fire_at - datetime.now()).total_seconds()
    if delay < 0:
        delay = 0

    def _fire():
        if _reminder_callback:
            _reminder_callback(description)

    timer = threading.Timer(delay, _fire)
    timer.daemon = True
    timer.start()

    _scheduled_reminders.append({
        "description": description,
        "fire_at": fire_at.strftime("%Y-%m-%d %H:%M"),
    })

    return f"Reminder set for {fire_at.strftime('%H:%M')}: {description}"


def list_reminders() -> str:
    """
    Lists all scheduled reminders.

    Returns:
        A list of pending reminders, or a message if none are set.
    """
    now = datetime.now()
    # Filter out expired ones
    active = [r for r in _scheduled_reminders
              if datetime.strptime(r['fire_at'], "%Y-%m-%d %H:%M") > now]
    if not active:
        return "No pending reminders."
    lines = ["Pending reminders:"]
    for r in active:
        lines.append(f"  ⏰ {r['fire_at']} — {r['description']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Master list — import this in main.py
# ---------------------------------------------------------------------------
ALL_TOOLS = {
    'launch_application': launch_application,
    'list_directory': list_directory,
    'organize_files': organize_files,
    'search_notes': search_notes,
    'get_system_stats': get_system_stats,
    'run_shell_command': run_shell_command,
    'read_file': read_file,
    'write_file': write_file,
    'web_search': web_search,
    'semantic_search_notes': semantic_search_notes,
    'clear_temp_files': clear_temp_files,
    'schedule_reminder': schedule_reminder,
    'list_reminders': list_reminders,
}

# OpenAI-compatible tool schemas (used by Groq / OpenAI / etc.)
ALL_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "launch_application",
            "description": "Opens an application, file, folder, or URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "An app shortcut name, file path, folder path, or URL to open."}
                },
                "required": ["app_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "Lists the contents of a directory — files, folders, sizes, and modification dates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory_path": {"type": "string", "description": "The absolute or relative path to the directory to list."}
                },
                "required": ["directory_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "organize_files",
            "description": "Moves files of a specific type into a subfolder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_dir": {"type": "string", "description": "The directory to scan for files."},
                    "extension": {"type": "string", "description": "The file extension to filter by (e.g. '.pdf')."},
                    "folder_name": {"type": "string", "description": "The name of the subfolder to move matching files into."}
                },
                "required": ["source_dir", "extension", "folder_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Searches through .txt or .md notes for a keyword or phrase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search term to look for inside note files."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_stats",
            "description": "Returns current system resource usage including CPU, memory, disk, and top processes.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Executes a shell command on the local system and returns the output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command string to execute."}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads and returns the contents of a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The absolute or relative path to the file to read."}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Creates or overwrites a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The path to the file to write."},
                    "content": {"type": "string", "description": "The text content to write into the file."}
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web using DuckDuckGo and returns the top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query string."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search_notes",
            "description": "Searches notes using semantic similarity rather than exact keywords.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A natural-language description of what you're looking for."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clear_temp_files",
            "description": "Clears temporary files from common Windows temp directories.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_reminder",
            "description": "Schedule a one-shot reminder. Specify either minutes from now, or an absolute time string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "What to remind the user about."},
                    "minutes": {"type": "string", "description": "Minutes from now to fire the reminder (e.g. '10'). Use '0' if using time_str instead."},
                    "time_str": {"type": "string", "description": "Absolute time like '15:30' or '3:00 PM'. Ignored if minutes > 0."}
                },
                "required": ["description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "Lists all scheduled reminders.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
]