"""Agent extensions: RAG (knowledge search), Code-Exec (run Python), local file input.

Design notes
- RAG uses lexical BM25 scoring, not embeddings. Tradeoff: no embedding API cost,
  works offline, deterministic and easy to test; weaker on synonyms. Good enough for a
  small, focused knowledge base of Python errors whose names appear verbatim in tracebacks.
- Code-Exec runs in a *restricted subprocess*, not a real sandbox: timeout, isolated
  mode (-I), temporary working folder, minimal environment, output cap. The real
  safety control is HITL: the graph never runs code without the user's explicit "y".
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from langsmith import traceable  # no-op unless LANGSMITH_TRACING=true

from agent.knowledge import DOCUMENTS

# ---------------------------------------------------------------- RAG (BM25)

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "to", "of", "in", "and", "or", "it", "you", "i",
    "my", "me", "do", "does", "how", "what", "why", "this", "that", "with", "for",
    "on", "be", "can", "not", "use", "when", "if", "python",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in (w.lower() for w in _TOKEN_RE.findall(text)) if t not in _STOPWORDS]


_DOC_TOKENS = [_tokenize(d["title"] + " " + d["title"] + " " + d["text"]) for d in DOCUMENTS]
_AVG_LEN = sum(len(t) for t in _DOC_TOKENS) / len(_DOC_TOKENS)
_DOC_FREQ: dict[str, int] = {}
for _tokens in _DOC_TOKENS:
    for _term in set(_tokens):
        _DOC_FREQ[_term] = _DOC_FREQ.get(_term, 0) + 1


@traceable(name="rag_search_knowledge", run_type="tool")
def search_knowledge(query: str, k: int = 3) -> list[dict[str, str]]:
    """Return the top-k knowledge documents for the query (BM25). Empty if nothing matches."""
    terms = _tokenize(query)
    if not terms:
        return []
    n_docs = len(DOCUMENTS)
    k1, b = 1.5, 0.75
    scored = []
    for doc, tokens in zip(DOCUMENTS, _DOC_TOKENS):
        score = 0.0
        for term in set(terms):
            tf = tokens.count(term)
            if tf == 0:
                continue
            df = _DOC_FREQ[term]
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(tokens) / _AVG_LEN))
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in scored[:k]]


# ---------------------------------------------------------------- Code execution

MAX_OUTPUT_CHARS = 4000
DEFAULT_TIMEOUT_SECONDS = 10

# Patterns shown as warnings in the approval prompt. They do not block: the user decides.
RISKY_PATTERNS = {
    r"\bos\.(system|remove|rmdir|unlink|popen)\b": "deletes files or runs shell commands (os)",
    r"\bshutil\.(rmtree|move)\b": "deletes or moves folders (shutil)",
    r"\bsubprocess\b": "starts other programs (subprocess)",
    r"\bsocket\b|\brequests\b|\burllib\b": "uses the network",
    r"\bopen\([^)]*['\"][wa]": "writes to files",
    r"\beval\(|\bexec\(": "runs dynamic code (eval/exec)",
}


def risky_operations(code: str) -> list[str]:
    """Human-readable list of risky operations detected in the code (for the HITL prompt)."""
    return [label for pattern, label in RISKY_PATTERNS.items() if re.search(pattern, code)]


def find_python_interpreter() -> str | None:
    """Find a Python to run user code with.

    Inside agent.exe, sys.executable is the exe itself, so we must look on PATH instead.
    """
    if not getattr(sys, "frozen", False):
        return sys.executable
    for name in ("python", "python3", "py"):
        found = shutil.which(name)
        if found:
            return found
    return None


@traceable(name="code_exec_run_python", run_type="tool")
def run_python_code(code: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    """Run Python code in a restricted subprocess and return a printable result."""
    interpreter = find_python_interpreter()
    if interpreter is None:
        return ("[code-exec unavailable] No Python interpreter was found on this computer. "
                "Install Python from python.org to let PyMentor run code.")

    with tempfile.TemporaryDirectory(prefix="pymentor_") as workdir:
        script = Path(workdir) / "student_code.py"
        script.write_text(code, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),  # required by Python on Windows
            "PYTHONIOENCODING": "utf-8",
        }
        try:
            completed = subprocess.run(
                [interpreter, "-I", str(script)],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return f"[timeout] The code ran longer than {timeout} seconds and was stopped (infinite loop?)."

    output = (completed.stdout or "") + (completed.stderr or "")
    output = output.replace(str(script), "student_code.py")
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"
    return f"[exit code {completed.returncode}]\n{output.strip() or '(no output)'}"


# ---------------------------------------------------------------- Local files

ALLOWED_EXTENSIONS = {".py", ".txt", ".md", ".log"}
MAX_FILE_BYTES = 200_000


class FileInputError(ValueError):
    """A user-facing error about a file path."""


def read_local_file(raw_path: str) -> tuple[Path, str]:
    """Validate and read a local text file given by the user. Returns (path, content)."""
    # Drag-and-drop into a terminal wraps paths in quotes; strip them.
    cleaned = raw_path.strip().strip("'\"").strip()
    if not cleaned:
        raise FileInputError("Please give a file path, for example: file C:\\homework\\ex1.py")
    path = Path(cleaned).expanduser()
    if not path.exists():
        raise FileInputError(f"File not found: {path}")
    if not path.is_file():
        raise FileInputError(f"This is a folder, not a file: {path}")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise FileInputError(f"Unsupported file type '{path.suffix}'. Allowed: {allowed}")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise FileInputError(f"File is too large ({size:,} bytes). Limit: {MAX_FILE_BYTES:,} bytes.")
    return path, path.read_text(encoding="utf-8", errors="replace")
