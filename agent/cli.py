"""PyMentor interactive command-line interface (the entry point of agent.exe)."""

from __future__ import annotations

import datetime as dt
import getpass
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

from agent import __version__

HELP_TEXT = """
PyMentor - your Python tutor
============================
Just type a Python question and press Enter. Examples:
  What is the difference between a list and a tuple?
  Why do I get: TypeError: can only concatenate str (not "int") to str
  Run this code: ```python
  print(sum(range(5)))
  ```

Commands:
  help  or  ?        show this guide
  file <path>        attach a local file (.py .txt .md .log) to your next questions
  file               show the attached file
  clear              detach the file
  model              show the current model
  model <id>         switch model, e.g.  model openai/gpt-4o-mini
  models             list the free models available on OpenRouter right now
  tools              show the agent's active extensions
  cost               show tokens and cost used in this session
  summary            save a summary of this conversation to a text file
  exit  or  quit     leave

Safety: PyMentor never runs code without asking you first [y/N].
Multi-line code: paste it between ``` lines, or put it in a file and use 'file <path>'.
"""

TOOLS_TEXT = """Active extensions:
  [on]  RAG         - searches a built-in knowledge base of Python errors and concepts
  [on]  Code-Exec   - runs Python code in a restricted subprocess, only after your approval
  [on]  File input  - reads a local file you attach with 'file <path>'
  [off] MCP         - not used by this agent (no external accounts needed)
  [off] Web-Search  - not used (answers come from the model + knowledge base)"""


# ------------------------------------------------------------------ terminal output

class Style:
    enabled = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
    RESET, DIM, BOLD, RED, GREEN, YELLOW, CYAN = "\033[0m", "\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[36m"

    @classmethod
    def paint(cls, text: str, color: str) -> str:
        return f"{color}{text}{cls.RESET}" if cls.enabled else text


def info(text: str) -> None:
    print(Style.paint(f"[INFO] {text}", Style.DIM))


def error(text: str) -> None:
    print(Style.paint(f"[ERROR] {text}", Style.RED))


def setup_console() -> None:
    """UTF-8 output (Hebrew) and ANSI colours on Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if os.name == "nt":
        os.system("")  # enables ANSI escape codes in the Windows console


# ------------------------------------------------------------------ configuration

def load_configuration() -> None:
    """Load .env from the current folder and from next to agent.exe; ask for a missing key."""
    load_dotenv(Path.cwd() / ".env")
    if getattr(sys, "frozen", False):
        load_dotenv(Path(sys.executable).parent / ".env")
    if not os.getenv("OPENROUTER_API_KEY"):
        print("No OPENROUTER_API_KEY found (.env file or environment variable).")
        print("Get a free key at https://openrouter.ai/keys")
        key = getpass.getpass("Paste your OpenRouter API key (input hidden): ").strip()
        if not key:
            raise SystemExit("An OpenRouter API key is required. Exiting.")
        os.environ["OPENROUTER_API_KEY"] = key
    # LangSmith: accept both the new (LANGSMITH_*) and the older (LANGCHAIN_*) variable names.
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" and not os.getenv("LANGSMITH_TRACING"):
        os.environ["LANGSMITH_TRACING"] = "true"
    from agent.tracing import resolve_langsmith_target

    notice = resolve_langsmith_target()
    if notice:
        print(notice)
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true" and not (
        os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    ):
        os.environ["LANGSMITH_TRACING"] = "false"
        print("LangSmith tracing was requested but LANGSMITH_API_KEY is missing - tracing is OFF.")


def langsmith_status() -> str:
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        from agent.tracing import describe_target

        return f"ON ({describe_target()})"
    return "OFF (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env)"


# ------------------------------------------------------------------ session

class Session:
    def __init__(self) -> None:
        from agent.graph import build_graph
        from agent.llm import default_model

        self.graph = build_graph()
        self.model = default_model()
        self.file_name: str | None = None
        self.file_content: str | None = None
        self.history: list[dict] = []
        self.usage: list[dict] = []

    # ---- commands
    def handle_command(self, line: str) -> bool:
        """Run a CLI command. Returns True if the line was a command."""
        command, _, argument = line.partition(" ")
        command, argument = command.lower(), argument.strip()
        if command in ("help", "?", "/help"):
            print(HELP_TEXT)
        elif command == "file":
            self.attach_file(argument)
        elif command == "clear":
            self.file_name = self.file_content = None
            print("File detached.")
        elif command == "model":
            if argument:
                self.model = argument
                print(f"Model switched to: {self.model}")
            else:
                print(f"Current model: {self.model}")
        elif command == "models":
            self.show_free_models()
        elif command == "tools":
            print(TOOLS_TEXT)
        elif command == "cost":
            self.show_cost()
        elif command == "summary":
            self.export_summary()
        else:
            return False
        return True

    def attach_file(self, raw_path: str) -> None:
        from agent.tools import FileInputError, read_local_file

        if not raw_path:
            print(f"Attached file: {self.file_name}" if self.file_name else "No file attached. Use: file <path>")
            return
        try:
            path, content = read_local_file(raw_path)
        except FileInputError as problem:
            error(str(problem))
            return
        self.file_name, self.file_content = path.name, content
        print(Style.paint(f"Attached {path.name} ({len(content.splitlines())} lines). "
                          "Now ask a question about it, e.g. 'why does this crash?'", Style.GREEN))

    def show_free_models(self) -> None:
        from agent.pricing import list_free_models

        info("Fetching the model list from OpenRouter...")
        try:
            models = list_free_models()
        except Exception as problem:
            error(f"Could not fetch the model list: {problem}")
            return
        print(f"{len(models)} free models right now (switch with: model <id>):")
        for model_id in models:
            print(f"  {model_id}")

    def show_cost(self) -> None:
        from agent.pricing import cost_usd, prices_for

        if not self.usage:
            print("No model calls yet.")
            return
        totals: dict[str, list[int]] = {}
        for record in self.usage:
            pair = totals.setdefault(record["model"], [0, 0])
            pair[0] += record["input_tokens"]
            pair[1] += record["output_tokens"]
        for model_id, (tokens_in, tokens_out) in totals.items():
            line = f"  {model_id}: {tokens_in:,} input + {tokens_out:,} output tokens"
            try:
                prices = prices_for(model_id)
                line += (f"  ->  actual ${cost_usd(prices['actual'], tokens_in, tokens_out):.6f}"
                         f", theoretical ${cost_usd(prices['theoretical'], tokens_in, tokens_out):.6f}")
            except Exception:
                line += "  (price unavailable)"
            print(line)

    def export_summary(self) -> None:
        from agent.llm import LLMError, call_llm

        if not self.history:
            print("Nothing to summarize yet.")
            return
        transcript = "\n\n".join(f"Student: {turn['question']}\nPyMentor: {turn['answer']}" for turn in self.history)
        info("Writing the summary...")
        try:
            summary, record = call_llm(
                "summary", self.model,
                "Summarize this Python tutoring session for the student: the topics covered, "
                "the mistakes found and how they were fixed, and what to practise next. "
                "Use short bullet points, in the language the student used.",
                transcript[-20000:],
            )
            self.usage.append(record)
        except LLMError as problem:
            error(f"{problem} - saving the transcript without an AI summary.")
            summary = "(AI summary unavailable)"
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path.cwd() / f"pymentor_summary_{stamp}.txt"
        path.write_text(f"PyMentor session summary - {stamp}\n\n{summary}\n\n"
                        f"{'=' * 60}\nFull transcript\n{'=' * 60}\n\n{transcript}\n", encoding="utf-8")
        print(Style.paint(f"Summary saved to: {path}", Style.GREEN))

    # ---- one question
    def ask(self, question: str) -> None:
        from agent.runner import NODE_LABELS, run_turn

        last_started: list[str] = []

        def on_start(node: str) -> None:
            if last_started and last_started[-1] == node:
                return   # LangGraph re-enters the HITL node after the [y/N] answer
            last_started.append(node)
            info(f"Entering Node: {NODE_LABELS.get(node, node)}")

        def on_end(node: str, result: dict) -> None:
            if node == "evaluator" and "eval_score" in result:
                info(f"Evaluator score: {result['eval_score']}/5")
            elif node == "retrieve":
                titles = [doc["title"] for doc in result.get("context", [])]
                info(f"RAG found: {', '.join(titles) if titles else 'nothing relevant'}")
            elif node == "input_guardrail" and result.get("blocked"):
                info(f"Blocked by guardrail: {result.get('block_reason')}")

        state = run_turn(
            self.graph,
            {"question": question, "model": self.model,
             "file_name": self.file_name, "file_content": self.file_content},
            approve=ask_code_approval,
            on_node_start=on_start,
            on_node_end=on_end,
        )
        self.usage.extend(state.get("usage", []))
        if state.get("code_output"):
            print(Style.paint("\n--- Real output of your code ---", Style.CYAN))
            print(state["code_output"])
        print(Style.paint("\nPyMentor:", Style.BOLD + Style.GREEN))
        print(state.get("answer", ""))
        print()
        self.history.append({"question": question, "answer": state.get("answer", "")})


def ask_code_approval(payload: dict) -> str:
    """HITL prompt in the terminal: show the code and ask [y/N]."""
    print(Style.paint("\n=== Approval needed: PyMentor wants to RUN this code ===", Style.YELLOW + Style.BOLD))
    print(payload.get("code", ""))
    print(Style.paint("=" * 56, Style.YELLOW))
    for warning in payload.get("warnings", []):
        print(Style.paint(f"WARNING: this code {warning}", Style.RED))
    while True:
        try:
            answer = input("Run this code on your computer? [y/N]: ").strip().lower()
        except EOFError:
            return "n"
        if answer in ("", "n", "no", "לא"):
            print("OK, the code will NOT be run.")
            return "n"
        if answer in ("y", "yes", "כן"):
            return "y"
        print("Please answer y or n.")


def read_question() -> str | None:
    """Read one line; if it opens a ``` block, keep reading until the closing ```."""
    try:
        line = input(Style.paint("You> ", Style.BOLD + Style.CYAN))
    except EOFError:
        return None
    if line.count("```") % 2 == 1:
        lines = [line]
        while True:
            try:
                more = input("... ")
            except EOFError:
                break
            lines.append(more)
            if "```" in more:
                break
        line = "\n".join(lines)
    return line.strip()


def main() -> int:
    setup_console()
    load_configuration()
    from agent.llm import LLMError

    print(Style.paint(f"PyMentor v{__version__} - a Python tutor agent (LangGraph + OpenRouter)", Style.BOLD))
    try:
        session = Session()
    except Exception as problem:
        error(f"Could not start the agent: {problem}")
        return 1
    print(f"Model: {session.model}   |   LangSmith tracing: {langsmith_status()}")
    print("Type 'help' or '?' for the guide, 'exit' to quit.\n")

    while True:
        try:
            question = read_question()
        except KeyboardInterrupt:
            question = None
        if question is None or question.lower() in ("exit", "quit", "/exit"):
            print("Goodbye! Keep coding.")
            return 0
        if not question:
            continue
        if session.handle_command(question):
            continue
        try:
            session.ask(question)
        except LLMError as problem:
            error(str(problem))
        except KeyboardInterrupt:
            print("\n[Cancelled]")
        except Exception as problem:  # the CLI must never crash on a single bad turn
            error(f"Unexpected problem: {type(problem).__name__}: {problem}")
            if os.getenv("PYMENTOR_DEBUG"):
                traceback.print_exc()
            else:
                print("(Set PYMENTOR_DEBUG=1 to see technical details.)")


if __name__ == "__main__":
    sys.exit(main())
