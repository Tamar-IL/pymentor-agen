"""Shared helpers for the eval scripts: config, running the real agent, the judge, CSV."""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # allow `python evals/xxx.py` from the project root

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from agent.tracing import resolve_langsmith_target  # noqa: E402

_notice = resolve_langsmith_target()
if _notice:
    print(_notice)

from agent.graph import build_graph  # noqa: E402
from agent.llm import LLMError, call_llm, parse_json_object  # noqa: E402
from agent.runner import run_turn  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RUBRIC_PATH = Path(__file__).resolve().parent / "rubric.md"

# One paid model is required by the assignment (cost eval). Override in .env.
FREE_MODELS = [m.strip() for m in os.getenv(
    # gemma-4-31b-it:free is left out: it was rate-limited on every call in real runs.
    "EVAL_FREE_MODELS", "nvidia/nemotron-3.5-lightning:free").split(",") if m.strip()]
PAID_MODELS = [m.strip() for m in os.getenv("EVAL_PAID_MODELS", "openai/gpt-4o-mini").split(",") if m.strip()]
ALL_MODELS = FREE_MODELS + PAID_MODELS
# The judge is ONE fixed model for every comparison, so all answers are graded the same way.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini")


RATE_LIMIT_RETRIES = 3          # free models are often rate-limited for a short while
RATE_LIMIT_WAIT_SECONDS = 20


def run_agent(question: str, model: str, approve_code: bool = True,
              use_fallback: bool = False) -> dict:
    """Run the real agent graph once, non-interactively.

    Code approval is scripted ("y") because the eval questions are our own fixed,
    harmless snippets; the interactive CLI always asks the human.
    use_fallback=False measures exactly the named model (model comparison, cost);
    True measures the agent as users get it, backup models included (scope eval).
    """
    if use_fallback:
        os.environ.pop("PYMENTOR_FALLBACK_MODELS", None)   # the agent's default backups
    else:
        os.environ["PYMENTOR_FALLBACK_MODELS"] = ""
    start = time.perf_counter()
    state, error = {}, ""
    for attempt in range(1, RATE_LIMIT_RETRIES + 1):
        try:
            state = run_turn(build_graph(), {"question": question, "model": model},
                             approve=lambda payload: "y" if approve_code else "n")
            error = ""
            break
        except LLMError as problem:
            state, error = {}, str(problem)
            if "rate-limited" not in error or attempt == RATE_LIMIT_RETRIES:
                break
            wait = RATE_LIMIT_WAIT_SECONDS * attempt
            print(f"   rate-limited, retrying in {wait}s (attempt {attempt}/{RATE_LIMIT_RETRIES})", flush=True)
            time.sleep(wait)
    return {
        "state": state,
        "answer": state.get("answer", ""),
        "blocked": bool(state.get("blocked")),
        "usage": state.get("usage", []),
        "seconds": round(time.perf_counter() - start, 2),
        "error": error,
    }


JUDGE_SYSTEM = """You are an impartial evaluator. Grade the AI tutor's ANSWER using the RUBRIC.
Be strict and consistent. Reply with JSON only, exactly these keys (integers 1-5):
{"role_relevance": n, "factual_accuracy": n, "completeness": n, "format_compliance": n,
 "overall": n, "justification": "<one sentence>"}"""

JUDGE_KEYS = ["role_relevance", "factual_accuracy", "completeness", "format_compliance", "overall"]


def judge(question: str, answer: str, reference: str) -> dict:
    """LLM-as-judge with the fixed JUDGE_MODEL and the rubric from evals/rubric.md."""
    rubric = RUBRIC_PATH.read_text(encoding="utf-8")
    prompt = (f"RUBRIC:\n{rubric}\n\nQUESTION:\n{question}\n\nREFERENCE NOTES (facts a good answer contains):\n"
              f"{reference}\n\nANSWER TO GRADE:\n{answer}")
    os.environ["PYMENTOR_FALLBACK_MODELS"] = ""
    reply, _ = call_llm("judge", JUDGE_MODEL, JUDGE_SYSTEM, prompt, temperature=0)
    parsed = parse_json_object(reply) or {}
    scores = {}
    for key in JUDGE_KEYS:
        try:
            scores[key] = max(1, min(5, int(parsed[key])))
        except (KeyError, TypeError, ValueError):
            scores[key] = ""
    scores["justification"] = str(parsed.get("justification", "judge returned invalid JSON"))[:300]
    return scores


def write_csv(name: str, rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    fields: list[str] = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:   # utf-8-sig: opens well in Excel
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {path}")
    return path


def require_api_key() -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is missing. Create a .env file (see .env.example).")
