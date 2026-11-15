"""Run the three eval scripts offline (fake model + fake prices) and check their CSVs."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

import common  # noqa: E402
import compare_models  # noqa: E402
import cost_eval  # noqa: E402
import scope_eval  # noqa: E402
from agent import pricing  # noqa: E402

CATALOG = {
    "free/model:free": {"input": 0.0, "output": 0.0, "name": "free"},
    "free/model": {"input": 1e-7, "output": 2e-7, "name": "free twin"},
    "paid/model": {"input": 1e-6, "output": 2e-6, "name": "paid"},
}


@pytest.fixture
def eval_env(fake_llm, monkeypatch, tmp_path):
    replies, _ = fake_llm
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pricing, "fetch_model_catalog", lambda: CATALOG)
    for module in (common, compare_models, cost_eval, scope_eval):
        if hasattr(module, "ALL_MODELS"):
            monkeypatch.setattr(module, "ALL_MODELS", ["free/model:free", "paid/model"])
        if hasattr(module, "FREE_MODELS"):
            monkeypatch.setattr(module, "FREE_MODELS", ["free/model:free"])
        if hasattr(module, "PAID_MODELS"):
            monkeypatch.setattr(module, "PAID_MODELS", ["paid/model"])
    return replies, tmp_path


def read(path):
    with open(path, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_compare_models_writes_judge_scores(eval_env):
    replies, out = eval_env
    replies["scope"] = '{"in_scope": true}'
    # The judge prompt is not one of the agent's prompts, so the fake treats it as "draft".
    replies["draft"] = ('Answer. Tip: x {"role_relevance": 5, "factual_accuracy": 4, "completeness": 4, '
                        '"format_compliance": 5, "overall": 4, "justification": "fine"}')
    compare_models.main()
    rows = read(out / "compare_models.csv")
    runs = [r for r in rows if r["task_id"] != "AVERAGE"]
    assert len(runs) == 2 * len(compare_models.TASKS)
    assert all(r["overall"] == "4" and r["seconds"] for r in runs)
    assert {r["model"] for r in rows if r["task_id"] == "AVERAGE"} == {"free/model:free", "paid/model"}


def test_cost_eval_has_per_step_rows_and_theoretical_cost_for_free(eval_env):
    _, out = eval_env
    cost_eval.main()
    rows = read(out / "cost_eval.csv")
    free_steps = [r for r in rows if r["model"] == "free/model:free" and r["step"]]
    assert free_steps and all(float(r["actual_cost_usd"]) == 0 for r in free_steps)
    assert all(float(r["theoretical_cost_usd"]) > 0 for r in free_steps)
    paid_total = [r for r in rows if r["model"] == "paid/model" and r["node"] == "GRAND TOTAL"][0]
    assert float(paid_total["actual_cost_usd"]) > 0
    assert {r["node"] for r in free_steps} >= {"input_guardrail", "classify", "draft", "evaluator"}


def test_scope_eval_reports_refusal_rate(eval_env):
    replies, out = eval_env
    # Simulate a model that refuses exactly the out-of-scope list.
    out_questions = set(scope_eval.OUT_OF_SCOPE)

    n_model_checked = len(scope_eval.OUT_OF_SCOPE) - 2   # 2 injections are blocked by regex first
    replies["scope"] = (['{"in_scope": false}'] * n_model_checked) + ['{"in_scope": true}'] * 4
    scope_eval.main()
    rows = read(out / "scope_eval.csv")
    summary = {r["question"].split(" ")[0]: r["refused"] for r in rows if r["type"] == "SUMMARY"}
    assert summary["refusal_rate"] == "100.0%"     # the model + 2 injections by regex
    assert summary["false_refusal_rate"] == "0.0%"
    assert summary["catalog"] == "PASS"
    assert len([r for r in rows if r["type"] == "out_of_scope"]) == len(out_questions) >= 5


class RateLimitError(Exception):
    status_code = 429


def test_scope_eval_does_not_count_errors_as_missed_refusals(eval_env, monkeypatch):
    """Regression: a rate-limited model made every question 'not refused' (11.1% FAIL)."""
    replies, out = eval_env
    monkeypatch.setattr(common, "RATE_LIMIT_RETRIES", 1)
    replies["scope"] = RateLimitError("busy")          # every model call fails
    scope_eval.main()
    rows = read(out / "scope_eval.csv")
    summary = {r["question"].split(" ")[0]: r["refused"] for r in rows if r["type"] == "SUMMARY"}
    assert summary["catalog"].startswith("INCOMPLETE")
    assert summary["errors"] == str(len(scope_eval.OUT_OF_SCOPE) - 2 + len(scope_eval.IN_SCOPE_CONTROLS))
    assert summary["refusal_rate"] == "100.0%"         # only the 2 regex-blocked questions counted
    errored = [r for r in rows if r["error"]]
    assert errored and all(r["refused"] == "" for r in errored)


def test_run_agent_retries_after_rate_limit(eval_env, monkeypatch):
    replies, _ = eval_env
    sleeps = []
    monkeypatch.setattr(common.time, "sleep", sleeps.append)
    replies["scope"] = [RateLimitError("busy"), '{"in_scope": true}']
    result = common.run_agent("What is a list?", "free/model:free")
    assert result["error"] == "" and result["answer"]
    assert sleeps == [common.RATE_LIMIT_WAIT_SECONDS]


def test_scope_eval_model_can_be_overridden(eval_env, monkeypatch):
    _, out = eval_env
    monkeypatch.setenv("EVAL_SCOPE_MODEL", "paid/model")
    scope_eval.main()
    assert {r["model"] for r in read(out / "scope_eval.csv")} == {"paid/model"}
