"""Unit tests for the extensions (RAG, code-exec, file input), JSON parsing and pricing."""

import pytest

from agent import pricing, tools
from agent.llm import parse_json_object


def test_rag_finds_the_matching_error_note():
    results = tools.search_knowledge("TypeError: can only concatenate str (not \"int\") to str")
    assert results and results[0]["title"].startswith("TypeError")


def test_rag_finds_index_error():
    assert "IndexError" in tools.search_knowledge("IndexError: list index out of range")[0]["title"]


def test_rag_returns_nothing_for_unrelated_text():
    assert tools.search_knowledge("pancakes blueberry syrup") == []


def test_run_python_code_returns_output_and_exit_code():
    result = tools.run_python_code("print(2 + 3)")
    assert "[exit code 0]" in result and "5" in result


def test_run_python_code_reports_errors():
    result = tools.run_python_code("x = [1]\nprint(x[5])")
    assert "IndexError" in result and "[exit code 1]" in result


def test_run_python_code_stops_infinite_loops():
    assert "[timeout]" in tools.run_python_code("while True: pass", timeout=1)


def test_run_python_code_truncates_huge_output():
    result = tools.run_python_code("print('x' * 100000)")
    assert "[output truncated]" in result and len(result) < tools.MAX_OUTPUT_CHARS + 200


def test_risky_operations_are_flagged():
    warnings = tools.risky_operations("import shutil\nshutil.rmtree('/')\nimport subprocess")
    assert any("shutil" in w for w in warnings) and any("subprocess" in w for w in warnings)
    assert tools.risky_operations("print('hello')") == []


def test_read_local_file_strips_drag_and_drop_quotes(tmp_path):
    file = tmp_path / "ex 1.py"
    file.write_text("print('hi')", encoding="utf-8")
    path, content = tools.read_local_file(f'"{file}"')
    assert path == file and "print" in content


@pytest.mark.parametrize("name,message", [("missing.py", "not found"), ("data.exe", "Unsupported")])
def test_read_local_file_rejects_bad_input(tmp_path, name, message):
    if name.endswith(".exe"):
        (tmp_path / name).write_bytes(b"MZ")
    with pytest.raises(tools.FileInputError, match=message):
        tools.read_local_file(str(tmp_path / name))


def test_read_local_file_rejects_folders_and_big_files(tmp_path):
    with pytest.raises(tools.FileInputError, match="folder"):
        tools.read_local_file(str(tmp_path))
    big = tmp_path / "big.txt"
    big.write_text("x" * (tools.MAX_FILE_BYTES + 1))
    with pytest.raises(tools.FileInputError, match="too large"):
        tools.read_local_file(str(big))


@pytest.mark.parametrize("text,expected", [
    ('{"score": 4}', {"score": 4}),
    ('```json\n{"intent": "debug"}\n```', {"intent": "debug"}),
    ('Sure! {"in_scope": false, "reason": "x"} done', {"in_scope": False, "reason": "x"}),
    ("no json here", None),
    ("{broken", None),
])
def test_parse_json_object(text, expected):
    assert parse_json_object(text) == expected


FAKE_CATALOG = {
    "meta-llama/llama-3.3-70b-instruct:free": {"input": 0.0, "output": 0.0, "name": "free"},
    "meta-llama/llama-3.3-70b-instruct": {"input": 1e-7, "output": 3e-7, "name": "paid"},
    "orphan/model:free": {"input": 0.0, "output": 0.0, "name": "orphan"},
    "openai/gpt-4o-mini": {"input": 1.5e-7, "output": 6e-7, "name": "mini"},
}


@pytest.fixture
def fake_catalog(monkeypatch):
    monkeypatch.setattr(pricing, "fetch_model_catalog", lambda: FAKE_CATALOG)


def test_free_model_has_zero_actual_and_paid_twin_theoretical_cost(fake_catalog):
    prices = pricing.prices_for("meta-llama/llama-3.3-70b-instruct:free")
    assert pricing.cost_usd(prices["actual"], 1000, 1000) == 0
    assert pricing.cost_usd(prices["theoretical"], 1_000_000, 1_000_000) == pytest.approx(0.4)
    assert "paid twin" in prices["source"]


def test_free_model_without_twin_uses_reference_rate(fake_catalog):
    prices = pricing.prices_for("orphan/model:free")
    assert prices["source"].startswith("reference")
    assert pricing.cost_usd(prices["theoretical"], 1_000_000, 0) == pytest.approx(0.10)


def test_paid_model_actual_equals_theoretical(fake_catalog):
    prices = pricing.prices_for("openai/gpt-4o-mini")
    assert prices["actual"] == prices["theoretical"]
    with pytest.raises(KeyError):
        pricing.prices_for("unknown/model")


def test_trace_metadata_reads_agent_id_and_environment(monkeypatch):
    from agent.runner import trace_metadata

    monkeypatch.setenv("LANGSMITH_AGENT_ID", "bachir")
    monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "local")
    assert trace_metadata("m/x:free") == {"model": "m/x:free", "agent_id": "bachir", "agent_environment": "local"}


def test_traced_tools_still_work_with_tracing_off(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert tools.search_knowledge("ZeroDivisionError")[0]["title"] == "ZeroDivisionError"
    assert "4" in tools.run_python_code("print(2 * 2)")


def test_langsmith_project_is_dropped_when_agent_address_is_set(monkeypatch):
    """Regression: with both set, LangSmith silently recorded zero traces."""
    from agent.tracing import describe_target, resolve_langsmith_target

    monkeypatch.setenv("LANGSMITH_PROJECT", "pymentor-agent")
    monkeypatch.setenv("LANGSMITH_AGENT_ID", "bachir")
    monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "local")
    assert "ignored" in resolve_langsmith_target()
    assert "LANGSMITH_PROJECT" not in __import__("os").environ
    assert describe_target() == "agent: bachir / local"


def test_langsmith_project_alone_is_kept(monkeypatch):
    from agent.tracing import resolve_langsmith_target

    monkeypatch.delenv("LANGSMITH_AGENT_ID", raising=False)
    monkeypatch.setenv("LANGSMITH_PROJECT", "pymentor-agent")
    assert resolve_langsmith_target() is None
    assert __import__("os").environ["LANGSMITH_PROJECT"] == "pymentor-agent"
