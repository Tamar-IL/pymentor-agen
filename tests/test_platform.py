"""The Safe-Agents platform contract: `graph` entry point, `messages` state, HITL format."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent import graph as graph_module


def platform_graph():
    # The platform's LangGraph server attaches its own checkpointer to the same builder.
    return graph_module._builder().compile(checkpointer=InMemorySaver())


def config(thread="t1", model="platform/model:free"):
    return {"configurable": {"thread_id": thread, "model": model}}


def test_module_level_graph_has_no_custom_checkpointer():
    assert graph_module.graph.checkpointer is None


def test_messages_in_messages_out_with_usage_for_cost(fake_llm):
    graph = platform_graph()
    result = graph.invoke({"messages": [HumanMessage("What is a tuple?")]}, config())
    last = result["messages"][-1]
    assert isinstance(last, AIMessage) and "tuple" in last.content
    assert last.usage_metadata["input_tokens"] > 0 and last.usage_metadata["output_tokens"] > 0
    assert last.response_metadata["model_name"] == "platform/model:free"   # model from config


def test_refusal_is_returned_as_a_message(fake_llm):
    replies, _ = fake_llm
    replies["scope"] = '{"in_scope": false}'
    result = platform_graph().invoke({"messages": [HumanMessage("תכתוב לי שיר על הים")]}, config())
    assert "PyMentor" in result["messages"][-1].content and result["blocked"]


def test_second_turn_on_same_thread_starts_clean(fake_llm):
    replies, _ = fake_llm
    replies["classify"] = ['{"intent": "debug"}', '{"intent": "explain"}']
    graph, cfg = platform_graph(), config("same-thread")
    graph.invoke({"messages": [HumanMessage("fix\n```python\nprint(1/0)\n```")]}, cfg)
    graph.invoke(Command(resume={"approved": True}), cfg)
    assert "ZeroDivisionError" in graph.get_state(cfg).values["code_output"]

    second = graph.invoke({"messages": [HumanMessage("What is a list?")]}, cfg)
    assert second["code_output"] is None and second["attempts"] == 1
    assert len([m for m in second["messages"] if isinstance(m, AIMessage)]) == 2
    # Usage on the second answer counts only the second turn's calls.
    first_turn_calls = second["usage_start"]
    assert first_turn_calls > 0 and len(second["usage"]) > first_turn_calls


def test_hitl_uses_contract_payload_and_dict_answers(fake_llm):
    replies, _ = fake_llm
    replies["classify"] = '{"intent": "run"}'
    graph, cfg = platform_graph(), config("hitl")
    graph.invoke({"messages": [HumanMessage("run\n```python\nprint('a')\n```")]}, cfg)
    payload = graph.get_state(cfg).interrupts[0].value
    assert payload["type"] == "approval" and payload["action"] == "run_python_code"
    assert "print('a')" in payload["details"]["code"]

    graph.invoke(Command(resume={"approved": True, "edited": "print('edited')"}), cfg)
    state = graph.get_state(cfg).values
    assert "edited" in state["code_output"] and state["code"] == "print('edited')"


def test_rejected_hitl_does_not_run_code(fake_llm):
    replies, _ = fake_llm
    replies["classify"] = '{"intent": "run"}'
    graph, cfg = platform_graph(), config("reject")
    graph.invoke({"messages": [HumanMessage("run\n```python\nprint('a')\n```")]}, cfg)
    graph.invoke(Command(resume={"approved": False}), cfg)
    assert graph.get_state(cfg).values["code_approved"] is False


def test_file_path_inside_message_is_read(fake_llm, tmp_path):
    replies, _ = fake_llm
    replies["classify"] = '{"intent": "debug"}'
    script = tmp_path / "ex1.py"
    script.write_text("nums = [1]\nprint(nums[5])\n", encoding="utf-8")
    graph, cfg = platform_graph(), config("file")
    graph.invoke({"messages": [HumanMessage(f"why does {script} crash?")]}, cfg)
    graph.invoke(Command(resume={"approved": True}), cfg)
    state = graph.get_state(cfg).values
    assert state["file_name"] == "ex1.py" and "IndexError" in state["code_output"]
