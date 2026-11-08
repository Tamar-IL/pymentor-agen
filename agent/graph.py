"""The LangGraph workflow for PyMentor.

    START
      |
  input_guardrail ----(blocked)----> refuse ---------------------------> END
      | (ok)
   classify
      |
   retrieve (RAG)
      |---(run/debug + code)---> approve_and_run (HITL interrupt + code-exec)
      |                                   |
    draft <-------------------------------+
      |
   evaluator ----(score < 3, first try)----> draft   (quality gate, max 1 rewrite)
      |
  output_guardrail --> END
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.nodes import AgentState


def build_graph(checkpointer=None):
    """Compile the graph. A checkpointer is required for HITL (interrupt/resume)."""
    builder = StateGraph(AgentState)

    builder.add_node("input_guardrail", nodes.input_guardrail)
    builder.add_node("refuse", nodes.refuse)
    builder.add_node("classify", nodes.classify)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("approve_and_run", nodes.approve_and_run)
    builder.add_node("draft", nodes.draft)
    builder.add_node("evaluator", nodes.evaluator)
    builder.add_node("output_guardrail", nodes.output_guardrail)

    builder.add_edge(START, "input_guardrail")
    builder.add_conditional_edges(
        "input_guardrail",
        lambda state: "refuse" if state.get("blocked") else "classify",
        ["refuse", "classify"],
    )
    builder.add_edge("refuse", END)
    builder.add_edge("classify", "retrieve")
    builder.add_conditional_edges("retrieve", nodes.route_after_retrieve, ["approve_and_run", "draft"])
    builder.add_edge("approve_and_run", "draft")
    builder.add_edge("draft", "evaluator")
    builder.add_conditional_edges("evaluator", nodes.route_after_evaluator, ["draft", "output_guardrail"])
    builder.add_edge("output_guardrail", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
