"""Run one user turn through the graph: streaming progress + HITL resume loop.

Shared by the CLI (interactive approval) and the evals (scripted approval), so both
exercise exactly the same workflow.
"""

from __future__ import annotations

import uuid
from typing import Callable

from langgraph.types import Command

NODE_LABELS = {
    "input_guardrail": "Input_Guardrail (safety + scope check)",
    "refuse": "Refuse (out of scope)",
    "classify": "Classify (what kind of request)",
    "retrieve": "Retrieve (RAG knowledge search)",
    "approve_and_run": "Approve_And_Run (HITL + code execution)",
    "draft": "Draft (writing the answer)",
    "evaluator": "Evaluator (quality check)",
    "output_guardrail": "Output_Guardrail (final safety check)",
}


def run_turn(
    graph,
    inputs: dict,
    approve: Callable[[dict], str],
    on_node_start: Callable[[str], None] | None = None,
    on_node_end: Callable[[str, dict], None] | None = None,
) -> dict:
    """Run the graph until it finishes. `approve(payload)` answers each HITL interrupt
    with "y" or "n". Returns the final state values."""
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},   # each question is independent
        "run_name": "PyMentor turn",                         # name shown in LangSmith
        "tags": ["pymentor"],
        "metadata": {"model": inputs.get("model")},
    }
    next_input: dict | Command = {**inputs, "usage": []}
    while True:
        for event in graph.stream(next_input, config, stream_mode="tasks"):
            if "triggers" in event:                     # a node is starting
                if on_node_start:
                    on_node_start(event["name"])
            elif on_node_end and not event.get("interrupts"):
                on_node_end(event["name"], event.get("result") or {})
        snapshot = graph.get_state(config)
        if not snapshot.interrupts:
            return snapshot.values
        next_input = Command(resume=approve(snapshot.interrupts[0].value))
