"""Test helpers: a fake chat model so tests run offline, free and deterministic."""

import pytest
from langchain_core.messages import AIMessage

from agent import llm


class FakeChatModel:
    """Answers by node, recognised from the system prompt. Records every call."""

    def __init__(self, replies: dict, calls: list):
        self.replies, self.calls = replies, calls

    def invoke(self, messages):
        system, user = messages[0].content, messages[1].content
        if "scope classifier" in system:
            key = "scope"
        elif "Classify the user's request" in system:
            key = "classify"
        elif "strict reviewer" in system:
            key = "evaluator"
        elif "Summarize this Python tutoring session" in system:
            key = "summary"
        else:
            key = "draft"
        self.calls.append(key)
        reply = self.replies[key]
        if isinstance(reply, list):          # a list = different reply per call
            reply = reply.pop(0) if len(reply) > 1 else reply[0]
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(content=reply, usage_metadata={
            "input_tokens": len(system + user) // 4, "output_tokens": len(reply) // 4,
            "total_tokens": (len(system + user) + len(reply)) // 4})


DEFAULT_REPLIES = {
    "scope": '{"in_scope": true, "reason": "python"}',
    "classify": '{"intent": "explain"}',
    "draft": "A list can change; a tuple cannot.\n\n- why\n\nTip: use tuples for fixed data.",
    "evaluator": '{"score": 5, "feedback": "good"}',
    "summary": "- talked about lists",
}


@pytest.fixture
def fake_llm():
    """Yields (replies, calls). Edit `replies` in a test to change model behaviour."""
    replies, calls = dict(DEFAULT_REPLIES), []
    llm.set_model_factory(lambda model, temperature: FakeChatModel(replies, calls))
    yield replies, calls
    llm.set_model_factory(None)
