"""LangSmith settings check.

LangSmith can route traces either to a project (LANGSMITH_PROJECT) or to an agent
address (LANGSMITH_AGENT_ID + LANGSMITH_AGENT_ENVIRONMENT). If both are set, LangSmith
refuses to record ANY trace ("neither outranks the other"), silently. This keeps the
agent address, which is the more specific choice, and says so.
"""

from __future__ import annotations

import os


def resolve_langsmith_target() -> str | None:
    """Fix a project/agent-address conflict in the environment. Returns a notice, or None."""
    if os.getenv("LANGSMITH_AGENT_ID") and os.getenv("LANGSMITH_PROJECT"):
        project = os.environ.pop("LANGSMITH_PROJECT")
        os.environ.pop("LANGCHAIN_PROJECT", None)
        return (f"LangSmith: LANGSMITH_AGENT_ID is set, so LANGSMITH_PROJECT='{project}' is ignored "
                "(LangSmith records nothing when both are set).")
    return None


def describe_target() -> str:
    """Where traces go, for the CLI start-up line."""
    agent_id = os.getenv("LANGSMITH_AGENT_ID")
    if agent_id:
        return f"agent: {agent_id} / {os.getenv('LANGSMITH_AGENT_ENVIRONMENT', 'local')}"
    return f"project: {os.getenv('LANGSMITH_PROJECT', 'default')}"
