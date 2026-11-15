"""OpenRouter connection: one place that creates chat models, calls them, and
turns provider failures into clear messages.

OpenRouter speaks the OpenAI API, so we use langchain-openai's ChatOpenAI with
OpenRouter's base URL. Every call made inside the LangGraph run is traced in
LangSmith automatically when LANGSMITH_TRACING=true.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Free models change often on OpenRouter. These defaults can be overridden in .env,
# and the CLI command `models` lists the free models available right now.
# (Checked against the live OpenRouter catalog, November 2026.)
DEFAULT_MODEL = "google/gemma-4-31b-it:free"
DEFAULT_FALLBACK_MODELS = [
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/free",   # OpenRouter's router over whatever free models exist today
]


class LLMError(RuntimeError):
    """A provider failure, already worded for the user."""


def default_model() -> str:
    return os.getenv("PYMENTOR_MODEL", DEFAULT_MODEL)


def fallback_models() -> list[str]:
    raw = os.getenv("PYMENTOR_FALLBACK_MODELS")
    if raw is None:
        return list(DEFAULT_FALLBACK_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]


# Tests replace the factory with a fake model so they run offline and free.
_model_factory: Callable[[str, float], Any] | None = None


def set_model_factory(factory: Callable[[str, float], Any] | None) -> None:
    global _model_factory
    _model_factory = factory


def make_chat_model(model: str, temperature: float = 0.2):
    if _model_factory is not None:
        return _model_factory(model, temperature)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY is missing. Put it in a .env file (see .env.example).")
    from langchain_openai import ChatOpenAI  # imported lazily: faster CLI start-up

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        timeout=90,
        max_retries=2,
        stream_usage=True,   # token counts also when streamed (Safe-Agents cost display)
        default_headers={"X-Title": "PyMentor Agent"},
    )


def _friendly_error(model: str, error: Exception) -> str:
    name = type(error).__name__
    text = str(error)
    status = getattr(error, "status_code", None)
    if name == "AuthenticationError" or status == 401:
        return "OpenRouter rejected the API key. Check OPENROUTER_API_KEY in your .env file."
    if status == 402:
        return f"Model '{model}' needs credits. Add credits on openrouter.ai or use a ':free' model."
    if name == "NotFoundError" or status == 404:
        return (f"Model '{model}' was not found or is no longer free. "
                "Type 'models' to see available free models, then 'model <id>'.")
    if name == "RateLimitError" or status == 429:
        return (f"Model '{model}' is rate-limited right now (free models have limits). "
                "Wait a minute or switch model with 'model <id>'.")
    if name in ("APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout"):
        return "Cannot reach OpenRouter. Check your internet connection and try again."
    return f"The model call failed ({name}): {text[:300]}"


def is_free_model(model: str) -> bool:
    return model.endswith(":free") or model == "openrouter/free"


def _is_model_unavailable(error: Exception) -> bool:
    """A provider-side problem with this model (gone, rate-limited, rejected the request).

    Key and credit problems (401/402) are not model problems: another model won't help.
    """
    status = getattr(error, "status_code", None)
    if status in (401, 402):
        return False
    return status is not None or type(error).__name__ in ("NotFoundError", "RateLimitError")


def call_llm(node: str, model: str, system: str, user: str,
             temperature: float = 0.2) -> tuple[str, dict]:
    """Call the model and return (text, usage record).

    If a ':free' model is unavailable or rate-limited, the fallback free models are
    tried in order. The usage record says which model really answered.
    """
    candidates = [model]
    if is_free_model(model):
        candidates += [m for m in fallback_models() if m != model]

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            response = make_chat_model(candidate, temperature).invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
        except LLMError:
            raise
        except Exception as error:  # provider errors come from several libraries
            last_error = error
            if _is_model_unavailable(error):
                continue
            raise LLMError(_friendly_error(candidate, error)) from error
        usage = getattr(response, "usage_metadata", None) or {}
        record = {
            "node": node,
            "model": candidate,
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
        }
        content = response.content if isinstance(response.content, str) else str(response.content)
        return content.strip(), record

    assert last_error is not None
    raise LLMError(_friendly_error(model, last_error)) from last_error


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> dict | None:
    """Extract the first JSON object from model text.

    We ask for JSON in the prompt instead of using tool-calling / structured output,
    because many free OpenRouter models do not support tool-calling.
    """
    match = _JSON_OBJECT_RE.search(text or "")
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
