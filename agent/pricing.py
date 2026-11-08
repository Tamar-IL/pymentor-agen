"""Model prices from OpenRouter's public /models endpoint, and cost = price x tokens.

For ':free' models the real cost is 0, so we also compute a *theoretical* cost:
what the same tokens would cost on the paid version of that model (same id without
':free'). If no paid twin is listed, a fixed reference rate is used and labelled.
"""

from __future__ import annotations

import json
import urllib.request
from functools import lru_cache

MODELS_URL = "https://openrouter.ai/api/v1/models"

# Fallback reference rate (USD per 1M tokens) when a free model has no paid twin.
REFERENCE_PRICE_PER_M = {"input": 0.10, "output": 0.30}


@lru_cache(maxsize=1)
def fetch_model_catalog() -> dict[str, dict]:
    """{model_id: {"input": usd_per_token, "output": usd_per_token, "name": str}}"""
    request = urllib.request.Request(MODELS_URL, headers={"User-Agent": "pymentor-agent"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)["data"]
    catalog = {}
    for model in data:
        pricing = model.get("pricing") or {}
        try:
            catalog[model["id"]] = {
                "input": float(pricing.get("prompt") or 0),
                "output": float(pricing.get("completion") or 0),
                "name": model.get("name", model["id"]),
            }
        except (TypeError, ValueError):
            continue
    return catalog


def list_free_models() -> list[str]:
    return sorted(m for m in fetch_model_catalog() if m.endswith(":free")) + ["openrouter/free"]


def prices_for(model: str) -> dict:
    """Actual and theoretical USD-per-token prices for a model, plus where they came from."""
    catalog = fetch_model_catalog()
    actual = catalog.get(model)
    if actual is None:
        raise KeyError(f"Model '{model}' is not in the OpenRouter catalog")
    if actual["input"] < 0 or actual["output"] < 0:
        raise KeyError(f"'{model}' is a router with variable pricing; price the model it routed to")
    is_free = model.endswith(":free") or (actual["input"] == 0 and actual["output"] == 0)
    if not is_free:
        return {"actual": actual, "theoretical": actual, "source": "openrouter list price"}
    zero = {"input": 0.0, "output": 0.0}
    twin = catalog.get(model.removesuffix(":free"))
    if twin and (twin["input"] or twin["output"]):
        return {"actual": zero, "theoretical": twin, "source": f"paid twin {model.removesuffix(':free')}"}
    reference = {k: v / 1_000_000 for k, v in REFERENCE_PRICE_PER_M.items()}
    return {"actual": zero, "theoretical": reference, "source": "reference rate (no paid twin listed)"}


def cost_usd(price: dict, input_tokens: int, output_tokens: int) -> float:
    return input_tokens * price["input"] + output_tokens * price["output"]
