"""Eval 2 - Cost.

Runs common tasks on free AND paid models and reports the cost of every step
(graph node) and the total:  cost = model price x tokens.
For free models it shows the actual cost (0) next to the THEORETICAL cost - what the
same tokens would cost at the paid equivalent's price.

Usage:  python evals/cost_eval.py
Output: evals/results/cost_eval.csv
"""

from __future__ import annotations

from common import ALL_MODELS, PAID_MODELS, require_api_key, run_agent, write_csv

from agent.pricing import cost_usd, prices_for

TASKS = [
    ("explain_concept", "What is a list comprehension? Show a simple example."),
    ("debug_error", "I get NameError: name 'totl' is not defined.\n```python\ntotal = 5\nprint(totl)\n```"),
    ("review_code", "Can you review my function?\n```python\ndef add_item(item, items=[]):\n    items.append(item)\n    return items\n```"),
    ("out_of_scope", "What is the capital of France?"),
]


def main() -> None:
    require_api_key()
    if not PAID_MODELS:
        raise SystemExit("The cost eval needs at least one paid model (EVAL_PAID_MODELS).")
    rows, grand = [], {}
    for model in ALL_MODELS:
        try:
            prices = prices_for(model)
        except KeyError as problem:
            print(f"Skipping {model}: {problem}")
            continue
        for task_id, question in TASKS:
            print(f"[{model}] {task_id} ...", flush=True)
            result = run_agent(question, model)
            if result["error"]:
                print(f"   ERROR: {result['error']}")
                rows.append({"model": model, "task_id": task_id, "node": "ERROR", "error": result["error"]})
                continue
            total = {"in": 0, "out": 0, "actual": 0.0, "theoretical": 0.0}
            for step, usage in enumerate(result["usage"], start=1):
                actual = cost_usd(prices["actual"], usage["input_tokens"], usage["output_tokens"])
                theoretical = cost_usd(prices["theoretical"], usage["input_tokens"], usage["output_tokens"])
                rows.append({
                    "model": model, "task_id": task_id, "step": step, "node": usage["node"],
                    "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
                    "price_in_per_1M": round(prices["theoretical"]["input"] * 1e6, 4),
                    "price_out_per_1M": round(prices["theoretical"]["output"] * 1e6, 4),
                    "actual_cost_usd": f"{actual:.8f}", "theoretical_cost_usd": f"{theoretical:.8f}",
                    "price_source": prices["source"],
                })
                total["in"] += usage["input_tokens"]
                total["out"] += usage["output_tokens"]
                total["actual"] += actual
                total["theoretical"] += theoretical
            rows.append({"model": model, "task_id": task_id, "step": "", "node": "TASK TOTAL",
                         "input_tokens": total["in"], "output_tokens": total["out"],
                         "actual_cost_usd": f"{total['actual']:.8f}",
                         "theoretical_cost_usd": f"{total['theoretical']:.8f}",
                         "seconds": result["seconds"], "price_source": prices["source"]})
            model_total = grand.setdefault(model, [0.0, 0.0])
            model_total[0] += total["actual"]
            model_total[1] += total["theoretical"]

    print("\n=== Total cost per model (all tasks) ===")
    for model, (actual, theoretical) in grand.items():
        rows.append({"model": model, "task_id": "ALL TASKS", "node": "GRAND TOTAL",
                     "actual_cost_usd": f"{actual:.8f}", "theoretical_cost_usd": f"{theoretical:.8f}"})
        print(f"{model:50} actual ${actual:.6f}   theoretical ${theoretical:.6f}")
    write_csv("cost_eval.csv", rows)


if __name__ == "__main__":
    main()
