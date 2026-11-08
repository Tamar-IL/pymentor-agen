"""Eval 1 - Model comparison.

Runs the SAME tasks through the agent on different models and compares time and
quality (LLM-as-judge, fixed judge model, rubric in evals/rubric.md).

Usage:  python evals/compare_models.py
Output: evals/results/compare_models.csv
"""

from __future__ import annotations

from statistics import mean

from common import ALL_MODELS, JUDGE_MODEL, judge, require_api_key, run_agent, write_csv

TASKS = [
    {
        "id": "explain_list_tuple",
        "question": "What is the difference between a list and a tuple in Python? When should I use each?",
        "reference": "Lists are mutable, tuples are immutable; both are ordered and allow duplicates; "
                     "tuples are used for fixed data and can be dict keys; lists for collections that change.",
    },
    {
        "id": "debug_concat",
        "question": "Why does this crash and how do I fix it?\n```python\nage = 15\nprint(\"I am \" + age + \" years old\")\n```",
        "reference": "TypeError: can only concatenate str (not \"int\") to str. Fix with str(age) or an "
                     "f-string: print(f\"I am {age} years old\").",
    },
    {
        "id": "debug_index",
        "question": "My loop gives IndexError: list index out of range. What is wrong?\n```python\nnums = [1, 2, 3]\nfor i in range(len(nums) + 1):\n    print(nums[i])\n```",
        "reference": "range(len(nums) + 1) goes up to index 3 but the last valid index is 2. Use "
                     "range(len(nums)) or better: for n in nums: print(n).",
    },
]


def main() -> None:
    require_api_key()
    rows = []
    for model in ALL_MODELS:
        for task in TASKS:
            print(f"[{model}] {task['id']} ...", flush=True)
            result = run_agent(task["question"], model)
            row = {"model": model, "task_id": task["id"], "seconds": result["seconds"],
                   "input_tokens": sum(u["input_tokens"] for u in result["usage"]),
                   "output_tokens": sum(u["output_tokens"] for u in result["usage"]),
                   "agent_self_eval": result["state"].get("eval_score", ""),
                   "judge_model": JUDGE_MODEL, "error": result["error"]}
            if result["error"]:
                print(f"   ERROR: {result['error']}")
            else:
                try:
                    row.update(judge(task["question"], result["answer"], task["reference"]))
                except Exception as problem:   # a judge failure must not lose the run
                    row["error"] = f"judge failed: {problem}"
            row["answer"] = result["answer"].replace("\r", "")
            rows.append(row)

    # Summary row per model (averages over tasks that succeeded).
    print("\n=== Summary (averages) ===")
    print(f"{'model':50} {'seconds':>8} {'overall':>8} {'accuracy':>9}")
    for model in ALL_MODELS:
        ok = [r for r in rows if r["model"] == model and isinstance(r.get("overall"), int)]
        if not ok:
            print(f"{model:50} {'(no successful runs)':>27}")
            continue
        summary = {"model": model, "task_id": "AVERAGE",
                   "seconds": round(mean(r["seconds"] for r in ok), 2)}
        for key in ("role_relevance", "factual_accuracy", "completeness", "format_compliance", "overall"):
            summary[key] = round(mean(r[key] for r in ok if isinstance(r[key], int)), 2)
        rows.append(summary)
        print(f"{model:50} {summary['seconds']:>8} {summary['overall']:>8} {summary['factual_accuracy']:>9}")
    write_csv("compare_models.csv", rows)


if __name__ == "__main__":
    main()
