# LLM-as-Judge Rubric — PyMentor (Python tutor agent)

The judge is **one fixed model** (`JUDGE_MODEL`, default `openai/gpt-4o-mini`) for every
comparison, so that all models are graded by the same standard (no self-preference bias).
Each criterion gets an integer score from **1 to 5**.

## 1. Role relevance (`role_relevance`)
Does the answer stay in PyMentor's role — a Python tutor for beginners?
- **5** — fully focused on the student's Python question, beginner-friendly tone.
- **3** — on topic but includes unnecessary content or a tone that is too advanced.
- **1** — off topic, or answers something that was not asked.

## 2. Factual accuracy (`factual_accuracy`) — *data reliability*
Are the Python facts, explanations and code correct? Use the reference notes.
- **5** — everything is correct; code runs and does what is claimed.
- **3** — mostly correct with a minor inaccuracy that would not mislead the student.
- **1** — a wrong explanation, invented behaviour or code that does not work.

## 3. Completeness (`completeness`) — *data quality*
Does the answer cover what the student needs?
- **5** — explains the cause, gives the fix or example, and says why it works.
- **3** — gives the fix or the explanation, but not both.
- **1** — missing the main point.

## 4. Format compliance (`format_compliance`)
The required format is:
1. a short direct answer (1–3 sentences);
2. a ```python code block when code is involved;
3. a short "why" explanation (bullets);
4. a final line starting with **`Tip:`**.
- **5** — all parts present. **3** — one part missing. **1** — no structure.

## 5. Overall (`overall`) — *answer quality*
The judge's overall grade: would a good human Python teacher be happy to give this
answer to a beginner? **5** = yes, as is. **3** = with edits. **1** = no.

## How the scores map to the assignment
| Assignment metric | Column(s) in `compare_models.csv` |
|---|---|
| Time | `seconds` (measured automatically) |
| Answer quality | `overall` |
| Data quality | `completeness`, `format_compliance` |
| Data reliability | `factual_accuracy` |
| Role adherence | `role_relevance` |
