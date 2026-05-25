# Seed generation prompt (v1)

**Target model:** Claude Opus 4.7
**Source PDF:** `data/Claude Certified Architect – Foundations Certification Exam Guide.pdf`
**Output file:** `eval/eval_testset_v1.jsonl`

---

## Prompt

You are generating a retrieval-evaluation test set for a RAG system over the attached PDF: "Claude Certified Architect – Foundations Certification Exam Guide."

Produce **exactly 60** question-answer pairs as **JSONL** (one JSON object per line, no surrounding array, no markdown fencing).

Distribute them as **12 per type** across these 5 types:

1. **definition** — asks what a specific term or concept means, as defined in the PDF.
2. **enumeration** — asks for a list the PDF explicitly enumerates (steps, components, requirements, sections, etc.).
3. **indirect** — answer is in the PDF, but the question paraphrases the source wording or requires combining 1–2 sentences.
4. **negation** — asks what is *not* something (e.g., "Which of these is not required for…"), with the negated fact present in the PDF.
5. **no_answer** — question sounds plausibly about the same domain, but the answer is not in the PDF. The `answer` field must be the literal string `"NOT_IN_DOCUMENT"`.

### Schema (per record)

| field      | type    | description |
|------------|---------|-------------|
| `id`       | string  | `"Q01"` … `"Q60"` |
| `type`     | string  | one of the 5 type names above |
| `question` | string  | self-contained question — do NOT reference "the document" or "this guide" |
| `answer`   | string  | grounded answer, ≤ 25 words (or `"NOT_IN_DOCUMENT"` for `no_answer`) |
| `source_quote` | string  | verbatim quote from the PDF supporting the answer (empty string for `no_answer`) |
| `section`  | string  | section heading where the answer appears (empty for `no_answer`) |
| `page`     | integer | page number in the PDF (`null` for `no_answer`) |

### Rules

- Do **not** invent facts. If you can't find verbatim evidence, pick a different question.
- For `no_answer`, the question must sound like it could be answered by this PDF but actually requires outside knowledge — e.g., pricing, comparison to other certifications, exam dates not stated in the guide, vendor-specific implementation details not covered.
- Questions must be **self-contained** (don't reference "the document," "this guide," "the attached PDF," etc.).
- Spread questions across the PDF — **no more than 3 questions per section**.
- For `enumeration`, the answer should list the items (comma-separated or short bullets in one string).
- For `negation`, phrase as multiple choice OR as "Which of the following is not …" with the wrong-fit option as the answer.

### Output

Output **only** the JSONL. No preamble, no trailing summary, no code fences.
