# LocalRAG — Interview Prep

Questions calibrated to the actual stack:

- **Ingestion:** PyMuPDF on the Claude Certified Architect – Foundations Certification Exam Guide PDF
- **Chunking:** `RecursiveCharacterTextSplitter` (not semantic)
- **Indexing:** ChromaDB (vector) + BM25 (lexical)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Retrieval:** Hybrid (semantic + BM25) fused with **RRF**
- **Reranker:** `BAAI/bge-reranker-base`, `top_n=5`
- **LLM:** `qwen2.5:7b` on Ollama (switched from `llama3.2:3b`)
- **Eval:** 60 Q&A pairs, 5 types × 12 each (definition, enumeration, indirect, negation, no_answer)
- **Baseline (2026-05-25):** def 0.78/0.92, enum 0.82/0.92, indirect 0.96/1.0, negation 0.92/0.92 (MRR / Recall@5)

Where you see `[fill in]` — pull the number from your eval before interviewing.

---

## A. RAG System Design

### A1. Frame the problem.
User = someone studying for the Claude Certified Architect exam who needs grounded, cited answers without sending queries to a cloud LLM. Success = correct + traceable to the PDF + conversational latency. *Bad* answer = a fluent, plausible-sounding completion that's not actually in the PDF — for study material, hallucination is worse than refusal.

### A2. Walk the four layers and their assumptions.
- **Ingestion:** assumes text-extractable PDF (no scanned-image-only pages). Out: chunks + section/page metadata.
- **Indexing:** assumes chunks are self-contained enough to score in isolation. Out: Chroma vector index + BM25 index.
- **Retrieval:** assumes the query is in-domain. Out: top-k candidates with fused score.
- **Generation:** assumes context is relevant; prompt enforces grounding. Out: answer (+ ideally citation).

### A3. Chunking — defend the choice.
Using recursive character splitting, not semantic chunking. The exam guide is structured (sections, headers, bullets), so paragraph/sentence boundaries already align with semantic units — semantic chunking pays compute for marginal gain on this corpus. Switch when the source is unstructured prose (transcripts, contracts), or when `definition` failures trace to related sentences split across chunks.

### A4. RRF vs. weighted fusion.
RRF needs no score normalization — robust when BM25 and cosine sim live on totally different scales. Weighted sum needs an α tuned per corpus. RRF *loses* when one retriever is strictly better (e.g., BM25 near-zero on paraphrased queries) — RRF still gives it equal rank weight. At that point I'd move to per-query-type fusion weights, ideally driven by a router.

### A5. Reranker delta — what does it actually buy?
Honest: not ablated yet — current eval is hybrid+reranker only. Next run: drop the reranker. Hypothesis — buys ~5–10 MRR points on `definition`/`enumeration` (high lexical overlap, multiple competing chunks) and ~0 on `indirect` (already 0.96 — ceilinged). Marginal gain → 0 around top_n=3; positions 4–5 are mostly padding.

### A6. MiniLM-L6-v2 choice — two failure modes.
Cheap (~80MB), CPU-fast, 384-dim — right for portfolio iteration speed. Failure modes: (1) domain jargon ("Constitutional AI", specific cert terminology) gets generic-embedded, hurting precision; (2) query-vs-chunk asymmetry — MiniLM was trained on symmetric pairs, so a short question vs. a long chunk isn't its strong shape.

### A7. What breaks `no_answer` refusal.
Refusal lives in the prompt ("if context doesn't contain the answer, output `NOT_IN_DOCUMENT`"). What breaks it: cranking top_k. More chunks → higher chance one is thematically adjacent but irrelevant → LLM finds *something* to say. Aggressive retrieval is the enemy of honest refusal.

### A8. Why `definition` (0.78) < `indirect` (0.96)?
Counterintuitive but explainable: definitions usually have one *exact* gold chunk — retrieval must hit it precisely. Indirect questions paraphrase, so multiple chunks reasonably qualify, and Recall@5 forgives near-misses. MRR punishes ranking errors hardest where ground truth is narrow. Lesson: easier-*sounding* queries can be the harder retrieval problem.

---

## B. LLM Application Design

### B1. Why fully-local?
Two reasons: (a) demonstrates a deployment story for users who can't ship data out — regulated industries, air-gapped environments; (b) portfolio differentiator — wiring up OpenAI is table stakes. Cost: 5–10s end-to-end latency, sub-frontier answer quality. Constraint that justifies it: a healthcare/legal customer who'd reject any cloud RAG outright.

### B2. 3B → 7B migration trigger, and signal for 7B → 14B.
Trigger was qualitative: `llama3.2:3b` was generating fluent nonsense on enumeration questions — confidently ignoring the order of items in the retrieved chunk. `qwen2.5:7b` held list structure. Signal for 7B → 14B: when an *answer-quality* eval (not built yet) shows ≥10pp gap on multi-hop or list-aggregation, *and* latency stays usable.

### B3. Grounding prompt sketch.
```
Answer using ONLY the context below.
If the answer is not in the context, output exactly: NOT_IN_DOCUMENT.

Context:
[1] {chunk text}
[2] {chunk text}
...

Question: {q}
Answer (cite chunks like [1], [2]):
```
Refusal must be a *literal exact string* — not "I don't know" or "the context doesn't say." Exact strings are scorable. Add one few-shot example where the context *does* contain the answer, to anchor against over-refusing.

### B4. Guardrails today.
Thin: refusal logic + length cap on input. No PII filter (single-user local). No output post-processor. If multi-tenant: regex check for citation tokens, max-answer-length, and a tiny classifier for off-topic queries *before* retrieval.

### B5. Four signals per trace.
1. Top-5 retrieved chunk IDs + fused scores
2. Reranker scores
3. Final prompt sent to LLM
4. Raw LLM output + token count + latency

Localization: missing from top-5 = retrieval; present but low-ranked = reranker; in prompt but ignored = LLM/prompt; correct but truncated = token cap.

### B6. Graceful degradation.
- Reranker dies → fall back to RRF top-5 verbatim, log warning.
- Ollama down → return retrieved chunks as-is ("LLM unavailable, showing source passages"). Still useful for a study tool.
- Chroma corrupted → block queries, surface a re-index command. Never silently serve stale.

---

## C. Senior Signal (Staff-level framing)

### C1. 90-second walkthrough.
"LocalRAG is a fully-local RAG over the Claude Certified Architect exam guide. Architecture: PyMuPDF → recursive chunking → dual index in Chroma (MiniLM embeddings) + BM25 → hybrid retrieval with RRF → bge-reranker-base, top_n=5 → qwen2.5:7b on Ollama. Trade-offs: RRF over learned fusion for simplicity, 7B over 3B after the smaller model failed on list-aggregation. Failure modes I watch: stale index, no_answer over-refusal, image-only PDF pages. Eval: 60 questions across 5 types — current MRR 0.78–0.96 by type, Recall@5 0.92–1.0. Next: an agentic router for question-type-aware retrieval and a self-correction loop on low-confidence answers."

### C2. 10× traffic — what breaks first?
LLM first (qwen2.5:7b is single-stream on a single GPU). Reranker second (cross-encoder, batchable but heavy). Embedding inference third (cacheable per query string). Chroma last (HNSW reads scale well). Ordering driven by per-query compute and parallelizability.

### C3. 50% cost cut.
Hardware is sunk; the lever is *what you can shut off*. First: drop the reranker — if eval shows the delta is small, the latency/compute payoff is big. Second: quantize qwen2.5:7b harder (Q4 → Q3) and measure. Third: smaller LLM only if quality holds. Embedding is already small.

### C4. Stale index strategy.
Corpus is ~50–100 pages — full rebuild is ~5 min, so I'd rebuild. Drift detection: store a content hash of source PDFs in the Chroma collection manifest; on each query, compare source hash to manifest hash, surface "index stale" if mismatch. For a 10k-doc corpus I'd switch to section-level diffing.

### C5. Where agentic pays off — with numbers.
- **Router** justified by: `definition` at MRR 0.78 (lowest type) — it needs precision-first retrieval; `enumeration` needs recall-first. Same retrieval policy serves both badly.
- **Self-correction** justified by: `negation` and `no_answer` — types where the LLM is most likely to confidently confabulate. A "check the answer against the cited chunk" verifier catches those without hurting easy types.

### C6. What I'd skip.
- **Query rewriting / HyDE:** eval doesn't show short-query failures, and rewriting doubles LLM calls per query.
- **Streaming output:** single-user local app — a 5-second answer is fine, no streaming complexity needed.

---

## D. Evaluation (the section that wins or loses Staff signal)

### D1. Why 60 / 5 types / 12 each?
60 is small enough to label by hand and re-run cheaply (the real constraint), large enough that per-type means aren't dominated by one outlier. 5 types cover the failure-mode space: definitions, lists, paraphrase, negation, refusal. 12/type gives a real per-type signal. 200 would buy tighter confidence intervals but I'd rather iterate on the system 4× than label 4× more questions.

### D2. Retrieval vs. generation eval.
Today is retrieval-only (MRR, Recall@5). Generation plan: an "answer correctness" score — exact-match on numeric/literal answers, LLM-judged semantic match on free-text, 1/0.5/0. Spot-check 10/run by hand to validate the judge.

### D3. Local judge.
`qwen2.5:7b` can't judge itself credibly. Two options: (a) larger local model used *only at eval time* — qwen2.5:14b or 32b, slower but offline-acceptable; (b) one-time cloud judge call to Claude Sonnet — violates fully-local eval. Go with (a), and validate it against ~50 human-labeled answers to measure judge–human agreement before trusting it.

### D4. Missing question types.
- **Multi-hop:** answer requires combining 2+ chunks
- **Temporal:** "what's the latest version mentioned"
- **Conflicting-source:** moot here (single PDF) but matters when this goes multi-doc

### D5. Hallucination rate — operationally.
- **Numerator:** answers where the cited chunk does *not* support the claim
- **Denominator:** answers where the system did NOT output `NOT_IN_DOCUMENT`
- **Labeling:** human reads chunk + answer, marks supported/unsupported

Refusal is *not* hallucination — keep them separate.

### D6. Stop criterion (the question that catches most candidates).
Stop tuning retrieval when Recall@5 ≥ 0.95 *and* MRR ≥ 0.85 on *every* type, for two consecutive eval runs with config changes between them. Current state: 3 of 4 types meet that bar; `definition` at 0.78 MRR is the holdout. One more retrieval experiment focused on definitions, then move to agentic.

### D7. Test-set leakage.
The questions were generated by Claude Opus from the same PDF the system retrieves over, so they (a) over-represent what Opus finds salient and (b) use phrasings close to the source wording. Hides paraphrase failures. Cheapest fix: hand-write 10–15 adversarial questions in your *own* words, using terminology a learner would use, not the PDF's exact terms. ~1 hour of work for a big realism win.

---

## E. Failure-Mode Drills (rapid-fire — answer each in <60s)

### E1. Silent embedding dim swap.
Store embedding model name + dim in Chroma collection metadata; assert match at startup. Without that, mismatched vectors either get rejected or, worse, padded silently.

### E2. Right chunk at rank 11.
Reranker input pool is too narrow. Expand pool from 10 → 20 candidates *before* reranking. If chunk surfaces, that's the fix. If still at 11 post-rerank, the retriever isn't returning it — diagnose embedding and BM25 separately.

### E3. Image-only PDF page.
Today PyMuPDF returns empty string — silently dropped. Fix: OCR pass at ingestion (Tesseract). Cost: slower ingest, new dependency. Only worth it if eval has image-page questions.

### E4. Reranker confidently wrong on negation.
Symptom: `negation` MRR drops while other types hold. Diagnosis: cross-encoders score on lexical + semantic similarity to the *nouns* in the question and don't model the "not" operator. Fix: rewrite negation queries before scoring, or post-filter using a small NLI step.

### E5. Over-refusal.
Track `NOT_IN_DOCUMENT` rate *per type* in the eval. If it spikes on types that *should* have answers (`definition`, `enumeration`), the prompt is over-pushing refusal. Tune the few-shot anchor.

### E6. Chroma corruption.
Re-ingest from source PDFs. Should be deterministic and ~5min. If ingest isn't reproducible (unpinned deps), this breaks — exactly the lesson from `Lessonlearned.md` on pinning requirements.

---

## F. Self-Audit (the brutal one — answer yes/no, then defend)

- **MRR/Recall per type from memory?** Yes — def 0.78/0.92, enum 0.82/0.92, indirect 0.96/1.0, negation 0.92/0.92. *Memorize these. Numbers from memory = credibility.*
- **Retrieval and generation separately?** Only retrieval today. Generation eval is the next gap — flag this proactively in the interview, don't let them catch it.
- **One concrete wrong query?** *Pre-interview prep:* open `eval/results_baseline.jsonl`, find a question with `rr=0`, read its `source_quote`, know *why* it failed. Be able to name it.
- **Different PDF?** Not yet. Single-corpus tuning risk — current numbers might be PDF-specific. Mitigation on roadmap.
- **Single command to re-run?** Likely a notebook cell today, not a CLI script. Worth converting to `python -m eval.run_baseline` before interviewing — reproducibility is a Staff signal.

---

## Drill Tactics

1. **Cold-open drill.** Pick 3 questions you'd *least* want asked. Answer them out loud against a 2-minute timer. Compare against the notes above. Gap = where to study tonight.
2. **Numbers drill.** Recite all 8 per-type metrics (4 types × MRR + Recall) without looking. If you stumble, you don't know your project well enough.
3. **Tradeoff drill.** Every claim you make in an interview should end with "...because [trade-off]." Practice ending sentences that way.
4. **Failure-first drill.** Before describing what your system *does*, describe what could go *wrong* with it. Staff-level pattern.
