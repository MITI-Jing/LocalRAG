# LocalRAG

A fully local RAG pipeline. No cloud calls anywhere in the inference path or eval loop. Embedding, retrieval, generation, and judging all run on local models via Ollama and sentence-transformers.

I'm building this in public on LinkedIn while moving toward production-grade agentic.

Corpus: the Claude Certified Architect Foundations Exam Guide (a single PDF, for now).

## Current state (as of 2026-05-25)

Retrieval baseline on a 60-question typed eval set, evenly distributed across 5 query types (12 each). MRR and Recall@5 measured on the 4 answerable types:

| Type | MRR | Recall@5 |
|---|---|---|
| definition | 0.78 | 0.92 |
| enumeration | 0.82 | 0.92 |
| indirect | 0.96 | 1.00 |
| negation | 0.92 | 0.92 |

`no_answer` queries (n=12) are scored separately on abstain accuracy, not MRR.

Overall MRR improved from 0.72 (semantic-only baseline) to 0.76 (hybrid semantic + BM25) to 0.87 (hybrid + cross-encoder reranker). Enumeration and negation are still the weakest types because list items chunk badly across boundaries; that's the next thing to fix.

Generation runs on `qwen2.5:7b`. Eval scoring uses a different local model as judge (`llama3.1:8b` or `gemma2:9b`) to avoid the self-grading inflation effect.

## Architecture

```
ingestion
  PDF (data/)
    → pymupdf4llm extraction
    → recursive character splitting
    → ChromaDB (sentence-transformers/all-MiniLM-L6-v2 embeddings)

query
  user question
    → semantic retrieval (Chroma)
    + lexical retrieval (rank-bm25)
    → RRF fusion
    → BAAI/bge-reranker-base cross-encoder
    → top 5 chunks
    → qwen2.5:7b generation (extraction prompt + synthesis prompt)
    → answer

eval (offline batch)
  question, ground-truth answer, retrieved chunks, generated answer
    → judge model: llama3.1:8b or gemma2:9b
    → faithfulness, answer relevancy, completeness scores
```

## Why these design choices

**Hybrid retrieval + reranker.** Semantic search misses on enumeration and negation queries where lexical overlap matters; BM25 catches those. RRF fusion then bge-reranker-base gave the biggest single jump (0.76 to 0.87 overall MRR) for moderate complexity.

**Cross-model LLM-as-judge.** Same-model judging inflates scores because the judge agrees with its own output. Using a different model family for the judge removes that. Even a same-class local model is a defensible judge once you give it the ground-truth answer as an answer key.

**qwen2.5:7b for generation.** Hit a synthesis ceiling on llama3.2:3b (couldn't combine across multiple chunks even with explicit prompts). llama3.1:8b improved but still dropped facts. qwen2.5:7b is the smallest model in my hardware envelope that follows multi-source synthesis instructions reliably.

**Single PDF for now.** Production-grade means an ingestion pipeline for N documents; I'm staying single-doc until the agentic and eval layers are solid, then scaling the corpus.

## What works

- Hybrid + reranker lifts overall MRR from 0.72 to 0.87.
- Per-query-type analysis surfaces specific failure modes instead of a single noisy aggregate number.
- Cross-model judging produces defensible eval signal without any cloud calls during eval runs.
- Recall@5 = 1.00 on indirect queries means the answer is always findable in the top-5 when it requires paraphrase or 1-2-sentence combination.

## What doesn't work yet

- Pure RAG, no agent layer. Multi-hop questions fail because the system can't decompose them.
- Single PDF corpus. No ingestion pipeline for N documents with reprocessing.
- Notebook execution. Not deployed as a service.
- No tracing or observability. LangFuse is next.

## Roadmap to production-grade agentic

1. RAGAS-based scoring on the cross-model judge for reproducible metrics.
2. Notebook to FastAPI service, Dockerised, env-config, basic auth.
3. LangFuse for tracing every retrieval and generation step, latency breakdown, eval scoring over time.
4. Agentic layer: query router (does this need retrieval?), query decomposition for multi-hop, tool use beyond retrieval.
5. Multi-document ingestion pipeline with reprocessing logic.
6. CI-triggered regression eval on every commit, with results posted to PR.

## Run it

```bash
git clone https://github.com/MITI-Jing
cd LocalRAG
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen2.5:7b llama3.1:8b gemma2:9b
jupyter notebook localrag_ingestion.ipynb
```


## Stack

- **Embedding:** sentence-transformers/all-MiniLM-L6-v2
- **Vector store:** ChromaDB
- **Retrieval:** hybrid (Chroma + rank-bm25) with RRF fusion
- **Reranker:** BAAI/bge-reranker-base cross-encoder
- **Generator:** qwen2.5:7b via Ollama
- **Judge:** llama3.1:8b or gemma2:9b via Ollama
- **Doc loader:** pymupdf4llm (after pymupdf created chunk boundary issues at page breaks)
- **Orchestration:** LangChain
- **Eval:** custom typed test set + RAGAS (in progress)

## Files in this repo

- `localrag_ingestion.ipynb` — main pipeline (ingestion through generation)
- `eval/eval_testset_v1.jsonl` — 60-question typed eval set with ground-truth chunk IDs
- `eval/eval testset prompt.md` — the prompt used to seed the test set (one cloud call, hand-reviewed)
- `eval/results_baseline.jsonl` — current baseline retrieval results


## Limitations

Working portfolio project. Runs on one corpus. The eval set was seeded by one cloud call (Claude) and hand-reviewed; eval execution stays fully local. Performance numbers above are on my specific hardware and corpus and will move if either changes.
