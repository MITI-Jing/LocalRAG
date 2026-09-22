# LocalRAG

[![tests](https://github.com/MITI-Jing/LocalRAG/actions/workflows/tests.yml/badge.svg)](https://github.com/MITI-Jing/LocalRAG/actions/workflows/tests.yml)


A fully local RAG pipeline. No cloud calls anywhere in the inference path or eval loop. Embedding, retrieval, generation, and judging all run on local models via Ollama and sentence-transformers.

I'm building this in public on LinkedIn while moving toward production-grade agentic.

Corpus: the Claude Certified Architect Foundations Exam Guide (a single PDF, for now).


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

**Hybrid retrieval + reranker.** Semantic search misses on enumeration and negation queries where lexical overlap matters; BM25 catches those. RRF fusion then bge-reranker-base took overall MRR from 0.72 (dense alone) to 0.87 — the largest gain of any single change, for moderate complexity.

**Cross-model LLM-as-judge.** Same-model judging inflates scores because the judge agrees with its own output. Using a different model family for the judge removes that. Even a same-class local model is a defensible judge once you give it the ground-truth answer as an answer key.

**qwen2.5:7b for generation.** Hit a synthesis ceiling on llama3.2:3b (couldn't combine across multiple chunks even with explicit prompts). llama3.1:8b improved but still dropped facts. qwen2.5:7b is the smallest model in my hardware envelope that follows multi-source synthesis instructions reliably.

**Single PDF for now.** Production-grade means an ingestion pipeline for N documents; I'm staying single-doc until the agentic and eval layers are solid, then scaling the corpus.




## Retrieval results

48 answerable questions at k=5 (the 12 `no_answer` rows carry no gold chunk and are excluded). Reproduce with `python -m localrag.evaluate`.

|                        | dense  | bm25   | hybrid | reranked   |
| ---------------------- | ------ | ------ | ------ | ---------- |
| **MRR**                | 0.7167 | 0.6719 | 0.7917 | **0.8681** |
| **Recall@5**           | 0.8542 | 0.8542 | 0.9167 | **0.9375** |
| Questions hit (of 48)  | 41     | 41     | 44     | **45**     |
| MRR — definition       | 0.8333 | 0.7917 | 0.8333 | 0.7778     |
| MRR — enumeration      | 0.7083 | 0.5139 | 0.7778 | 0.8194     |
| MRR — indirect         | 0.5333 | 0.7292 | 0.8611 | 0.9583     |
| MRR — negation         | 0.7917 | 0.6528 | 0.6944 | 0.9167     |

Dense and BM25 tie on Recall@5 — 41/48 each — but fail on *different* questions, which is what makes fusing them worth the complexity. Reranking then adds just one question of recall (44 → 45) while moving MRR 0.79 → 0.87: it reorders the pool rather than widening it. The honest regression: definition MRR drops 0.8333 → 0.7778 under reranking.

## Run it

```bash
git clone https://github.com/MITI-Jing
cd LocalRAG
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ollama pull qwen2.5:7b llama3.1:8b gemma2:9b
python -m localrag.ingest       # build chroma_db/ with chunk-0000 ....
python -m localrag.evaluate     # reproduces the retrieval table
uvicorn localrag.api:app --port 8000
# or: docker compose up --build
```


## Stack

- **Embedding:** sentence-transformers/all-MiniLM-L6-v2
- **Vector store:** ChromaDB
- **Retrieval:** hybrid (Chroma + rank-bm25) with RRF fusion
- **Reranker:** BAAI/bge-reranker-base cross-encoder
- **Generator:** qwen2.5:7b via Ollama
- **Judge:** gemma2:9b via Ollama
- **Doc loader:** pymupdf4llm (after pymupdf created chunk boundary issues at page breaks)
- **Orchestration:** LangChain
- **Eval:** custom typed test set, hand-built retrieval metrics, cross-model LLM judge

## Files in this repo

- `localrag_ingestion.ipynb` — main pipeline (ingestion through generation)
- `eval/eval_testset_v1.jsonl` — 60-question typed eval set with ground-truth chunk IDs (48 answerable, scored; 12 `no_answer`)
- `eval/eval testset prompt.md` — the prompt used to seed the test set (one cloud call, hand-reviewed)
- `eval/results_retrieval.jsonl` — every retrieval eval run, appended (never overwritten)


## Limitations

Working portfolio project. Runs on one corpus. The eval set was seeded by one cloud call (Claude) and hand-reviewed; eval execution stays fully local. Performance numbers above are on my specific hardware and corpus and will move if either changes.
