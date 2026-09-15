"""Retrieval eval: MRR and Recall@k per question type, for all 4 retrievers.

    python -m localrag.evaluate    # appends to eval/results_retrieval
    python -m localrag.evaluate --persist-dir chroma_db_v2
"""

import argparse
import datetime
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from langchain_core.retrievers import BaseRetriever

from localrag.metrics import recall_at_k, reciprocal_rank
from localrag.retrieval import (
    EMBED_MODEL,
    bm25_retriever,
    dense_retriever,
    hybrid_retriever,
    load_chunks,
    load_vectorstore,
    reranked_retriever,
)

K = 5

TESTSET = Path("eval/eval_testset_v1.jsonl")
OUT = Path("eval/results_retrieval.jsonl")

def build_systems(persist_dir: Path) -> dict[str, BaseRetriever]:
    vs = load_vectorstore(persist_dir)
    dense = dense_retriever(vs)
    bm25 = bm25_retriever(load_chunks(vs))
    hybrid = hybrid_retriever(dense, bm25)
    return {"dense": dense, "bm25": bm25, "hybrid": hybrid, "reranked": reranked_retriever(hybrid)}

def evaluate(retriever: BaseRetriever, rows: list[dict], k: int = K) -> dict[str, list[dict]]:
    """Score every answerable row by chunk id. no_answer rows have no gold chunk,so skip them."""
    per_type = defaultdict(list)
    for row in rows:
        if row["type"] == "no_answer":
            continue
        retrieved = [doc.id for doc in retriever.invoke(row["question"])][:k]
        gold = row["ground_truth_chunk_ids"]
        per_type[row["type"]].append({
            "qid": row["id"],
            "rr": reciprocal_rank(retrieved, gold),
            "rk": recall_at_k(retrieved, gold, k),
        })
    return dict(per_type)

def summarize(system: str, per_type: dict[str, list[dict]], k: int = K) -> dict:
    """One results_retrieval.jsonl record: the notebook's shape, plus qid in every raw cell."""
    flat = [cell for cells in per_type.values() for cell in cells]
    return {
        "date": datetime.date.today().isoformat(),
        "system": system,
        "embedding_model": EMBED_MODEL,
        "k": k,
        "n_questions": len(flat),
        "overall": {
            "mrr": mean(c["rr"] for c in flat),
            f"recall_at_{k}": mean(c["rk"] for c in flat),
        },
        "per_type": {
            qtype: {
                "n": len(cells),
                "mrr": mean(c["rr"] for c in cells),
                f"recall_at_{k}": mean(c["rk"] for c in cells),
                "raw": cells,
            }
            for qtype, cells in per_type.items()
        },
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Score the four retrievers on the eval set.")
    parser.add_argument("--persist-dir", type=Path, default=Path("chroma_db"))
    parser.add_argument("--testset", type=Path, default=TESTSET)
    parser.add_argument("--out", type=Path, default=OUT, help="appended to, never overwritten")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.testset.open(encoding="utf-8")]
    with args.out.open("a", encoding="utf-8") as f:
        for name, retriever in build_systems(args.persist_dir).items():
            record = summarize(name, evaluate(retriever, rows))
            f.write(json.dumps(record) + "\n")
            f.flush()
            overall = record["overall"]
            print(f"{name:9s} MRR {overall['mrr']:.4f} Recall@{K} {overall[f'recall_at_{K}']:.4f}")

if __name__ == "__main__":
    main()