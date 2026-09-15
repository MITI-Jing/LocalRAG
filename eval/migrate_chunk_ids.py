"""One-off: move_ground_truth_chunk_ids from the random-UUID index to the deterministic one.

Chunks are matched on exact text, so every gold label - including any fixed by hand - carriers
over unchanged. Nothing is written unless every old id finds its chunk in the new index.

    python eval/migrate_chunk_ids.py --old chroma_db  --new chroma_db_v2
"""

import argparse
import json
from pathlib import Path

from localrag.retrieval import load_chunks, load_vectorstore

def chunk_texts(persist_dir: Path) -> dict[str, str]:
    """chunk id -> chunk text"""
    return {c.id: c.page_content for c in load_chunks(load_vectorstore(persist_dir))}

def main() -> None:
    parser = argparse.ArgumentParser(description="Move gold chunk ids to a rebuild index.")
    parser.add_argument("--old", type=Path, default=Path("chroma_db"))
    parser.add_argument("--new", type=Path, default=Path("chroma_db_v2"))
    parser.add_argument("--testset", type=Path, default=Path("eval/eval_testset_v1.jsonl"))
    args = parser.parse_args()

    old = chunk_texts(args.old)
    new = chunk_texts(args.new)
    new_by_text = {text: cid for cid, text in new. items()}
    if len(new_by_text) != len(new):
        raise SystemExit(f"{args.new} has duplicate chunk texts - text cannot identify a chunk")

    rows = [json.loads(line) for line in args.testset.open(encoding="utf-8")]
    unmatched = [
        (row["id"], cid)
        for row in rows
        for cid in row["ground_truth_chunk_ids"]
        if old.get(cid) not in new_by_text
    ]
    if unmatched:
        raise SystemExit(
            f"{len(unmatched)} gold ids have no identical chunk in {args.new}: {unmatched[:5]}"
        )

    for row in rows:
        gold = row["ground_truth_chunk_ids"]
        row["ground_truth_chunk_ids"] = [new_by_text[old[cid]] for cid in gold]
    with args.testset.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    n = sum(len(row["ground_truth_chunk_ids"]) for row in rows)
    print(f"Migrated {n} gold ids across {len(rows)} rows in {args.testset}")


if __name__ == "__main__":
    main()