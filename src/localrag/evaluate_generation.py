"""Generation eval: answer every test-set question, then grade each answer with the judge.

"""

import argparse
import json
from pathlib import Path
from statistics import mean

from langchain_core.exceptions import OutputParserException

from localrag.evaluate import TESTSET
from localrag.judge import JUDGE_MODEL, JUDGE_PROMPT_V, grade
from localrag.pipeline import (
    GEN_MODEL,
    GEN_PROMPT_V,
    answer,
    build_llm,
    build_retriever,
    format_context,
    is_abstention,
)
from localrag.schemas import AskRequest

RUNS = Path("eval/runs")
METRICS = ("faithfulness", "correctness", "completeness")

def tag(model: str) -> str:
    return model.replace(":", "-")

def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def done_ids(path: Path) -> set[str]:
    return {r["id"] for r in load_jsonl(path)} if path.exists() else set()


def run_generate(rows: list[dict], out: Path, k: int, persist_dir: Path) -> None:
    done = done_ids(out)
    todo = [r for r in rows if r["id"] not in done]
    print(f"generate: {len(done)} done, {len(todo)} to go -> {out}")
    if not todo:
        return
    retriever = build_retriever(persist_dir, top_n=k)
    llm = build_llm()
    with out.open("a", encoding="utf-8") as f:
        for row in todo:
            rag = answer(row["question"], retriever, llm)
            f.write(json.dumps({
                "id": row["id"], "type": row["type"], "question": row["question"],
                "gold": row["answer"], "result": rag["result"],
                "context": format_context(rag["source_documents"]),
                "gen_prompt_v": GEN_PROMPT_V, "k": k,
            }) + "\n")
            f.flush()
            print("generated", row["id"])


def run_judge(gen_file: Path, out: Path) -> None:
    done = done_ids(out)
    todo = [r for r in load_jsonl(gen_file) if r["id"] not in done]
    print(f"judge: {len(done)} done, {len(todo)} to go -> {out}")
    with out.open("a", encoding="utf-8") as f:
        for r in todo:
            try:
                s = grade(r["type"], r["question"], r["gold"], r["result"], r["context"])
            except OutputParserException as e:
                print("judge failed", r["id"], e)
                continue 
            f.write(json.dumps({
                "id": r["id"], "type": r["type"], "judge_prompt_v": JUDGE_PROMPT_V,
                **s.model_dump(),
            }) + "\n")
            f.flush()
            print("judged", r["id"], s.faithfulness, s.correctness, s.completeness)

def summarize(gen_file: Path, judge_file: Path) -> None:
    """Abstention counts come straight from the answers; judge means cover answerable rows only."""
    gen = load_jsonl(gen_file)
    scores = load_jsonl(judge_file)
    no_answer = [r for r in gen if r["type"] == "no_answer"]
    answerable = [r for r in gen if r["type"] != "no_answer"]
    print(f"\njudge {len(scores)}/{len(gen)}")
    print(f"abstained on no_answer  {sum(is_abstention(r['result']) for r in no_answer)}"
          f"/{len(no_answer)}  (want all)")
    print(f"abstained on answerable   {sum(is_abstention(r['result']) for r in answerable)}"
          f"/{len(answerable)} (want none)")
    scored = [s for s in scores if s["type"] != "no_answer"]
    if scored:
        for m in METRICS:
            print(f"{m:13s} {mean(s[m] for s in scored):.3f}   (n={len(scored)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ansers for the eval set, then judge.")
    parser.add_argument("--k", type=int, default=AskRequest.model_fields["k"].default,
                        help="sources passed to the generator(default: what /ask serves)")
    parser.add_argument("--persist-dir", type=Path, default=Path("chroma_db"))
    parser.add_argument("--testset", type=Path, default=TESTSET)
    parser.add_argument("--judge-only", type=Path, metavar="GEN_FILE",
                        help="skip generation and grade this existing gen file")
    parser.add_argument("--out", type=Path, help="judge output (default: named after the gen file)")
    args = parser.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    if args.judge_only:
        gen_file = args.judge_only
    else:
        # k is in the name: the notebook's k=5 files have none, and resume would reuse them
        gen_file = RUNS/ f"gen_{tag(GEN_MODEL)}_{GEN_PROMPT_V}_k{args.k}.jsonl"
        run_generate(load_jsonl(args.testset), gen_file, args.k, args.persist_dir)
    judge_file = args.out or RUNS / f"judge_{tag(JUDGE_MODEL)}_{JUDGE_PROMPT_V}_on_{gen_file.name}"
    run_judge(gen_file, judge_file)
    summarize(gen_file, judge_file) 


if __name__ == "__main__":
    main()