"""evaluate() scores by chunk id and records the query id in every raw cell."""

from langchain_core.documents import Document

from localrag.evaluate import evaluate


class FakeRetriever:
    def __init__(self, ranking: dict[str, list[str]]):
        self.ranking = ranking

    def invoke(self, question: str) -> list[Document]:
        return [Document(id=cid, page_content=cid) for cid in self.ranking[question]]

def row(qid: str, qtype: str, question: str, gold: list[str]) -> dict:
    return {"id": qid, "type": qtype, "question": question, "ground_truth_chunk_ids": gold}

def test_cells_carry_qid_and_scores_and_skip_no_answer():
    rows = [
        row("Q01", "definition", "a", ["chunk-0002"]),
        row("Q02", "definition", "b", ["chunk-0009"]),
        row("Q49", "no_answer", "c", []),
    ]
    ranking = {"a": ["chunk-0001", "chunk-0002"], "b": ["chunk-0001"], "c": ["chunk-0001"]}

    per_type = evaluate(FakeRetriever(ranking), rows, k=5)

    assert per_type == {
        "definition": [
            {"qid": "Q01", "rr": 0.5, "rk": 1.0},
            {"qid": "Q02", "rr": 0.0, "rk": 0.0},
        ]
    }