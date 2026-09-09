"""Invariants of the 60-row eval test set.

eval/build_report.py joins retrieval results to queries *by position*,assuming the
test set is grouped by type in TYPE_ORDER with exactly 12 rows per type. Nothing
enforced that until this file. If a row is added, reordered or retyped, the report
silently attributes scores to the wrong question - these tests fail instead.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTSET = REPO_ROOT / "eval" /"eval_testset_v1.jsonl"

# The four retrieval-scored types, in the order build_report.py indexes them,
# followed by no_answer, which evaluate() skips.
RETRIEVAL_TYPES = ["definition", "enumeration", "indirect", "negation"]
ALL_TYPES = [*RETRIEVAL_TYPES, "no_answer"]
ROWS_PER_TYPE =12


@pytest.fixture(scope="module")
def rows():
    with TESTSET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def test_every_line_is_valid_json(rows):
    assert rows, "test set is empty"

def test_row_count(rows):
    assert len(rows) == len(ALL_TYPES) * ROWS_PER_TYPE == 60

def test_ids_are_unique_and_sequential(rows):
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids)
    assert ids == [f"Q{i:02d}" for i in range(1, len(rows) + 1)]

def test_twelve_rows_per_type(rows):
    assert Counter(r["type"] for r in rows) == dict.fromkeys(ALL_TYPES, ROWS_PER_TYPE)

def test_types_are_contiguous_and_in_report_order(rows):
    """The positional join needs the file grouped, not just balanced."""
    runs = []
    for row in rows:
        if not runs or runs[-1][0] != row["type"]:
            runs.append([row["type"], 0])
        runs[-1][1] += 1
    assert runs == [[t, ROWS_PER_TYPE] for t in ALL_TYPES]

def test_required_fields_present(rows):
    required = {"id", "type", "question", "answer", "section", "page",
                "source_quote", "ground_truth_chunk_ids"}
    for row in rows:
        assert required <= row.keys(), f"{row.get('id')} is missing {required - row.keys()}"

def test_answerable_rows_carry_gold_chunks_and_a_quote(rows):
    answerable = [r for r in rows if r["type"] != "no_answer"]
    assert len(answerable) == len(RETRIEVAL_TYPES) * ROWS_PER_TYPE ==48
    for row in answerable:
        assert row["ground_truth_chunk_ids"], f"{row['id']} has no gold chunk_ids"
        assert row["source_quote"].strip(), f"{row['id']} has no source quote"

def test_no_answer_rows_carry_no_gold_chunks(rows):
    for row in (r for r in rows if r["type"] == "no_answer"):
        assert not row["ground_truth_chunk_ids"], f"{row['id']} should have no gold chunk ids"

