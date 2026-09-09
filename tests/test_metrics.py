"""Unit tests for the retrieval metrics every commited eval number dpends on."""

from localrag.metrics import chunk_contains_quote, normalize, recall_at_k, reciprocal_rank

class TestReciprocalRank:
    def test_gold_at_first_position(self):
        assert reciprocal_rank(["gold", "b", "c"], ["gold"]) == 1.0

    def test_gold_at_second_position(self):
        assert reciprocal_rank(["a", "gold", "c"], ["gold"]) == 0.5

    def test_no_hit_is_zero(self):
        assert reciprocal_rank(["a", "b"], ["gold"]) == 0.0

    def test_empty_retrieval_is_zero(self):
        assert reciprocal_rank([], ["gold"]) == 0.0

    def test_scores_first_hit_only_when_several_gold_chunks(self):
        # Gold at ranks 2 and 3 -> 1/2, never 1/3 and never a sum.
        assert reciprocal_rank(["a", "g1", "g2"], ["g1", "g2"]) == 0.5   

    def test_no_gold_ids_is_zero(self):
        # The no_answer case. evaluate() skips these rows.
        assert reciprocal_rank(["a", "b"], []) == 0.0

class TestRecallAtK:
    def test_respects_k(self):
        ranked = ["a", "b", "c", "d", "gold"]
        assert recall_at_k(ranked, ["gold"], k=5) == 1.0
        assert recall_at_k(ranked,["gold"], k=3) == 0.0

    def test_position_k_is_inside_the_window(self):
        assert recall_at_k(["a", "b", "gold"], ["gold"], k=3) == 1.0
        assert recall_at_k(["a", "b", "gold"], ["gold"], k=2) == 0.0

    def test_any_single_gold_chunk_counts(self):
        assert recall_at_k(["a", "g2"], ["g1", "g2"], k=5) == 1.0

    def test_miss_is_zero(self):
        assert recall_at_k(["a", "b"], ["gold"], k=5) == 0.0


    def test_no_gold_ids_is_zero(self):
        assert recall_at_k(["a", "b"], [], k=5) == 0.0


class TestNormalize:
    def test_strips_markdown_emphasis(self):
            assert normalize("**Claude _Architect_") == "claude architect"


    def test_collapses_newlines_and_runs_of_spaces(self):
        assert normalize("line one\n\n  line two") == "line one line two"

    def test_strips_heading_markers_and_backticks(self):
        assert normalize("## `Overview`") == "overview"


class TestChunkContainsQuote:
    def test_matches_across_markdown_and_line_breaks(self):
        # The reason normalize() exists: this quote is one sentence in the PDF but
        # arrives from pymupdf4llm split by a heading, a bold run and a newline.

        chunk = "## Overview\nThe **Foundations** certification   validates\ntradeoffs."
        quote = "The Foundations certification validates tradeoffs."
        assert chunk_contains_quote(chunk, quote)

    def test_absent_quote_is_false(self):
        assert not chunk_contains_quote("Some other text.", "Foundations certification")




