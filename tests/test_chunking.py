"""Unit tests for the chunker - text in, Documents out, so no PDF and no model are needed."""

from localrag.chunking import chunk_recursive_with_breadcrumbs, split_by_headers


def chunk(md: str, **kwargs):
    return chunk_recursive_with_breadcrumbs(split_by_headers(md), **kwargs)

def test_breadcrumb_joins_h2_and_h3():
    [c] = chunk("## Domain 1\nSome text.")
    assert c.page_content == "Section: Domain 1\n\nSome text."

def test_headers_stay_in_metadata():
    [c] = chunk("## Domain 1\n### Task 1.1\nSome text.")
    assert c.metadata == {"H2": "Domain 1", "H3": "Task 1.1"}

def test_chunk_size_limits_the_body_not_the_breadcrumb():
    """The breadcrumb is added after splitting, so a full chunk can exceed chunk_size."""
    body = " ".join(["word"] * 100)
    prefix = "Section: Domain 1\n\n"
    chunks = chunk(f"## Domain 1\n{body}", chunk_size=100, chunk_overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        assert c.page_content.startswith(prefix)
        assert len(c.page_content) - len(prefix) <= 100

def test_text_before_first_heading_gets_an_empt_breadcrumb():
    """In the real index this is the document title - H1 is no a split level."""
    first, *_ = chunk("Preamble text. \n## Domain 1\n\nSome text.")
    assert first.page_content == "Section: \n\nPreamble text."