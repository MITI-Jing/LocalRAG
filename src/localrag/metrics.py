"""Retrieval metrics and text normalisation."""

import re


def normalize(text: str) -> str:
    """Lowercase, strip markdown emphasis, and collapse all whitespace to single spaces.
    
    pymupdf4llm emits headings and bold runs mid-sentence, so an exact substring
    match against a source quote fails without this.
    """

    text = re.sub(r"[*_#`]+", "", text)
    return " ".join(text.lower().split())

def chunk_contains_quote(chunk_text: str, quote:str) -> bool:
    """True if 'quote' appears inside 'chunk_text' once both sides are normalised."""
    return normalize(quote) in normalize(chunk_text)

def recall_at_k(retrieved_ids: list[str], ground_truth_ids: list[str], k:int) -> float:
    """1.0 if any gold chunk id appears in the top 'k' retrieved ids, else 0.0.
    
    Binary per query, averaged across queries by the caller. With no gold ids this 
    return 0.0, which is why 'evaluate()' excludes no_answer rows rather than 
    scoring them - they have nothing to retrieve.
    """
    return 1.0 if any(gid in retrieved_ids[:k] for gid in ground_truth_ids) else 0.0

def reciprocal_rank(retrieved_ids: list[str], ground_truth_ids: list[str]) -> float:
    """1/rank of the first gold id in 'retrieved_ids', or 0.0 if none of them hit.
    
    Scores the first hit only, so a query with several gold chunks is not rewarded
    for finding more than one.
    """
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in ground_truth_ids:
            return 1.0 / rank
    return 0.0

    

