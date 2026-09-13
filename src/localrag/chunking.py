""" PDF -> markdown -> header sections -> breadcrumbed chunks. No models, no index."""

from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter
from pathlib import Path
import pymupdf4llm
from langchain_core.documents import Document
import re


HEADERS_TO_SPLIT_ON = [("##", "H2"), ("###", "H3")]

def load_markdown(pdf_path: Path) -> str:
    """pymupdf4llm.to_markdown + two cleanup regexes."""
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    md_text = re.sub(r"^\s*-{3,}\s*$","",md_text, flags=re.MULTILINE)

    return re.sub(r"\n{3,}","\n\n", md_text)
    

def split_by_headers(md_text: str) -> list[Document]:
    "MarkdownHeaderTextSplitter"
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    return splitter.split_text(md_text)


def chunk_recursive_with_breadcrumbs(
        sections: list[Document],chunk_size: int = 1000,
                                     chunk_overlap : int = 200) -> list[Document]:
    """Recurisve split, then prefix each chunk with 'Section: H2 > H3'.
Keeps the splitter's default separators - the existing index was build with them.
"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks=splitter.split_documents(sections)

    for chunk in chunks:
        breadcrumb = " > ".join(
            v for k, v in chunk.metadata.items() if k.lower().startswith("h")
        )
        chunk.page_content = f"Section: {breadcrumb}\n\n{chunk.page_content}"

    return chunks


