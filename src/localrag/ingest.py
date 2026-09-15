"""Build the Chroma index from the PDF. Run once, offline:

python -m localrag.ingest   # uses the single PDF in data/
python -m localrag.ingest path/to/file.pdf
"""

import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from localrag.chunking import chunk_recursive_with_breadcrumbs, load_markdown, split_by_headers
from localrag.retrieval import COLLECTION, EMBED_MODEL

DATA_DIR = Path("data")

def build_index(pdf_path: Path, persist_dir: Path = Path("chroma_db")) -> int:
    """load_markdown -> split_by_headers -> chunk_with_breadcrumbs -> Chroma.
    Chunk ids are chunk-0000, chunk-0001, ... in chunking order, so the same PDF and the 
    same chunker always give the same ids. Refuses to overwrite an existing index.
    Returns the number of chunks indexed.
    """

    if persist_dir.exists():
        raise FileExistsError(
            f"{persist_dir} already exists - delete it first to rebuild"
            )

    chunks = chunk_recursive_with_breadcrumbs(split_by_headers(load_markdown(pdf_path)))
    for chunk in chunks:
        chunk.metadata["source_doc"] = pdf_path.name

    Chroma.from_documents(
        chunks,
        HuggingFaceEmbeddings(model_name=EMBED_MODEL),
        ids=[f"chunk-{i:04d}" for i in range(len(chunks))],
        collection_name=COLLECTION,
        persist_directory=str(persist_dir),
    )
    return len(chunks)

def default_pdf() -> Path:
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if len(pdfs) != 1:
        raise SystemExit(
            f"Expected exactly one PDF in {DATA_DIR}/, found {len(pdfs)} - pass a path"
        )
    return pdfs[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Chroma index from the PDF.")
    parser.add_argument("pdf", nargs="?", type=Path, help="defaults to the single PDF in data/")
    parser.add_argument("--persist-dir", type=Path, default=Path("chroma_db"))
    args = parser.parse_args()

    n = build_index(args.pdf or default_pdf(), args.persist_dir)
    print(f"Indexed {n} chunks into {args.persist_dir}")

