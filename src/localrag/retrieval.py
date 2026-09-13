"""Query-time retrievers, all built from the persisted Chroma collection."""

from pathlib import Path

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings

COLLECTION = "md_chunks_breadcrumb"
EMBED_MODEL ="sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL ="BAAI/bge-reranker-base"

def load_vectorstore(persist_dir: Path = Path("chroma_db")) -> Chroma:
    """Open the existing index. Never builds one - that is ingest.build_index's job."""
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"No index at {persist_dir} - build one with `python -m localrag.ingest`"
        )
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
        persist_directory=str(persist_dir),
    )


def load_chunks(vectorestore: Chroma) -> list[Document]:
    """Every stored chunk as a Documents, carrying its Chroma id on Document.id."""
    stored = vectorestore.get(include=["documents", "metadatas"])

    return [
        Document(id=chunk_id, page_content=text, metadata=metadata or {})
        for chunk_id, text, metadata in zip(
            stored["ids"], stored["documents"], stored["metadatas"], strict=True
        )
    ]


def dense_retriever(vectorstore: Chroma, k:int = 10) -> BaseRetriever:
    return vectorstore.as_retriever(search_kwargs={"k": k})


def bm25_retriever(chunks: list[Document], k: int = 10) -> BaseRetriever:
    return BM25Retriever.from_documents(chunks, k=k)


def hybrid_retriever(
        dense: BaseRetriever, bm25: BaseRetriever, 
        weights: tuple[float, float] = (0.3, 0.7)) -> BaseRetriever:
    """Weighted reciprocal-rank fusion of dense + BM25; BM25 weighted heavier."""

    return EnsembleRetriever(retrievers=[dense, bm25], weights=list(weights))

def reranked_retriever(base: BaseRetriever, top_n: int = 5) -> BaseRetriever:
    """Cross-encoder rerank of base's candidates - the production retriever."""
    
    reranker = CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_MODEL), top_n=top_n)

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base)



