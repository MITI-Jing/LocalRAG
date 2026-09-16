"""Query path: retriever + prompt + LLM. The API calls each stage separately so it can time them;
batch scripts call answer()."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama


from localrag.errors import EmptyCorpusError
from localrag.retrieval import (
    bm25_retriever,
    dense_retriever,
    hybrid_retriever,
    load_chunks,
    load_vectorstore,
    reranked_retriever,
)

GEN_MODEL = "qwen2.5:7b"
ABSTAIN_PHRASE = "not found in context" # the rule the judge uses

def build_llm(timeout_s: float | None = None) -> ChatOllama:
    """timeout_s bounds each call; None waits forever, which is what the batch eval wants."""
    return ChatOllama(
        model=GEN_MODEL,
        temperature=0,
        num_predict=1024,
        num_ctx=8192,
        client_kwargs={"timeout": timeout_s},
        )


def build_retriever(persist_dir: Path= Path("chroma_db"), top_n: int = 5) -> BaseRetriever:
    vs = load_vectorstore(persist_dir)
    chunks = load_chunks(vs)
    if not chunks:
        raise EmptyCorpusError(f"{persist_dir} holds no chunks - run `python -m localrag.ingest`")
    dense = dense_retriever(vs)
    bm25 = bm25_retriever(chunks)
    return reranked_retriever(hybrid_retriever(dense, bm25), top_n=top_n)


def build_prompt(question: str, docs: list[Document]) -> str:
    """The v3 generation prompt, verbatim. Changing this text makes it a new prompt version."""
    context = "\n\n".join(f"[SOURCE {i + 1}]:\n{doc.page_content}" for i, doc in enumerate(docs))
    return(
        " Answer the question using only the provided context."
        " If ANY source contains information relevant to the question, you MUST answer from it."
        " If the context covers only part of the answer, give that partial answer."
        " For questions asking which option is NOT in a given list, compare each option"
        " against the context: the one option missing from the context's list is the"
        " answer — this counts as answerable, do not abstain."
        " If the question asks for a list, items, examples, or steps,"
        " include EVERY relevant item from the context — do not summarise or pick representatives."
        " Only when NO source contains anything relevant to the question,reply exactly:"
        " Not found in context."
        "Sources:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:\n"
    )

def is_abstention(result: str) -> bool:
    return ABSTAIN_PHRASE in result.lower()


def answer(question: str, retriever:BaseRetriever, llm:ChatOllama) -> dict:
    """Retrieve, prompt and generate in one call - the bathc path."""
    docs = retriever.invoke(question)
    response = llm.invoke(build_prompt(question,docs))
    return {
        "result": response.content,
        "source_documents": docs,
    }

