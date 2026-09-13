from pathlib import Path

from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama

from localrag.retrieval import (
    bm25_retriever,
    dense_retriever,
    hybrid_retriever,
    load_chunks,
    load_vectorstore,
    reranked_retriever,
)


def build_llm() -> ChatOllama:
    return ChatOllama(model="qwen2.5:7b",temperature=0,num_predict=1024,num_ctx=8192)


def build_retriever(persist_dir: Path= Path("chroma_db")) -> BaseRetriever:
    vs = load_vectorstore(persist_dir)
    dense = dense_retriever(vs)
    bm25 = bm25_retriever(load_chunks(vs))
    return reranked_retriever(hybrid_retriever(dense, bm25))


def answer(question: str, retriever:BaseRetriever, llm:ChatOllama) -> dict:
    """Retrieve, build the v3 prompt from the sources, and ask the LLM."""
    docs = retriever.invoke(question)
    context = "\n\n".join(
        f"[SOURCE {i+1}]:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    prompt = (
        f" Answer the question using only the provided context."
        f" If ANY source contains information relevant to the question, you MUST answer from it."
        f" If the context covers only part of the answer, give that partial answer."
        f" For questions asking which option is NOT in a given list, compare each option"
        f" against the context: the one option missing from the context's list is the"
        f" answer — this counts as answerable, do not abstain."
        f" If the question asks for a list, items, examples, or steps,"
        f" include EVERY relevant item from the context — do not summarise or pick representatives."
        f" Only when NO source contains anything relevant to the question,reply exactly:"
        f" Not found in context."
        f"Sources:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:\n"
    )

    response = llm.invoke(prompt)
    return {
        "result": response.content,
        "source_documents": docs,
    }

