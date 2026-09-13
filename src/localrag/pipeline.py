from localrag.retrieval import load_vectorstore,reranked_retriever,hybrid_retriever,dense_retriever,bm25_retriever,load_chunks
from langchain_core.documents import question 

vs = load_vectorstore()
retriever = reranked_retriever(hybrid_retriever(dense_retriever(vs), bm25_retriever(load_chunks(vs))))
docs = retriever.invoke(question)
