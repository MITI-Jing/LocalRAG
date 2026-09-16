
"""API behaviour with a fake retriever and LLM - no models, no Ollama, so it runs in CI."""

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from localrag.api import Services, create_app

DOCS = [
    Document(
        id=f"chunk-{i:04d}",
        page_content=f"Section: Domain {i}\n\ntext {i}",
        metadata={"H2": f"Domain {i}", "source_doc": "guide.pdf"},
    )
    for i in range(6)
]

class FakeRetriever:
    def __init__(self, docs: list[Document]):
        self.docs = docs

    def invoke(self, question:str) -> list[Document]:
        return list(self.docs)

class BrokenRetriever:
    def invoke(self, question: str) -> list[Document]:
        raise RuntimeError("collection corrupted")

class FakeLLM:
    """Plays back a script: each item is a reply string, or an exception to raise."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = 0

    def invoke(self, prompt: str) -> AIMessage:
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        usage = {"input_tokens": 50, "output_tokens": 7, "total_tokens": 57}
        return AIMessage(content=item, usage_metadata=usage)

def client_for(llm, docs=DOCS, indexed=True, ollama=(True, True), retriever=None):
    services = Services(
        retriever=retriever or (FakeRetriever(docs) if indexed else None),
        llm=llm,
        ollama_status=lambda: ollama,
        chunk_count=lambda: len(docs) if indexed else 0,
    )
    return TestClient(create_app(lambda: services))

def ask(client, **body):
    return client.post("/ask", json={"question": "What does the exam cover?" **body})

def test_ask_returns_answer_sources_and_measurements():
    with client_for(FakeLLM("The exam covers five domains.")) as client:
        r = ask(client)
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]  == "The exam covers five domains."
    assert body["abstained"] is False
    assert body["degraded"] is False
    assert [s["chunk_id"] for s in body["sources"]] == [f"chunk-{i:04d}" for i in range(5)]
    assert body["sources"][0]["section"] == "Domain 0"
    assert body["sources"][0]["source_doc"] == "guide.pdf"
    assert body["tokens"] == {"input": 50, "output": 7}
    assert set(body["timings_ms"]) == {"retrieve_ms", "generate_ms", "total_ms"}
    assert len(body["trace_id"]) == 32

def test_k_limits_the_sources():
    with client_for(FakeLLM("ok")) as client:
        assert len(ask(client, k=2).json()["sources"]) == 2

def test_abstention_is_flagged():
    with client_for(FakeLLM("Not found in context.")) as client:
        assert ask(client).json()["abstained"] is True

@pytest.mark.parameterize(
    "body",
    [
        {"question": "hi"},
        {"question": "long enough", "k": 0},
        {"question": "long enough", "k": 21},
    ],
)

def test_invalid_requests_are_422(body):
    with client_for(FakeLLM()) as client:
        assert client.post("/ask", json=body).status_code == 422

def test_generation_timeout_degrades_to_passages_without_retrying():
    llm = FakeLLM(httpx.ReadTimeout("timed out"))
    with client_for(llm) as client:
        r = ask(client)
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["answer"] == ""
    assert len(body["sources"]) == 5
    assert llm.calls == 1

def test_timeout_with_no_passages_is_504():
    with client_for(FakeLLM(httpx.ReadTimeout("timed out")), docs=[]) as client:
        r = ask(client)
    assert r.status_code == 504
    assert r.json()["error"] == "generation_timeout"

def test_ollama_down_retries_once_then_503():
    llm = FakeLLM(httpx.ConnectError("refused"), httpx.ConnectError("refused"))
    with client_for(llm) as client:
        r = ask(client)

    assert r.status_code == 503
    assert r.json()["error"] == "ollama_unavailable"
    assert len(r.json()["trace_id"]) == 32
    assert llm.calls == 2

def test_ollama_blip_recovers_on_the_retry():
    llm = FakeLLM(httpx.ConnectError("refused"), "Recoverd answer.")
    with client_for(llm) as client:
        r = ask(client)
    assert r.status_code == 200
    assert r.json()["answer"] == "Recovered answer."

def test_no_index_is_409():
    with client_for(FakeLLM(), indexed=False) as client:
        r = ask(client)
    assert r.status_code == 409
    assert r.json()["error"] == "empty_corpus"

def test_retrieval_failure_is_500():
    with client_for(FakeLLM(), retriever=BrokenRetriever()) as client:
        r = ask(client)
    assert r.status_code == 500
    assert r.json()["error"] == "retrieval_error"

def test_health_ok():
    with client_for(FakeLLM()) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "ollama": True, "model_available": True, "chunks": 6}


@pytest.mark.parametrize(
    ("ollama", "indexed"),
    [((False, False), True), ((True, False), True), ((True, True), False)],
)
def test_health_is_503_when_anything_is_missing(ollama, indexed):
    with client_for(FakeLLM(), indexed=indexed, ollama=ollama) as client:
        r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "unhealthy"
    