"""HTTP service: POST /ask and GET /health. Models load once, at startup.
    uvicorn localrag.api:app --port 8000      # interactive docs at http://localhost:8000/docs

Config (env): LOCALRAG_INDEX (default chroma_db), LOCALRAG_GEN_TIMEOUT_S(default 120),
OLLAMA_HOST (read by the Ollama client; default http://localhost:11434).
"""

import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import ollama
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever

from localrag import pipeline
from localrag.errors import (
    EmptyCorpusError,
    GenerationTimeout,
    LocalRAGError,
    OllamaUnavailable,
    RetrievalError,
)

from localrag.retrieval import count_chunks
from localrag.schemas import (
    MAX_K,
    AskRequest,
    AskResponse,
    ErrorResponse,
    Health,
    Source,
    Timings,
    TokenUsage,
)

log = logging.getLogger("localrag")

def configure_logging() -> None:
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False

def log_event(stage: str, trace_id: str, **fields: Any) -> None:
    """One JSON line per stage, every line carrying the trace_id."""
    log.info(json.dumps({"stage": stage, "trace_id": trace_id, **fields}))


@dataclass
class Services:
    """Everything a request needs: built by default_services() in production, faked in tests."""

    retriever: BaseRetriever | None    # None = nothing ingested
    llm: Any   # anything with .invoke(prompt) -> AIMessage
    ollama_status: Callable[[], tuple[bool, bool]]  #(daemon reachable, model pulled)
    chunk_count: Callable[[], int]

def ollama_status(model: str) -> tuple[bool, bool]:
    try:
        models = ollama.Client(timeout=2).list().models
    except Exception:
        return False, False
    return True, any(m.model == model for m in models)

def default_services() -> Services:
    index = Path(os.environ.get("LOCALRAG_INDEX", "chroma_db"))
    timeout_s = float(os.environ.get("LOCALRAG_GEN_TIMEOUT_S", "120"))
    try:
        # top_n=MAX_K keeps every candidate in reranker order; /ask slices the first k.
        retriever = pipeline.build_retriever(index, top_n=MAX_K)
    except (FileNotFoundError, EmptyCorpusError):
        retriever = None
    return Services(
        retriever=retriever,
        llm=pipeline.build_llm(timeout_s=timeout_s),
        ollama_status=lambda: ollama_status(pipeline.GEN_MODEL),
        chunk_count=lambda: count_chunks(index),
    )

def generate(llm: Any, prompt: str) -> AIMessage:
    """One LLM call, with transport failures translated into the error taxonomy."""
    try:
        return llm.invoke(prompt)
    except httpx.TimeoutException as e:  # before TransportError: it is a subclass of it
        raise GenerationTimeout("no reply within the generation budget") from e
    except (httpx.TransportError, ollama.ResponseError) as e:
        raise OllamaUnavailable(f"{type(e).__name__}: {e}") from e

def generate_with_retry(llm: Any, prompt: str, trace_id: str) -> AIMessage:
    try:
        return generate(llm, prompt)
    except OllamaUnavailable as e:   # one retry, for this failure only - nver for timeouts
        log_event("retry", trace_id, reason=str(e))
        return generate(llm, prompt)


def to_source(doc: Document) -> Source:
    section = " > ".join(v for k, v in doc.metadata.items() if k.lower().startswith("h"))
    return Source(
        chunk_id=doc.id or "",
        section=section,
        source_doc=doc.metadata.get("source_doc"),
        text=doc.page_content,
    )

def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)

ERRORS = {code: {"model": ErrorResponse} for code in (409, 500, 503, 504)}

def create_app(services_factory: Callable[[], Services] = default_services) -> FastAPI:
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services_factory()
        yield

    app = FastAPI(title="LocalRAG", lifespan=lifespan)

    @app.exception_handler(LocalRAGError)
    async def on_error(request: Request, exc: LocalRAGError) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", uuid.uuid4().hex)
        log_event("error", trace_id, code=exc.code, detail=str(exc))
        body = ErrorResponse(error=exc.code, detail=str(exc), trace_id=trace_id)
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.post("/ask", response_model=AskResponse, responses=ERRORS)
    def ask(req: AskRequest, request: Request) -> AskResponse:
        services: Services = request.app.state.services
        trace_id = request.state.trace_id = uuid.uuid4().hex
        start = time.perf_counter()
        log_event("request", trace_id, k=req.k, question_charts=len(req.question))

        if services.retriever is None:
            raise EmptyCorpusError("no index loaded - run `python -m localrag.ingest`")
        try:
            docs = services.retriever.invoke(req.question)[: req.k]
        except Exception as e:
            raise RetrievalError(f"{type(e).__name__}: {e}") from e
        retrieve_ms = elapsed_ms(start)
        log_event("retrieve", trace_id, ms=retrieve_ms, chunk_ids=[d.id for d in docs])
        sources = [to_source(d) for d in docs]

        gen_start = time.perf_counter()
        try:
            reply = generate_with_retry(
                services.llm, pipeline.build_prompt(req.question, docs), trace_id
            )
        except GenerationTimeout:
            if not docs:
                raise    # nothing to degrade to
            generate_ms = elapsed_ms(gen_start)
            log_event("fallback", trace_id, reason="generation_timeout", ms=generate_ms)
            return AskResponse(
                answer="",
                abstained=False,
                sources=sources,
                timings_ms=Timings(
                    retrieve_ms=retrieve_ms, generate_ms=generate_ms, total_ms=elapsed_ms(start)
                ),
                tokens=TokenUsage(),
                degraded=True,
                trace_id=trace_id,
            )
        generate_ms = elapsed_ms(gen_start)

        usage = reply.usage_metadata or {}
        tokens = TokenUsage(
            input=usage.get("input_tokens", 0), output=usage.get("output_tokens", 0)
        )
        abstained = pipeline.is_abstention(reply.content)
        log_event(
            "generate", trace_id, ms=generate_ms, input_tokens=tokens.input,
            output_tokens=tokens.output, abstained=abstained,
        )
        total_ms = elapsed_ms(start)
        log_event("response", trace_id, total_ms=total_ms, degraded=False)
        return AskResponse(
            answer=reply.content,
            abstained=abstained,
            sources=sources,
            timings_ms=Timings(retrieve_ms=retrieve_ms, generate_ms=generate_ms, total_ms=total_ms),
            tokens=tokens,
            trace_id=trace_id,
        )

    @app.get("/health", response_model=Health, responses={503: {"model": Health}})
    def health(request: Request) -> JSONResponse:
        """Pings Ollama and counts the collection on every call, so it can actually fail."""
        services: Services = request.app.state.services
        reachable, model_ok = services.ollama_status()
        chunks = services.chunk_count()
        ok = reachable and model_ok and chunks > 0 and services.retriever is not None
        body = Health(
            status="ok" if ok else "unhealthy", ollama=reachable, model_available=model_ok,
            chunks=chunks,
        )
        return JSONResponse(status_code=200 if ok else 503, content=body.model_dump())

    return app

app = create_app()



