"""The API boudary as types. 

FastAPI validates requests against them(422) and documents both at /docs.
 """

from typing import Literal

from pydantic import BaseModel, Field

MAX_K = 20  #the hybrid stage yields at most 10 dense + 10 BM25 candidates

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=MAX_K)

class Source(BaseModel):
    chunk_id: str
    section: str
    source_doc: str | None
    text: str

class Timings(BaseModel):
    retrieve_ms: float
    generate_ms: float
    total_ms: float

class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0

class AskResponse(BaseModel):
    answer: str
    abstained: bool
    sources: list[Source]
    timings_ms: Timings
    tokens: TokenUsage
    degraded: bool = False
    trace_id: str

class ErrorResponse(BaseModel):
    error: str
    detail: str
    trace_id: str

class Health(BaseModel):
    status: Literal["ok", "unhealthy"]
    ollama: bool
    model_available: bool
    chunks: int



