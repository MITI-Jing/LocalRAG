"""Error taxonomy: every failure the API reports on purpose, and its HTTP status.

The rule: a slow model degrades (timeout -> 200 with the retrieved passages), while a missing
daemon or model fails loudly (503 after one retry) - degrading there would hide an ops problem.
"""

class LocalRAGError(Exception):
    status_code = 500
    code = "internal_error"

class OllamaUnavailable(LocalRAGError):
    """Daemon unreachable, connection dropped mid-reply, or model not pulled."""

    status_code = 503
    code = "ollama_unavailable"

class GenerationTimeout(LocalRAGError):
    """No reply within the budget. /ask degrades to passages when it has any to return."""

    status_code = 504
    code = "generation_timeout"

class RetrievalError(LocalRAGError):
    status_code = 500
    code = "retrieval_error"

class EmptyCorpusError(LocalRAGError):
    """No index, or an index with no chunks - nothing has been ingested."""

    status_code = 409
    code = "empty_corpus"
