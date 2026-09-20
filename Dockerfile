# LocalRAG API: CPU-only, index and models baked in at build time, no downloads at runtime

FROM python:3.12-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 ANONYMIZED_TELEMETRY=False

# CPU-only torch first, so sentence-transformers does not pull the mult-GB CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN pip install .

# Build the index and download both models now. Chunk ids are deterministic(chunk-0000...),
# so this index matches the gold ids committed in eval/eval_testset_v1.jsonl.

COPY data ./data
RUN python -m localrag.ingest \
&& python -c "from localrag.pipeline import build_retriever; build_retriever()"

# Everything is on disk - forbid runtime downloads.

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

EXPOSE 8000
CMD ["uvicorn", "localrag.api:app", "--host", "0.0.0.0", "--port", "8000"]


