# Serving image: API only (no torch). Build context = repo root; the snapshot is mounted at runtime.
FROM python:3.11-slim
ARG HTTPS_PROXY=""
# Optional extra CA certs (e.g. a corporate/sandbox proxy); docker/ca/ is empty by default.
COPY docker/ca/ /usr/local/share/ca-certificates/
RUN update-ca-certificates >/dev/null 2>&1 || true
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-serving.txt .
RUN HTTPS_PROXY=$HTTPS_PROXY pip install --no-cache-dir -r requirements-serving.txt
COPY recsys/ recsys/
# One BLAS/OpenMP thread per worker: parallelism comes from worker processes, not threads.
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ENV SNAPSHOT_DIR=/snapshot REDIS_URL=redis://redis:6379/0 INDEX=hnsw CACHE_ITEM_FEATURES=1
EXPOSE 8000
CMD ["uvicorn", "recsys.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
