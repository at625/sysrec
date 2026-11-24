"""FastAPI service: GET /recommend, GET /health, GET /metrics (Prometheus).

Config via env vars: SNAPSHOT_DIR, REDIS_URL, INDEX (hnsw|flat), CACHE_ITEM_FEATURES (0|1),
MMR_LAMBDA, MAX_PER_GENRE, CANDIDATES.
The pipeline is CPU-bound, so requests run in a thread pool (FastAPI does this for sync endpoints)
and the event loop stays free for /health and /metrics.
"""

from __future__ import annotations

import os
import time

import redis
from fastapi import FastAPI, HTTPException, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from recsys.serving.engine import Engine, EngineConfig

BUCKETS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
STAGE_LATENCY = Histogram(
    "recsys_stage_latency_seconds", "Latency per pipeline stage", ["stage"], buckets=BUCKETS
)
REQUEST_LATENCY = Histogram(
    "recsys_request_latency_seconds", "End-to-end /recommend latency", buckets=BUCKETS
)
REQUESTS = Counter("recsys_requests_total", "Requests by result source", ["source"])


def _env_engine_config() -> EngineConfig:
    mpg = os.environ.get("MAX_PER_GENRE", "3")
    return EngineConfig(
        candidates=int(os.environ.get("CANDIDATES", "200")),
        index=os.environ.get("INDEX", "hnsw"),
        mmr_lambda=float(os.environ.get("MMR_LAMBDA", "0.8")),
        max_per_genre=None if mpg in ("", "none", "None") else int(mpg),
        cache_item_features=os.environ.get("CACHE_ITEM_FEATURES", "0") == "1",
    )


app = FastAPI(title="sysrec")
_engine: Engine | None = None


def engine() -> Engine:
    global _engine
    if _engine is None:
        r = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
        _engine = Engine(
            os.environ.get("SNAPSHOT_DIR", "artifacts/serving"), r, _env_engine_config()
        )
    return _engine


@app.on_event("startup")
def _startup() -> None:
    engine()


@app.get("/health")
def health() -> dict:
    e = engine()
    try:
        ok = bool(e.redis.ping()) and e.redis.get("snapshot:cutoff") is not None
    except redis.RedisError:
        ok = False
    if not ok:
        raise HTTPException(503, "redis unavailable or snapshot not loaded")
    return {
        "status": "ok",
        "snapshot_cutoff": e.cutoff,
        "index": e.cfg.index,
        "cache_item_features": e.cfg.cache_item_features,
    }


@app.get("/recommend")
def recommend(user_id: int, k: int = Query(10, ge=1, le=100), genres: str | None = None) -> dict:
    t0 = time.perf_counter()
    res = engine().recommend(user_id, k, genres.split(",") if genres else None)
    REQUEST_LATENCY.observe(time.perf_counter() - t0)
    REQUESTS.labels(res.source).inc()
    for stage, ms in res.timings_ms.items():
        STAGE_LATENCY.labels(stage).observe(ms / 1000)
    return {
        "user_id": user_id,
        "source": res.source,
        "items": [{"movie_id": m, "score": s} for m, s in zip(res.items, res.scores, strict=True)],
        "timings_ms": res.timings_ms,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
