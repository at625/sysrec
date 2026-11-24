"""Integration tests against a running service (docker compose up, or uvicorn + redis-server).

  SERVICE_URL=http://localhost:8000 pytest -m integration
The parity test expects the service to run with INDEX=flat (exact retrieval).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration
URL = os.environ.get("SERVICE_URL")
SNAP = Path(os.environ.get("SNAPSHOT_DIR", "artifacts/serving"))


@pytest.fixture(scope="module")
def client():
    if not URL:
        pytest.skip("SERVICE_URL not set")
    import httpx

    with httpx.Client(base_url=URL, timeout=30) as c:
        yield c


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_unknown_user_gets_popular_fallback(client) -> None:
    body = client.get("/recommend", params={"user_id": 999_999_999, "k": 5}).json()
    assert body["source"] == "fallback_popular" and len(body["items"]) == 5


def test_unknown_user_with_genres(client) -> None:
    body = client.get(
        "/recommend", params={"user_id": 999_999_999, "k": 5, "genres": "Comedy,Horror"}
    ).json()
    assert body["source"] == "fallback_genre" and len(body["items"]) == 5


def test_known_user_and_metrics(client) -> None:
    ref = np.load(SNAP / "offline_reference.npz")
    body = client.get("/recommend", params={"user_id": int(ref["user_ids"][0]), "k": 10}).json()
    assert body["source"] == "model" and len(body["items"]) == 10
    assert set(body["timings_ms"]) >= {"retrieval", "ranking", "rerank"}
    assert "recsys_stage_latency_seconds" in client.get("/metrics").text


def test_online_matches_offline(client) -> None:
    if client.get("/health").json()["index"] != "flat":
        pytest.skip("parity is defined for exact retrieval (INDEX=flat)")
    ref = np.load(SNAP / "offline_reference.npz")
    same = 0
    n = min(50, len(ref["user_ids"]))
    for uid, want in zip(ref["user_ids"][:n], ref["movie_ids"][:n], strict=True):
        got = [
            it["movie_id"]
            for it in client.get("/recommend", params={"user_id": int(uid), "k": 10}).json()[
                "items"
            ]
        ]
        same += got == want.tolist()
    assert same / n >= 0.98
