"""Online recommendation pipeline (framework-free so it can be tested without FastAPI).

retrieve (Redis user features -> ONNX user tower -> FAISS) -> item features (Redis, optional cache)
-> rank (feature assembly -> ONNX DCN-v2) -> re-rank (MMR + genre cap) -> response.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort
import polars as pl

from recsys.features.pit import user_item_cross
from recsys.ranking.transform import transform_features
from recsys.rerank.mmr import mmr


@dataclass
class EngineConfig:
    candidates: int | None = None  # None -> snapshot value
    index: str = "hnsw"  # "flat" for exact (parity tests), "hnsw" for speed
    ef_search: int = 128
    mmr_lambda: float | None = None  # None -> value stored in the snapshot (chosen on val in M4)
    max_per_genre: int | None = -1  # -1 -> value stored in the snapshot
    cache_item_features: bool = False  # M5 optimisation: keep item features in process memory
    threads: int = 1


@dataclass
class Result:
    items: list[int]  # MovieLens movie IDs
    scores: list[float]
    source: str  # "model" | "fallback_popular" | "fallback_genre"
    timings_ms: dict[str, float] = field(default_factory=dict)


class Engine:
    def __init__(self, snapshot: str | Path, redis_client, cfg: EngineConfig | None = None) -> None:
        self.cfg = cfg or EngineConfig()
        snap = Path(snapshot)
        self.meta = json.loads((snap / "meta.json").read_text())
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.cfg.threads
        so.inter_op_num_threads = 1
        self.user_tower = ort.InferenceSession(
            str(snap / "user_tower.onnx"), so, providers=["CPUExecutionProvider"]
        )
        self.ranker = ort.InferenceSession(
            str(snap / "ranker.onnx"), so, providers=["CPUExecutionProvider"]
        )
        self.item_emb = np.load(snap / "item_emb.npy")
        faiss.omp_set_num_threads(self.cfg.threads)
        self.index = faiss.read_index(str(snap / f"faiss_{self.cfg.index}.index"))
        a = np.load(snap / "arrays.npz")
        self.genres, self.als_item = a["genres"], a["als_item"]
        self.recent_pop, self.genre_pop = a["recent_pop"], a["genre_pop"]
        self.movie_ids = a["movie_ids"]
        self._item_cache = a["item_feats"] if self.cfg.cache_item_features else None
        self.genre_index = {g.lower(): j for j, g in enumerate(self.meta["genre_names"])}
        self.redis = redis_client
        self.cutoff = int(self.meta["cutoff"])
        if self.cfg.candidates is None:
            self.cfg.candidates = int(self.meta.get("candidates", 200))
        if self.cfg.mmr_lambda is None:
            self.cfg.mmr_lambda = float(self.meta.get("mmr_lambda", 1.0))
        if self.cfg.max_per_genre == -1:
            self.cfg.max_per_genre = self.meta.get("max_per_genre")
        self.K = self.cfg.candidates

    # ---------------- stages ----------------
    def fetch_user(self, user_id: int) -> dict | None:
        h = self.redis.hgetall(f"u:{user_id}")
        if not h:
            return None
        g = lambda k, dt: np.frombuffer(h[k], dtype=dt)  # noqa: E731
        return {
            "idx": int(h[b"idx"]),
            "hist": g(b"hist", np.int32),
            "dense": g(b"dense", np.float32),
            "has_pos": h[b"has_pos"] == b"1",
            "ufeat": g(b"ufeat", np.float32),
            "gshare": g(b"gshare", np.float32),
            "glast": g(b"glast", np.int64),
            "als": g(b"als", np.float32) if h[b"als"] else None,
            "seen": g(b"seen", np.int32),
        }

    def retrieve(self, u: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Top-K unseen items: (items, retrieval scores, user embedding). Exact-match of offline logic."""
        seen = u["seen"]
        if not u["has_pos"]:
            mask = ~np.isin(self.recent_pop, seen)
            items = self.recent_pop[mask][: self.K]
            return items.astype(np.int64), np.full(len(items), np.nan, np.float32), None
        ue = self.user_tower.run(
            None,
            {
                "users": np.array([u["idx"] + 1], dtype=np.int64),
                "hist": u["hist"][None, :].astype(np.int64),
                "dense": u["dense"][None, :],
            },
        )[0]
        k = self.K + len(seen)
        if self.cfg.index == "hnsw":
            self.index.hnsw.efSearch = max(self.cfg.ef_search, k)
        scores, ids = self.index.search(ue.astype(np.float32), k)
        ids, scores = ids[0], scores[0]
        keep = (ids >= 0) & ~np.isin(ids, seen)
        ids, scores = ids[keep][: self.K], scores[keep][: self.K]
        # stable tie-break by item index to match offline exact top-k
        order = np.lexsort((ids, -scores))
        return ids[order].astype(np.int64), scores[order].astype(np.float32), ue[0]

    def fetch_item_features(self, items: np.ndarray) -> np.ndarray:
        if self._item_cache is not None:
            return self._item_cache[items]
        vals = self.redis.mget([f"i:{i}" for i in items])
        return np.stack([np.frombuffer(v, dtype=np.float32) for v in vals])

    def rank(
        self, u: dict, items: np.ndarray, rscores: np.ndarray, ifeat: np.ndarray
    ) -> np.ndarray:
        n = len(items)
        cols: dict[str, np.ndarray] = {
            "retrieval_score": rscores,
            "retrieval_rank": np.arange(n, dtype=np.float32),
        }
        cols["als_score"] = (
            (self.als_item[items] @ u["als"]).astype(np.float32)
            if u["als"] is not None
            else np.full(n, np.nan, np.float32)
        )
        for j, f in enumerate(self.meta["u_feats"]):
            cols[f] = np.full(n, u["ufeat"][j], np.float32)
        for j, f in enumerate(self.meta["i_feats"]):
            cols[f] = ifeat[:, j]
        users = np.zeros(n, dtype=np.int64)
        cols |= user_item_cross(
            users, items, u["gshare"][None, :], u["glast"][None, :], self.genres, self.cutoff
        )
        x = transform_features(cols, self.meta["features"])
        return self.ranker.run(None, {"x": x})[0].astype(np.float32)

    def rerank(
        self, items: np.ndarray, scores: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        order = np.lexsort((items, -scores))  # same ordering offline uses before MMR
        items, scores = items[order], scores[order]
        out = mmr(
            items,
            scores,
            self.item_emb,
            k,
            self.cfg.mmr_lambda,
            self.genres,
            self.cfg.max_per_genre,
        )
        pos = {it: i for i, it in enumerate(items)}
        return out, scores[[pos[i] for i in out]]

    # ---------------- entry points ----------------
    def fallback(self, k: int, genres: list[str] | None = None) -> Result:
        idx = [self.genre_index[g.lower()] for g in genres or [] if g.lower() in self.genre_index]
        if idx:
            pools = [self.genre_pop[j] for j in idx]
            merged = list(dict.fromkeys(x for tup in zip(*pools, strict=False) for x in tup))[:k]
            return Result(self.movie_ids[merged].tolist(), [0.0] * len(merged), "fallback_genre")
        top = self.recent_pop[:k]
        return Result(self.movie_ids[top].tolist(), [0.0] * len(top), "fallback_popular")

    def recommend(self, user_id: int, k: int = 10, genres: list[str] | None = None) -> Result:
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        u = self.fetch_user(user_id)
        t["user_features"] = time.perf_counter() - t0
        if u is None:
            r = self.fallback(k, genres)
            r.timings_ms = {s: v * 1000 for s, v in t.items()}
            return r
        t1 = time.perf_counter()
        items, rscores, _ = self.retrieve(u)
        t2 = time.perf_counter()
        ifeat = self.fetch_item_features(items)
        t3 = time.perf_counter()
        scores = self.rank(u, items, rscores, ifeat)
        t4 = time.perf_counter()
        top, top_scores = self.rerank(items, scores, k)
        t5 = time.perf_counter()
        t |= {"retrieval": t2 - t1, "item_features": t3 - t2, "ranking": t4 - t3, "rerank": t5 - t4}
        return Result(
            self.movie_ids[top].tolist(),
            [float(s) for s in top_scores],
            "model",
            {s: v * 1000 for s, v in t.items()},
        )


def movie_to_item(snapshot: str | Path) -> dict[int, int]:
    ids = np.load(Path(snapshot) / "arrays.npz")["movie_ids"]
    return {int(m): i for i, m in enumerate(ids)}


def user_id_map(snapshot: str | Path) -> pl.DataFrame:
    return pl.read_parquet(Path(snapshot) / "users_map.parquet")
