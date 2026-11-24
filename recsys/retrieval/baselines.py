"""Baselines: global popularity, recent popularity, and implicit-feedback ALS."""

from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp

from recsys.eval.harness import FitContext, mask_seen, topk_from_scores

DAY = 86_400


class PopularityRecommender:
    """Ranks items by number of positive interactions in the history (optionally a recent window)."""

    def __init__(self, days: int | None = None, name: str | None = None) -> None:
        self.days = days
        self.name = name or ("popular" if days is None else f"recent_popular_{days}d")

    def fit(self, ctx: FitContext) -> None:
        pos = ctx.positives
        if self.days is not None:
            pos = pos.filter(pl.col("ts") >= ctx.cutoff - self.days * DAY)
        counts = np.bincount(pos["item_idx"].to_numpy(), minlength=ctx.n_items).astype(np.float32)
        # Tie-break by all-time popularity so the tail of a short window is still sensible.
        self.scores = counts + 1e-6 * (
            ctx.item_popularity / max(ctx.item_popularity.max(), 1)
        ).astype(np.float32)
        self.ctx = ctx

    def recommend(self, users: np.ndarray, k: int, batch: int = 1024) -> np.ndarray:
        out = []
        for s in range(0, len(users), batch):
            u = users[s : s + batch]
            sc = np.repeat(self.scores[None, :], len(u), axis=0)
            out.append(topk_from_scores(mask_seen(sc, u, self.ctx.seen), k))
        return np.concatenate(out) if out else np.zeros((0, k), dtype=np.int64)


class ALSRecommender:
    """Implicit ALS (Hu, Koren, Volinsky 2008) via the `implicit` library on positive interactions.

    Confidence = 1 + alpha * 1[positive]. Users without positives in the (optionally recent-only)
    training history fall back to `fallback` (recent popularity), as do cold users.
    """

    name = "als"

    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.05,
        alpha: float = 10.0,
        iterations: int = 15,
        fallback_days: int = 30,
        seed: int = 0,
        since_days: int | None = None,
    ) -> None:
        self.params = dict(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=iterations,
            since_days=since_days,
        )
        self.since_days = since_days
        self.fallback = PopularityRecommender(days=fallback_days)
        self.seed = seed

    def fit(self, ctx: FitContext) -> None:
        from implicit.als import AlternatingLeastSquares

        pos = ctx.positives
        if self.since_days is not None:  # train only on recent positives (tastes and catalog drift)
            pos = pos.filter(pl.col("ts") >= ctx.cutoff - self.since_days * DAY)
        m = sp.csr_matrix(
            (
                np.ones(pos.height, dtype=np.float32),
                (pos["user_idx"].to_numpy(), pos["item_idx"].to_numpy()),
            ),
            shape=(ctx.n_users, ctx.n_items),
        )
        m.data[:] = 1.0
        self.has_pos = np.diff(m.indptr) > 0
        self.model = AlternatingLeastSquares(
            factors=self.params["factors"],
            regularization=self.params["regularization"],
            alpha=self.params["alpha"],
            iterations=self.params["iterations"],
            random_state=self.seed,
            calculate_training_loss=False,
        )
        self.model.fit(m, show_progress=False)
        self.user_f = np.asarray(self.model.user_factors, dtype=np.float32)
        self.item_f = np.asarray(self.model.item_factors, dtype=np.float32)
        self.fallback.fit(ctx)
        self.ctx = ctx

    def recommend(self, users: np.ndarray, k: int, batch: int = 1024) -> np.ndarray:
        out = np.zeros((len(users), k), dtype=np.int64)
        warm = self.has_pos[users]
        if (~warm).any():
            out[~warm] = self.fallback.recommend(users[~warm], k)
        wu = users[warm]
        res = []
        for s in range(0, len(wu), batch):
            u = wu[s : s + batch]
            sc = self.user_f[u] @ self.item_f.T
            res.append(topk_from_scores(mask_seen(sc, u, self.ctx.seen), k))
        if res:
            out[warm] = np.concatenate(res)
        return out
