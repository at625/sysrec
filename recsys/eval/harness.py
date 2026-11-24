"""Offline evaluation harness: fit on history before a window, rank the full catalog, score the window.

Every recommender implements the `Recommender` protocol:
  fit(ctx)              learn from ctx.history (strictly before ctx.cutoff)
  recommend(users, k)   top-k item indices per user, already excluding ctx.seen[user]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import polars as pl
import scipy.sparse as sp

from recsys.config import Config
from recsys.data.io import n_items as count_items
from recsys.data.io import n_users as count_users
from recsys.data.split import Window, eval_targets, history_before
from recsys.eval.metrics import catalog_coverage, popularity_bias, ranking_metrics

KS = (10, 50, 100, 500)


@dataclass
class FitContext:
    """Everything a model may see when it is evaluated on a window that starts at `cutoff`."""

    cfg: Config
    cutoff: int
    history: pl.DataFrame  # user_idx, item_idx, rating, ts; all ts < cutoff
    n_users: int
    n_items: int
    seen: sp.csr_matrix = field(
        repr=False
    )  # (n_users, n_items) 1 where the user interacted before cutoff
    item_popularity: np.ndarray = field(repr=False)  # interaction count before cutoff

    @property
    def positives(self) -> pl.DataFrame:
        return self.history.filter(pl.col("rating") >= self.cfg.data.positive_threshold)

    def warm_users(self) -> np.ndarray:
        return np.flatnonzero(np.diff(self.seen.indptr) > 0)


def build_context(cfg: Config, cutoff: int) -> FitContext:
    hist = history_before(cfg, cutoff).collect()
    nu, ni = count_users(cfg), count_items(cfg)
    u = hist["user_idx"].to_numpy()
    i = hist["item_idx"].to_numpy()
    seen = sp.csr_matrix((np.ones(len(u), dtype=np.float32), (u, i)), shape=(nu, ni))
    seen.data[:] = 1.0  # duplicates summed by constructor; binarize
    pop = np.bincount(i, minlength=ni).astype(np.float64)
    assert hist.height == 0 or hist["ts"].max() < cutoff
    return FitContext(cfg, cutoff, hist, nu, ni, seen, pop)


class Recommender(Protocol):
    name: str

    def fit(self, ctx: FitContext) -> None: ...

    def recommend(self, users: np.ndarray, k: int) -> np.ndarray: ...


def topk_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """Row-wise top-k indices sorted by descending score (ties broken by lower index)."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(scores, part, 1)
    idx = np.lexsort((part, -order), axis=1)
    return np.take_along_axis(part, idx, 1)


def mask_seen(scores: np.ndarray, users: np.ndarray, seen: sp.csr_matrix) -> np.ndarray:
    """Set scores of items each user already interacted with to -inf (in place)."""
    sub = seen[users]
    rows = np.repeat(np.arange(len(users)), np.diff(sub.indptr))
    scores[rows, sub.indices] = -np.inf
    return scores


@dataclass
class EvalResult:
    model: str
    window: str
    metrics: dict[str, float]
    metrics_warm: dict[str, float]
    n_users: int
    n_warm: int
    fit_seconds: float
    recommend_seconds: float


def evaluate(
    model: Recommender,
    ctx: FitContext,
    window: Window,
    ks: tuple[int, ...] = KS,
    fit: bool = True,
    return_topk: bool = False,
) -> EvalResult | tuple[EvalResult, np.ndarray, pl.DataFrame]:
    if window.start != ctx.cutoff:
        raise ValueError("context cutoff must equal window start (no gap, no leakage)")
    ks = tuple(k for k in ks if k <= ctx.n_items)  # tiny synthetic catalogs
    t0 = time.perf_counter()
    if fit:
        model.fit(ctx)
    t1 = time.perf_counter()
    tg = eval_targets(ctx.cfg, window)
    users = tg["user_idx"].to_numpy()
    targets = [np.asarray(t, dtype=np.int64) for t in tg["targets"].to_list()]
    topk = model.recommend(users, max(ks))
    t2 = time.perf_counter()
    m = ranking_metrics(topk, targets, ks)
    for k in ks:
        m[f"coverage@{k}"] = catalog_coverage(topk, ctx.n_items, k)
        m[f"popbias@{k}"] = popularity_bias(topk, ctx.item_popularity, k)
    warm = np.diff(ctx.seen.indptr)[users] > 0
    mw = (
        ranking_metrics(topk[warm], [t for t, w in zip(targets, warm, strict=True) if w], ks)
        if warm.any()
        else {}
    )
    res = EvalResult(model.name, window.name, m, mw, len(users), int(warm.sum()), t1 - t0, t2 - t1)
    if return_topk:
        return res, topk, tg
    return res
