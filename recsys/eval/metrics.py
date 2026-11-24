"""Full-catalog top-K ranking metrics with binary relevance.

Conventions (documented because papers differ):
  Recall@K  = |top-K ∩ T| / |T|                       (T = user's positive targets in the window)
  HitRate@K = 1 if |top-K ∩ T| > 0 else 0
  NDCG@K    = DCG@K / IDCG@K, DCG = sum_{rank r hit} 1/log2(r+1), IDCG uses min(|T|, K) hits
  MRR@K     = 1 / rank of the first hit within top-K, else 0
All are averaged over users with |T| >= 1. Ranks are 1-based.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def hits_matrix(topk: np.ndarray, targets: Sequence[np.ndarray]) -> np.ndarray:
    """Boolean (n_users, K) matrix: whether the item at each rank is a target."""
    out = np.zeros(topk.shape, dtype=bool)
    for u, t in enumerate(targets):
        if len(t):
            out[u] = np.isin(topk[u], t)
    return out


def ranking_metrics(
    topk: np.ndarray, targets: Sequence[np.ndarray], ks: Sequence[int]
) -> dict[str, float]:
    """Mean Recall/NDCG/HitRate/MRR at each K. `topk` must have >= max(ks) columns."""
    kmax = max(ks)
    assert topk.shape[1] >= kmax, (topk.shape, kmax)
    hits = hits_matrix(topk[:, :kmax], targets)
    n_t = np.array([len(t) for t in targets], dtype=np.float64)
    assert (n_t > 0).all(), "every evaluated user needs >= 1 target"
    discounts = 1.0 / np.log2(np.arange(2, kmax + 2))
    first_hit = np.where(hits.any(1), hits.argmax(1), kmax)  # 0-based rank, kmax = no hit
    out: dict[str, float] = {}
    for k in ks:
        h = hits[:, :k]
        n_hit = h.sum(1)
        dcg = (h * discounts[:k]).sum(1)
        idcg = np.cumsum(discounts[:k])[np.minimum(n_t, k).astype(int) - 1]
        out[f"recall@{k}"] = float(np.mean(n_hit / n_t))
        out[f"ndcg@{k}"] = float(np.mean(dcg / idcg))
        out[f"hitrate@{k}"] = float(np.mean(n_hit > 0))
        out[f"mrr@{k}"] = float(np.mean(np.where(first_hit < k, 1.0 / (first_hit + 1), 0.0)))
    return out


def per_user_metrics(
    topk: np.ndarray, targets: Sequence[np.ndarray], k: int
) -> dict[str, np.ndarray]:
    """Per-user Recall@k and NDCG@k (for bootstrap CIs)."""
    hits = hits_matrix(topk[:, :k], targets)
    n_t = np.array([len(t) for t in targets], dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    idcg = np.cumsum(discounts)[np.minimum(n_t, k).astype(int) - 1]
    return {
        f"recall@{k}": hits.sum(1) / n_t,
        f"ndcg@{k}": (hits * discounts).sum(1) / idcg,
        f"hitrate@{k}": (hits.sum(1) > 0).astype(float),
    }


def catalog_coverage(topk: np.ndarray, n_items: int, k: int) -> float:
    """Fraction of the catalog that appears in at least one user's top-k."""
    return float(len(np.unique(topk[:, :k])) / n_items)


def popularity_bias(topk: np.ndarray, item_popularity: np.ndarray, k: int) -> float:
    """Mean popularity (history interaction count) of recommended items in the top-k."""
    return float(item_popularity[topk[:, :k]].mean())
