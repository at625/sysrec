"""Diversity re-ranking: Maximal Marginal Relevance (Carbonell & Goldstein 1998) + a genre cap rule."""

from __future__ import annotations

import numpy as np


def mmr(
    cand_items: np.ndarray,
    relevance: np.ndarray,
    item_emb: np.ndarray,
    k: int,
    lam: float,
    item_genres: np.ndarray | None = None,
    max_per_genre: int | None = None,
) -> np.ndarray:
    """Greedy MMR over one user's candidates.

    score(i) = lam * rel(i) - (1 - lam) * max_{j in selected} cos(i, j)
    `relevance` is min-max normalised to [0, 1] so lam trades off comparable quantities.
    With `max_per_genre`, an item is skipped if adding it would put any of its genres above the cap
    (if every remaining item violates the cap, the cap is relaxed for that slot).
    lam = 1 returns the relevance order (subject to the genre cap).
    """
    n = len(cand_items)
    k = min(k, n)
    rel = relevance.astype(np.float64)
    span = rel.max() - rel.min()
    rel = (rel - rel.min()) / span if span > 0 else np.zeros_like(rel)
    emb = item_emb[cand_items]
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    sim = emb @ emb.T
    selected: list[int] = []
    max_sim = np.full(n, -np.inf)
    available = np.ones(n, dtype=bool)
    genre_counts = np.zeros(item_genres.shape[1]) if item_genres is not None else None
    for _ in range(k):
        score = lam * rel - (1 - lam) * np.where(np.isfinite(max_sim), max_sim, 0.0)
        score = np.where(available, score, -np.inf)
        if max_per_genre is not None and genre_counts is not None:
            g = item_genres[cand_items] > 0
            violates = (g & (genre_counts[None, :] >= max_per_genre)).any(1)
            capped = np.where(violates, -np.inf, score)
            if np.isfinite(capped).any():
                score = capped
        j = int(np.argmax(score))
        selected.append(j)
        available[j] = False
        max_sim = np.maximum(max_sim, sim[j])
        if genre_counts is not None:
            genre_counts += item_genres[cand_items[j]] > 0
    return cand_items[np.array(selected, dtype=np.int64)]


def intra_list_diversity(lists: np.ndarray, item_emb: np.ndarray) -> float:
    """Mean over users of mean pairwise (1 - cosine similarity) within each list."""
    e = item_emb / np.maximum(np.linalg.norm(item_emb, axis=1, keepdims=True), 1e-12)
    vals = []
    for row in lists:
        x = e[row]
        s = x @ x.T
        n = len(row)
        vals.append(1 - (s.sum() - np.trace(s)) / (n * (n - 1)))
    return float(np.mean(vals))


def genre_coverage(lists: np.ndarray, item_genres: np.ndarray) -> float:
    """Mean over users of (#distinct genres in the list / #genres)."""
    return float(np.mean([(item_genres[row].sum(0) > 0).mean() for row in lists]))
