"""Ranking examples: retrieval candidates + point-in-time features + labels from the window after.

A *snapshot* at cutoff c:
  1. retrieval models are fit on history < c,
  2. each user gets top-K candidates (hard negatives come from retrieval, not random items),
  3. features are computed from history < c only,
  4. labels come from the label window [c, c + L): label_pos = rated >= threshold in the window,
     label_rating = the rating if the item was rated in the window (else null).
The ranker for window W is trained on the snapshot whose label window is the period just before W,
so ranker training never sees W.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from recsys.eval.harness import FitContext
from recsys.features.pit import item_features, user_features, user_genre_profile, user_item_cross

FEATURES = [
    "retrieval_score",
    "retrieval_rank",
    "als_score",
    "u_n_ratings",
    "u_n_pos",
    "u_mean_rating",
    "u_days_since_last",
    "u_days_since_first",
    "u_n_last30d",
    "i_n_ratings",
    "i_n_pos",
    "i_pos_7d",
    "i_pos_30d",
    "i_pos_365d",
    "i_avg_rating_smoothed",
    "i_days_since_first_rating",
    "i_release_age_years",
    "x_genre_affinity_mean",
    "x_genre_affinity_max",
    "x_days_since_similar",
]


@dataclass
class Candidates:
    users: np.ndarray  # (U,)
    items: np.ndarray  # (U, K)
    scores: np.ndarray  # (U, K) retrieval score (nan for fallback rows)


def retrieve_candidates(retriever, users: np.ndarray, k: int) -> Candidates:
    """Top-k candidates from a fitted TwoTowerRecommender, with their scores."""
    items = retriever.recommend(users, k)
    scores = np.full(items.shape, np.nan, dtype=np.float32)
    warm = retriever.has_pos[users]
    if warm.any():
        ue = retriever.user_embeddings(users[warm])
        scores[warm] = np.einsum("ud,ukd->uk", ue, retriever.item_emb[items[warm]])
    return Candidates(users, items, scores)


def build_frame(
    ctx: FitContext,
    cands: Candidates,
    item_genres: np.ndarray,
    als=None,
    labels: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per (user, candidate) with FEATURES and (if `labels` given) label columns."""
    cfg = ctx.cfg
    thr = cfg.data.positive_threshold
    U, K = cands.items.shape
    users = np.repeat(cands.users, K)
    items = cands.items.reshape(-1)
    cols = {
        "user_idx": users.astype(np.int32),
        "item_idx": items.astype(np.int32),
        "retrieval_score": cands.scores.reshape(-1),
        "retrieval_rank": np.tile(np.arange(K, dtype=np.float32), U),
    }
    if als is not None:
        s = np.einsum("rd,rd->r", als.user_f[users], als.item_f[items]).astype(np.float32)
        cols["als_score"] = np.where(als.has_pos[users], s, np.nan).astype(np.float32)
    else:
        cols["als_score"] = np.full(len(users), np.nan, dtype=np.float32)
    share, last = user_genre_profile(ctx.history, item_genres, ctx.n_users, ctx.cutoff, thr)
    cols |= user_item_cross(users, items, share, last, item_genres, ctx.cutoff)
    df = pl.DataFrame(cols)
    from recsys.data.io import load_items

    df = df.join(user_features(ctx.history, ctx.cutoff, thr), on="user_idx", how="left")
    df = df.join(
        item_features(ctx.history, load_items(cfg), ctx.cutoff, thr), on="item_idx", how="left"
    )
    if labels is not None:
        df = df.join(labels, on=["user_idx", "item_idx"], how="left").with_columns(
            pl.col("label_pos").fill_null(0).cast(pl.Int8)
        )
    return df.with_columns(pl.col(FEATURES).cast(pl.Float32))


def window_labels(window_df: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """(user, item) -> label_pos (max over the window) and label_rating (last rating in the window)."""
    return (
        window_df.sort("ts")
        .group_by(["user_idx", "item_idx"])
        .agg(
            (pl.col("rating") >= threshold).any().cast(pl.Int8).alias("label_pos"),
            pl.col("rating").last().alias("label_rating"),
        )
        .with_columns(pl.col("user_idx", "item_idx").cast(pl.Int32))
    )
