"""Point-in-time feature computation for the ranker.

Every function takes `history` (interactions with ts < cutoff) and `cutoff`, and only ever reads
`history`. A feature snapshot "as of cutoff" therefore cannot see the label window that starts at
cutoff. `tests/test_features.py` checks this by perturbing all data at/after the cutoff.
"""

from __future__ import annotations

import numpy as np
import polars as pl

DAY = 86_400
POP_WINDOWS = (7, 30, 365)


def user_features(history: pl.DataFrame, cutoff: int, threshold: float) -> pl.DataFrame:
    """Activity stats per user as of `cutoff`."""
    assert history.height == 0 or history["ts"].max() < cutoff
    return (
        history.group_by("user_idx")
        .agg(
            pl.len().alias("u_n_ratings"),
            (pl.col("rating") >= threshold).sum().alias("u_n_pos"),
            pl.col("rating").mean().alias("u_mean_rating"),
            ((cutoff - pl.col("ts").max()) / DAY).alias("u_days_since_last"),
            ((cutoff - pl.col("ts").min()) / DAY).alias("u_days_since_first"),
            (pl.col("ts") >= cutoff - 30 * DAY).sum().alias("u_n_last30d"),
        )
        .with_columns(pl.col("user_idx").cast(pl.Int32))
    )


def item_features(
    history: pl.DataFrame,
    items: pl.DataFrame,
    cutoff: int,
    threshold: float,
    prior_weight: float = 10.0,
) -> pl.DataFrame:
    """Per-item popularity (several windows), smoothed average rating, and age, as of `cutoff`."""
    assert history.height == 0 or history["ts"].max() < cutoff
    global_mean = float(history["rating"].mean()) if history.height else 3.5
    pos = pl.col("rating") >= threshold
    agg = history.group_by("item_idx").agg(
        pl.len().alias("i_n_ratings"),
        pos.sum().alias("i_n_pos"),
        *[
            (pos & (pl.col("ts") >= cutoff - d * DAY)).sum().alias(f"i_pos_{d}d")
            for d in POP_WINDOWS
        ],
        pl.col("rating").sum().alias("_rsum"),
        ((cutoff - pl.col("ts").min()) / DAY).alias("i_days_since_first_rating"),
    )
    year = pl.from_epoch(pl.lit(cutoff), time_unit="s").dt.year()
    out = (
        items.select("item_idx", "year")
        .join(agg, on="item_idx", how="left")
        .with_columns(
            pl.col("i_n_ratings", "i_n_pos", *[f"i_pos_{d}d" for d in POP_WINDOWS]).fill_null(0),
            pl.col("i_days_since_first_rating").fill_null(-1.0),
        )
        .with_columns(
            (
                (pl.col("_rsum").fill_null(0) + prior_weight * global_mean)
                / (pl.col("i_n_ratings") + prior_weight)
            ).alias("i_avg_rating_smoothed"),
            pl.when(pl.col("year") > 0)
            .then(year - pl.col("year"))
            .otherwise(None)
            .cast(pl.Float32)
            .alias("i_release_age_years"),
        )
        .drop("_rsum", "year")
    )
    return out.with_columns(pl.col("item_idx").cast(pl.Int32))


def user_genre_profile(
    history: pl.DataFrame, item_genres: np.ndarray, n_users: int, cutoff: int, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """(n_users, n_genres) share of each user's positives per genre, and last positive ts per genre.

    Last-ts is -1 when the user never liked that genre before `cutoff`.
    """
    assert history.height == 0 or history["ts"].max() < cutoff
    pos = history.filter(pl.col("rating") >= threshold)
    u = pos["user_idx"].to_numpy()
    g = item_genres[pos["item_idx"].to_numpy()]
    share = np.zeros((n_users, item_genres.shape[1]), dtype=np.float32)
    np.add.at(share, u, g)
    tot = share.sum(1, keepdims=True)
    share = np.divide(share, tot, out=np.zeros_like(share), where=tot > 0)
    last = np.full((n_users, item_genres.shape[1]), -1, dtype=np.int64)
    ts = pos["ts"].to_numpy()
    for j in range(item_genres.shape[1]):
        m = g[:, j] > 0
        np.maximum.at(last[:, j], u[m], ts[m])
    return share, last


def user_item_cross(
    users: np.ndarray,
    items: np.ndarray,
    genre_share: np.ndarray,
    genre_last: np.ndarray,
    item_genres: np.ndarray,
    cutoff: int,
) -> dict[str, np.ndarray]:
    """Pairwise features for candidate rows (user, item)."""
    ig = item_genres[items]  # (R, G)
    n_g = np.maximum(ig.sum(1), 1)
    aff = genre_share[users] * ig
    last = np.where(ig > 0, genre_last[users], -1).max(1)
    days_since_similar = np.where(last >= 0, (cutoff - last) / DAY, -1.0)
    return {
        "x_genre_affinity_mean": (aff.sum(1) / n_g).astype(np.float32),
        "x_genre_affinity_max": aff.max(1).astype(np.float32),
        "x_days_since_similar": days_since_similar.astype(np.float32),
    }
