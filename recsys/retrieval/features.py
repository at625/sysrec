"""Static item features and point-in-time user sequences for the two-tower model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from recsys.config import Config
from recsys.data.io import load_items

YEAR_BUCKETS = np.array(
    [0, 1930, 1950, 1960, 1970, 1980, 1990, 1995, 2000, 2005, 2010, 2013, 2016, 2019, 2022]
)


def year_bucket(years: np.ndarray) -> np.ndarray:
    """0 = unknown year; otherwise 1 + index of the bucket."""
    b = np.searchsorted(YEAR_BUCKETS[1:], years, side="right") + 1
    return np.where(years <= 0, 0, b).astype(np.int64)


@dataclass
class ItemFeatures:
    genres: np.ndarray  # (n_items, n_genres) float32 multi-hot
    year_bucket: np.ndarray  # (n_items,) int64
    text: np.ndarray  # (n_items, d_text) float32
    genre_names: list[str]

    @property
    def n_year_buckets(self) -> int:
        return len(YEAR_BUCKETS) + 1


def item_features(cfg: Config, text_emb: np.ndarray) -> ItemFeatures:
    items = load_items(cfg).sort("item_idx")
    names = sorted({g for gs in items["genres"].to_list() for g in gs})
    col = {g: j for j, g in enumerate(names)}
    mh = np.zeros((items.height, len(names)), dtype=np.float32)
    for i, gs in enumerate(items["genres"].to_list()):
        for g in gs:
            mh[i, col[g]] = 1.0
    assert text_emb.shape[0] == items.height
    return ItemFeatures(mh, year_bucket(items["year"].to_numpy()), text_emb, names)


@dataclass
class Sequences:
    """Each user's positive interactions sorted by time, flattened.

    For user u the positives are items[start[u]:start[u+1]] at times ts[...]. A training example at
    flat position j uses only positions < j as history (strictly earlier positives), so the target
    never leaks into its own history.
    """

    items: np.ndarray  # (N,) int64 item_idx
    ts: np.ndarray  # (N,) int64
    users: np.ndarray  # (N,) int64 user of each position
    start: np.ndarray  # (n_users + 1,) offsets
    ratings_mean_before: (
        np.ndarray
    )  # (N,) user's mean rating over ALL ratings strictly before ts[j]
    n_ratings_before: np.ndarray  # (N,) count of all ratings strictly before ts[j]


def build_sequences(history: pl.DataFrame, n_users: int, threshold: float) -> Sequences:
    """Build per-user positive sequences plus point-in-time user stats from `history`."""
    h = history.sort(["user_idx", "ts", "item_idx"])
    # Running stats over all ratings, counting only rows with strictly smaller ts (ties excluded).
    stats = (
        h.group_by(["user_idx", "ts"], maintain_order=True)
        .agg(pl.col("rating").sum().alias("s"), pl.len().alias("c"))
        .with_columns(
            (pl.col("s").cum_sum().over("user_idx") - pl.col("s")).alias("s_before"),
            (pl.col("c").cum_sum().over("user_idx") - pl.col("c")).alias("c_before"),
        )
        .select("user_idx", "ts", "s_before", "c_before")
    )
    pos = h.filter(pl.col("rating") >= threshold).join(stats, on=["user_idx", "ts"], how="left")
    pos = pos.sort(["user_idx", "ts", "item_idx"])
    users = pos["user_idx"].to_numpy().astype(np.int64)
    counts = np.bincount(users, minlength=n_users)
    start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    c = pos["c_before"].to_numpy().astype(np.float32)
    s = pos["s_before"].to_numpy().astype(np.float32)
    mean = np.where(c > 0, s / np.maximum(c, 1), 0.0).astype(np.float32)
    return Sequences(
        pos["item_idx"].to_numpy().astype(np.int64), pos["ts"].to_numpy(), users, start, mean, c
    )


def gather_history(
    seq_items: np.ndarray, user_start: np.ndarray, end: np.ndarray, length: int
) -> np.ndarray:
    """Last `length` items before flat position `end` (exclusive) for each row; -1 = padding."""
    idx = end[:, None] - length + np.arange(length)[None, :]
    valid = idx >= user_start[:, None]
    out = np.where(valid, seq_items[np.clip(idx, 0, max(len(seq_items) - 1, 0))], -1)
    return out
