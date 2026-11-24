"""Loading processed Parquet, with optional stable user subsampling."""

from __future__ import annotations

import polars as pl

from recsys.config import Config


def user_in_sample(col: str, fraction: float) -> pl.Expr:
    """Stable hash-based user subsample: the same users are kept for every model and run."""
    if fraction >= 1.0:
        return pl.lit(True)
    return (pl.col(col).cast(pl.UInt64) * 2654435761 % 4294967296 % 10_000) < int(fraction * 10_000)


def scan_ratings(cfg: Config) -> pl.LazyFrame:
    lf = pl.scan_parquet(cfg.processed / "ratings.parquet")
    return lf.filter(user_in_sample("user_idx", cfg.data.user_fraction))


def load_items(cfg: Config) -> pl.DataFrame:
    return pl.read_parquet(cfg.processed / "items.parquet")


def n_items(cfg: Config) -> int:
    return pl.scan_parquet(cfg.processed / "items.parquet").select(pl.len()).collect().item()


def n_users(cfg: Config) -> int:
    return pl.scan_parquet(cfg.processed / "users.parquet").select(pl.len()).collect().item()


def scan_tags(cfg: Config) -> pl.LazyFrame:
    return pl.scan_parquet(cfg.processed / "tags.parquet")
