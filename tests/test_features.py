"""M3: point-in-time correctness. Features as of a cutoff must not change when anything at or
after the cutoff changes, and feature functions refuse histories that contain the future."""

from __future__ import annotations

import shutil

import numpy as np
import polars as pl
import pytest

from recsys.config import Config, load_config
from recsys.data.io import load_items
from recsys.data.split import windows
from recsys.eval.harness import build_context
from recsys.features.pit import item_features, user_features
from recsys.ranking.data import Candidates, build_frame, window_labels
from recsys.retrieval.features import item_features as tt_item_features


def _perturbed_cfg(cfg: Config, tmp_path, cutoff: int) -> Config:
    """Copy of the processed data where every interaction at/after `cutoff` is altered."""
    dst = tmp_path / "processed"
    shutil.copytree(cfg.data.processed_dir, dst)
    r = pl.read_parquet(dst / "ratings.parquet")
    fut = pl.col("ts") >= cutoff
    r = r.with_columns(
        pl.when(fut)
        .then(6.0 - pl.col("rating"))
        .otherwise(pl.col("rating"))
        .cast(pl.Float32)
        .alias("rating"),
        pl.when(fut)
        .then((pl.col("item_idx") + 1) % r["item_idx"].max())
        .otherwise(pl.col("item_idx"))
        .cast(pl.Int32)
        .alias("item_idx"),
    )
    extra = (
        r.filter(fut).head(50).with_columns(pl.lit(cutoff, dtype=pl.Int64).alias("ts"))
    )  # rows exactly at cutoff
    pl.concat([r, extra]).sort("ts").write_parquet(dst / "ratings.parquet")
    return load_config(cfg.path, {"data.processed_dir": str(dst), "data.raw_dir": cfg.data.raw_dir})


def test_features_ignore_future(syn_cfg: Config, tmp_path) -> None:
    cutoff = windows(syn_cfg)["test"].start
    feats_by_cfg = []
    for cfg in (syn_cfg, _perturbed_cfg(syn_cfg, tmp_path, cutoff)):
        ctx = build_context(cfg, cutoff)
        rng = np.random.default_rng(0)
        users = np.arange(0, 50, dtype=np.int64)
        items = rng.integers(0, ctx.n_items, size=(50, 10))
        cands = Candidates(users, items, rng.normal(size=(50, 10)).astype(np.float32))
        genres = tt_item_features(cfg, np.zeros((ctx.n_items, 4), np.float32)).genres
        feats_by_cfg.append(
            build_frame(ctx, cands, genres).sort(["user_idx", "item_idx", "retrieval_rank"])
        )
    a, b = feats_by_cfg
    assert a.equals(b)


def test_feature_functions_reject_future_rows(syn_cfg: Config) -> None:
    cutoff = windows(syn_cfg)["test"].start
    ctx = build_context(syn_cfg, windows(syn_cfg)["test"].end)  # contains rows >= cutoff
    with pytest.raises(AssertionError):
        user_features(ctx.history, cutoff, 4.0)
    with pytest.raises(AssertionError):
        item_features(ctx.history, load_items(syn_cfg), cutoff, 4.0)


def test_item_popularity_windows(syn_cfg: Config) -> None:
    h = pl.DataFrame(
        {
            "user_idx": [0, 1, 2],
            "item_idx": [0, 0, 0],
            "rating": [5.0, 5.0, 2.0],
            "ts": [100, 990_000, 999_000],
        },
        schema={"user_idx": pl.Int32, "item_idx": pl.Int32, "rating": pl.Float32, "ts": pl.Int64},
    )
    items = pl.DataFrame(
        {"item_idx": [0], "year": [2000]}, schema={"item_idx": pl.Int32, "year": pl.Int16}
    )
    f = item_features(h, items, cutoff=1_000_000, threshold=4.0).row(0, named=True)
    assert f["i_n_ratings"] == 3 and f["i_n_pos"] == 2
    assert f["i_pos_7d"] == 1  # only the ts=990000 positive is within 7 days of the cutoff


def test_window_labels() -> None:
    w = pl.DataFrame(
        {"user_idx": [0, 0, 1], "item_idx": [5, 5, 6], "rating": [2.0, 4.5, 3.0], "ts": [1, 2, 3]},
        schema={"user_idx": pl.Int32, "item_idx": pl.Int32, "rating": pl.Float32, "ts": pl.Int64},
    )
    lab = window_labels(w, 4.0).sort("user_idx")
    assert lab["label_pos"].to_list() == [1, 0]
    assert lab["label_rating"].to_list() == [4.5, 3.0]
