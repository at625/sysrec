"""Shared M3+ pipeline pieces: fitted retrieval stage at a cutoff, ranking frames, list metrics."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import polars as pl

from recsys.config import Config
from recsys.data.split import history_before, positives
from recsys.eval.harness import FitContext, build_context
from recsys.ranking.data import Candidates, build_frame, retrieve_candidates, window_labels
from recsys.retrieval.baselines import ALSRecommender
from recsys.retrieval.features import item_features as tt_item_features
from recsys.retrieval.train_two_tower import TwoTowerRecommender, train_config_from_section

DAY = 86_400


@dataclass
class Stage:
    ctx: FitContext
    retriever: TwoTowerRecommender
    als: ALSRecommender


def fit_stage(cfg: Config, cutoff: int, text_cutoff: int | None = None) -> Stage:
    """Fit the two-tower retriever (main config, cached) and the selected ALS on history < cutoff."""
    import json

    ctx = build_context(cfg, cutoff)
    text_path = cfg.artifacts / f"text_emb_{text_cutoff or cutoff}.npy"
    if not text_path.exists():
        from recsys.features.text import embed_texts, item_texts

        np.save(
            text_path,
            embed_texts(item_texts(cfg, cutoff))
            if not os.environ.get("RECSYS_FAKE_TEXT")
            else _fake_text(ctx.n_items),
        )
    feats = tt_item_features(cfg, np.load(text_path))
    tc = train_config_from_section(cfg.section("two_tower"), cfg.seed)
    tt = TwoTowerRecommender(
        feats, tc, name="two_tower", checkpoint_dir=cfg.artifacts / "checkpoints"
    )
    tt.fit(ctx)
    sel = json.loads((cfg.results / "m1_baselines.json").read_text())["selection"]
    als = ALSRecommender(
        **sel["als"],
        iterations=cfg.section("baselines")["als_iterations"],
        fallback_days=sel["recent_days"],
        seed=cfg.seed,
    )
    als.fit(ctx)
    return Stage(ctx, tt, als)


def _fake_text(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 384)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def label_users(cfg: Config, start: int, end: int) -> tuple[np.ndarray, pl.DataFrame]:
    """Users with >= 1 positive in [start, end) and the (user, item) labels of that window."""
    from recsys.data.io import scan_ratings

    wdf = scan_ratings(cfg).filter((pl.col("ts") >= start) & (pl.col("ts") < end)).collect()
    users = np.sort(
        positives(wdf.lazy(), cfg.data.positive_threshold)
        .select(pl.col("user_idx").unique())
        .collect()["user_idx"]
        .to_numpy()
    )
    return users.astype(np.int64), window_labels(wdf, cfg.data.positive_threshold)


def ranking_frame(
    stage: Stage, users: np.ndarray, k: int, labels: pl.DataFrame | None
) -> tuple[pl.DataFrame, Candidates]:
    cands = retrieve_candidates(stage.retriever, users, k)
    df = build_frame(stage.ctx, cands, stage.retriever.item_feats.genres, stage.als, labels)
    return df, cands


def rerank_lists(df: pl.DataFrame, scores: np.ndarray, users: np.ndarray, k: int) -> np.ndarray:
    """Order each user's candidates by `scores` (desc) and return the top-k item matrix aligned to `users`."""
    ranked = (
        df.select("user_idx", "item_idx")
        .with_columns(pl.Series("s", scores))
        .sort(["user_idx", "s"], descending=[False, True])
        .group_by("user_idx", maintain_order=True)
        .head(k)
        .group_by("user_idx", maintain_order=True)
        .agg("item_idx")
    )
    m = dict(zip(ranked["user_idx"].to_list(), ranked["item_idx"].to_list(), strict=True))
    return np.array([m[u][:k] for u in users], dtype=np.int64)


def history_exists(cfg: Config, cutoff: int) -> bool:
    return history_before(cfg, cutoff).select(pl.len()).collect().item() > 0
