"""M1: harness behaviour and baselines on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from recsys.config import Config
from recsys.data.split import windows
from recsys.eval.harness import build_context, evaluate, mask_seen, topk_from_scores
from recsys.retrieval.baselines import ALSRecommender, PopularityRecommender


def test_topk_sorted_and_tiebreak() -> None:
    s = np.array([[0.1, 0.9, 0.9, 0.5]])
    assert topk_from_scores(s, 3).tolist() == [[1, 2, 3]]


def test_context_has_no_future(syn_cfg: Config) -> None:
    w = windows(syn_cfg)["test"]
    ctx = build_context(syn_cfg, w.start)
    assert ctx.history["ts"].max() < w.start


def test_cutoff_must_match_window(syn_cfg: Config) -> None:
    w = windows(syn_cfg)
    ctx = build_context(syn_cfg, w["val"].start)
    with pytest.raises(ValueError):
        evaluate(PopularityRecommender(), ctx, w["test"])


def test_recommendations_exclude_seen(syn_cfg: Config) -> None:
    w = windows(syn_cfg)["val"]
    ctx = build_context(syn_cfg, w.start)
    model = PopularityRecommender()
    model.fit(ctx)
    users = ctx.warm_users()[:50]
    top = model.recommend(users, 20)
    for u, row in zip(users, top, strict=True):
        seen = set(ctx.seen[u].indices)
        assert not seen & set(row.tolist())


def test_mask_seen() -> None:
    import scipy.sparse as sp

    seen = sp.csr_matrix(np.array([[1, 0, 1], [0, 0, 0]], dtype=np.float32))
    sc = mask_seen(np.ones((2, 3)), np.array([0, 1]), seen)
    assert np.isneginf(sc[0, 0]) and np.isneginf(sc[0, 2]) and sc[1].min() == 1


def test_als_beats_popularity_on_synthetic(syn_cfg: Config) -> None:
    w = windows(syn_cfg)["val"]
    ctx = build_context(syn_cfg, w.start)
    pop = evaluate(PopularityRecommender(), ctx, w)
    als = evaluate(ALSRecommender(factors=8, iterations=5), ctx, w)
    assert als.metrics["ndcg@10"] > pop.metrics["ndcg@10"]
