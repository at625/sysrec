"""M2: point-in-time user sequences and the sampled-softmax loss."""

from __future__ import annotations

import numpy as np
import polars as pl
import torch

from recsys.config import Config
from recsys.data.split import windows
from recsys.eval.harness import build_context, evaluate
from recsys.retrieval.features import build_sequences, gather_history, item_features
from recsys.retrieval.train_two_tower import TrainConfig, TwoTowerRecommender
from recsys.retrieval.two_tower import TwoTowerConfig, sampled_softmax_loss


def _toy_history() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_idx": [0, 0, 0, 0, 1],
            "item_idx": [10, 11, 12, 13, 10],
            "rating": [5.0, 2.0, 4.0, 4.5, 4.0],
            "ts": [100, 200, 200, 300, 50],
        },
        schema={"user_idx": pl.Int32, "item_idx": pl.Int32, "rating": pl.Float32, "ts": pl.Int64},
    )


def test_sequences_history_is_strictly_before() -> None:
    seq = build_sequences(_toy_history(), n_users=2, threshold=4.0)
    # user 0 positives in time order: 10@100, 12@200, 13@300
    assert seq.items[seq.start[0] : seq.start[1]].tolist() == [10, 12, 13]
    j = np.arange(seq.start[0], seq.start[1])
    hist = gather_history(seq.items, seq.start[seq.users[j]], j, length=5)
    for row, target_pos in zip(hist, j, strict=True):
        h = [x for x in row if x >= 0]
        assert seq.items[target_pos] not in h
        assert all(seq.ts[seq.items[:].tolist().index(x)] < seq.ts[target_pos] for x in h)
    # point-in-time stats: at ts=200 only the ts=100 rating is "before" (ties excluded)
    pos12 = seq.start[0] + 1
    assert seq.n_ratings_before[pos12] == 1 and seq.ratings_mean_before[pos12] == 5.0
    pos13 = seq.start[0] + 2
    assert seq.n_ratings_before[pos13] == 3
    assert np.isclose(seq.ratings_mean_before[pos13], (5 + 2 + 4) / 3)


def test_loss_masks_duplicate_positives() -> None:
    u = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    v = u.clone()
    items = torch.tensor([7, 7, 8])
    loss = sampled_softmax_loss(u, items, v, 0.1, None)
    assert torch.isfinite(loss)
    # with duplicates masked, rows 0 and 1 only compete against item 8
    logits = u @ v.T / 0.1
    manual = torch.stack(
        [
            -torch.log_softmax(torch.stack([logits[0, 0], logits[0, 2]]), 0)[0],
            -torch.log_softmax(torch.stack([logits[1, 1], logits[1, 2]]), 0)[0],
            -torch.log_softmax(logits[2], 0)[2],
        ]
    ).mean()
    assert torch.isclose(loss, manual, atol=1e-5)


def test_logq_shifts_logits() -> None:
    u = torch.nn.functional.normalize(torch.randn(4, 4), dim=-1)
    v = torch.nn.functional.normalize(torch.randn(4, 4), dim=-1)
    items = torch.arange(4)
    a = sampled_softmax_loss(u, items, v, 0.1, torch.zeros(4))
    b = sampled_softmax_loss(u, items, v, 0.1, None)
    assert torch.isclose(a, b)


def test_two_tower_trains_on_synthetic(syn_cfg: Config) -> None:
    w = windows(syn_cfg)["val"]
    ctx = build_context(syn_cfg, w.start)
    rng = np.random.default_rng(0)
    text = rng.normal(size=(ctx.n_items, 8)).astype(np.float32)
    feats = item_features(syn_cfg, text)
    tc = TrainConfig(
        epochs=2,
        batch_size=64,
        lr=3e-3,
        log_every=0,
        device="cpu",
        model=TwoTowerConfig(dim=8, id_dim=8, hidden=16, history_len=5, temperature=0.1),
    )
    rec = TwoTowerRecommender(feats, tc)
    r = evaluate(rec, ctx, w)
    assert np.isfinite(rec.history_log[-1]["loss"])
    assert r.metrics["recall@100"] > 0.0
