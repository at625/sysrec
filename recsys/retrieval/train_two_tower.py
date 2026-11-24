"""Training loop + Recommender wrapper for the two-tower model."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from recsys.eval.harness import FitContext, mask_seen, topk_from_scores
from recsys.retrieval.baselines import PopularityRecommender
from recsys.retrieval.features import ItemFeatures, Sequences, build_sequences, gather_history
from recsys.retrieval.two_tower import TwoTower, TwoTowerConfig, sampled_softmax_loss
from recsys.utils.device import get_device

DAY = 86_400


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 0.0
    train_days: int | None = None  # only targets within this many days before the cutoff
    logq: bool = True
    uniform_negatives: int = 0  # > 0 = mixed negatives (in-batch + this many uniform per batch)
    device: str = "auto"
    seed: int = 0
    log_every: int = 500
    model: TwoTowerConfig = field(default_factory=TwoTowerConfig)


def dense_features(n_before: np.ndarray, mean_before: np.ndarray) -> np.ndarray:
    return np.stack([np.log1p(n_before) / 5.0, mean_before / 5.0], axis=1).astype(np.float32)


class TwoTowerRecommender:
    def __init__(
        self,
        item_feats: ItemFeatures,
        cfg: TrainConfig,
        name: str = "two_tower",
        checkpoint_dir: Path | None = None,
        fallback_days: int = 30,
    ) -> None:
        self.item_feats = item_feats
        self.cfg = cfg
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.fallback = PopularityRecommender(days=fallback_days)
        self.history_log: list[dict] = []

    # ---------- training ----------
    def _examples(self, seq: Sequences, cutoff: int) -> np.ndarray:
        pos = np.arange(len(seq.items))
        if self.cfg.train_days is not None:
            pos = pos[seq.ts >= cutoff - self.cfg.train_days * DAY]
        return pos

    def fit(self, ctx: FitContext) -> None:
        c = self.cfg
        torch.manual_seed(c.seed)
        rng = np.random.default_rng(c.seed)
        dev = get_device(c.device)
        t_start = time.perf_counter()
        seq = build_sequences(ctx.history, ctx.n_users, ctx.cfg.data.positive_threshold)
        ex = self._examples(seq, ctx.cutoff)
        L = c.model.history_len
        # Item sampling distribution of training targets -> logQ correction.
        q = np.bincount(seq.items[ex], minlength=ctx.n_items).astype(np.float64)
        q = q / q.sum()
        B, M = c.batch_size, c.uniform_negatives
        q_cand = (B * q + M / ctx.n_items) / (B + M)
        log_q = torch.from_numpy(np.log(np.maximum(q_cand, 1e-12)).astype(np.float32)).to(dev)

        model = TwoTower(ctx.n_users, ctx.n_items, self.item_feats, c.model).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        user_start = seq.start[seq.users]
        dense_all = dense_features(seq.n_ratings_before, seq.ratings_mean_before)
        self.train_info = {"examples": int(len(ex)), "device": str(dev)}
        for epoch in range(c.epochs):
            model.train()
            perm = rng.permutation(ex)
            t0, losses = time.perf_counter(), []
            for step, s in enumerate(range(0, len(perm) - B + 1, B)):
                j = perm[s : s + B]
                hist = torch.from_numpy(gather_history(seq.items, user_start[j], j, L)).to(dev)
                users = torch.from_numpy(seq.users[j] + 1).to(dev)
                dense = torch.from_numpy(dense_all[j]).to(dev)
                items = torch.from_numpy(seq.items[j]).to(dev)
                u = model.user_tower(users, hist, dense)
                v = model.item_tower(items)
                extra_items = extra_vec = extra_lq = None
                if M > 0:
                    extra_items = torch.from_numpy(rng.integers(0, ctx.n_items, size=M)).to(dev)
                    extra_vec = model.item_tower(extra_items)
                    extra_lq = log_q[extra_items] if c.logq else None
                loss = sampled_softmax_loss(
                    u,
                    items,
                    v,
                    c.model.temperature,
                    log_q[items] if c.logq else None,
                    extra_items,
                    extra_vec,
                    extra_lq,
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.item())
                if c.log_every and step % c.log_every == 0:
                    print(
                        f"  [{self.name}] epoch {epoch} step {step} loss {np.mean(losses[-c.log_every :]):.4f}",
                        flush=True,
                    )
            info = {
                "epoch": epoch,
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "seconds": time.perf_counter() - t0,
            }
            self.history_log.append(info)
            print(f"  [{self.name}] epoch {epoch} done: {info}", flush=True)
            if self.checkpoint_dir is not None:
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "train_config": dataclasses.asdict(c),
                    },
                    self.checkpoint_dir / f"{self.name}_cut{ctx.cutoff}.pt",
                )
        self.train_info["train_seconds"] = time.perf_counter() - t_start
        self.model = model.eval()
        self.ctx = ctx
        self.fallback.fit(ctx)
        self._prepare_inference(seq)

    # ---------- inference ----------
    def _prepare_inference(self, seq: Sequences) -> None:
        """User state at the cutoff: last L positives and stats over all history."""
        ctx = self.ctx
        L = self.cfg.model.history_len
        ends = seq.start[1:]  # end of each user's sequence (exclusive)
        self.user_hist = gather_history(seq.items, seq.start[:-1], ends, L)  # (n_users, L)
        self.has_pos = np.diff(seq.start) > 0
        h = ctx.history
        cnt = np.bincount(h["user_idx"].to_numpy(), minlength=ctx.n_users).astype(np.float32)
        s = np.bincount(
            h["user_idx"].to_numpy(), weights=h["rating"].to_numpy(), minlength=ctx.n_users
        )
        self.user_dense = dense_features(
            cnt, np.where(cnt > 0, s / np.maximum(cnt, 1), 0).astype(np.float32)
        )
        self.item_emb = self.model.all_item_embeddings().cpu().numpy()

    @torch.no_grad()
    def user_embeddings(self, users: np.ndarray) -> np.ndarray:
        dev = next(self.model.parameters()).device
        out = self.model.user_tower(
            torch.from_numpy(users + 1).to(dev),
            torch.from_numpy(self.user_hist[users]).to(dev),
            torch.from_numpy(self.user_dense[users]).to(dev),
        )
        return out.cpu().numpy()

    def score_users(self, users: np.ndarray) -> np.ndarray:
        return self.user_embeddings(users) @ self.item_emb.T

    def recommend(self, users: np.ndarray, k: int, batch: int = 1024) -> np.ndarray:
        out = np.zeros((len(users), k), dtype=np.int64)
        warm = self.has_pos[users]
        if (~warm).any():
            out[~warm] = self.fallback.recommend(users[~warm], k)
        wu = users[warm]
        res = [
            topk_from_scores(
                mask_seen(self.score_users(wu[s : s + batch]), wu[s : s + batch], self.ctx.seen), k
            )
            for s in range(0, len(wu), batch)
        ]
        if res:
            out[warm] = np.concatenate(res)
        return out
