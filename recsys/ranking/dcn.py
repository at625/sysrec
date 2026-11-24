"""DCN-v2 multi-task ranker (Wang et al. 2021): cross network + deep MLP, two heads.

Heads: P(positive) (BCE on label_pos) and predicted rating (MSE on rows rated in the label window).
Final score = sigmoid(logit) + rating_weight * (predicted_rating / 5).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from recsys.ranking.data import FEATURES
from recsys.ranking.transform import LOG_FEATURES, transform_features  # noqa: F401
from recsys.utils.device import get_device


class CrossLayer(nn.Module):
    """x_{l+1} = x0 * (W x_l + b) + x_l  (full-rank DCN-v2 cross)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * self.lin(x) + x


class DCNv2(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_cross: int = 3,
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        n_items: int = 0,
        item_dim: int = 16,
    ) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(n_items, item_dim) if n_items else None
        d = d_in + (item_dim if n_items else 0)
        self.cross = nn.ModuleList([CrossLayer(d) for _ in range(n_cross)])
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.deep = nn.Sequential(*layers)
        self.head_pos = nn.Linear(d + prev, 1)
        self.head_rating = nn.Linear(d + prev, 1)

    def forward(
        self, x: torch.Tensor, items: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.item_emb is not None and items is not None:
            x = torch.cat([x, self.item_emb(items)], dim=-1)
        xc = x
        for layer in self.cross:
            xc = layer(x, xc)
        h = torch.cat([xc, self.deep(x)], dim=-1)
        return self.head_pos(h).squeeze(-1), 1.0 + 4.0 * torch.sigmoid(
            self.head_rating(h).squeeze(-1)
        )


@dataclass
class DCNConfig:
    epochs: int = 5
    batch_size: int = 4096
    lr: float = 1e-3
    n_cross: int = 3
    hidden: tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    rating_loss_weight: float = 0.5
    rating_weight: float = 0.2  # weight of predicted rating in the final score
    use_item_id: bool = False
    device: str = "auto"
    seed: int = 0


class DCNRanker:
    name = "dcn_v2_multitask"

    def __init__(self, cfg: DCNConfig, features: list[str] | None = None, n_items: int = 0) -> None:
        self.cfg = cfg
        self.features = features or FEATURES
        self.n_items = n_items
        self.log: list[dict] = []

    def _matrix(self, df: pl.DataFrame) -> np.ndarray:
        return transform_features(
            {f: df[f].cast(pl.Float32).to_numpy() for f in self.features}, self.features
        )

    def fit(self, train: pl.DataFrame) -> None:
        c = self.cfg
        torch.manual_seed(c.seed)
        rng = np.random.default_rng(c.seed)
        dev = get_device(c.device)
        X = self._matrix(train)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        X = (X - self.mu) / self.sd
        y = train["label_pos"].to_numpy().astype(np.float32)
        r = train["label_rating"].fill_null(np.nan).to_numpy().astype(np.float32)
        items = train["item_idx"].to_numpy().astype(np.int64)
        self.model = DCNv2(
            X.shape[1],
            c.n_cross,
            tuple(c.hidden),
            c.dropout,
            n_items=self.n_items if c.use_item_id else 0,
        ).to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=c.lr)
        Xt, yt, rt, it = (torch.from_numpy(a) for a in (X, y, r, items))
        for epoch in range(c.epochs):
            self.model.train()
            t0, tot = time.perf_counter(), []
            perm = torch.from_numpy(rng.permutation(len(X)))
            for s in range(0, len(X), c.batch_size):
                j = perm[s : s + c.batch_size]
                logit, pred_r = self.model(Xt[j].to(dev), it[j].to(dev))
                loss = F.binary_cross_entropy_with_logits(logit, yt[j].to(dev))
                rr = rt[j].to(dev)
                m = ~torch.isnan(rr)
                if m.any():
                    loss = loss + c.rating_loss_weight * F.mse_loss(pred_r[m], rr[m])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot.append(loss.item())
            self.log.append(
                {"epoch": epoch, "loss": float(np.mean(tot)), "seconds": time.perf_counter() - t0}
            )
        self.model.eval()

    @torch.no_grad()
    def predict_heads(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        dev = next(self.model.parameters()).device
        X = (self._matrix(df) - self.mu) / self.sd
        items = torch.from_numpy(df["item_idx"].to_numpy().astype(np.int64))
        ps, rs = [], []
        for s in range(0, len(X), 65536):
            logit, r = self.model(
                torch.from_numpy(X[s : s + 65536]).to(dev), items[s : s + 65536].to(dev)
            )
            ps.append(torch.sigmoid(logit).cpu().numpy())
            rs.append(r.cpu().numpy())
        return np.concatenate(ps), np.concatenate(rs)

    def predict(self, df: pl.DataFrame, rating_weight: float | None = None) -> np.ndarray:
        p, r = self.predict_heads(df)
        w = self.cfg.rating_weight if rating_weight is None else rating_weight
        return p + w * r / 5.0
