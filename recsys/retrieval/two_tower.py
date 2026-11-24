"""Two-tower retrieval model (Covington et al. 2016; Yi et al. 2019).

User tower: user-ID embedding (row 0 = unknown/cold bucket) + mean-pooled embeddings of the user's
last L positive items before the example time + simple point-in-time user features.
Item tower: item-ID embedding + genre multi-hot + release-year bucket + frozen MiniLM text vector.
Both towers end in an MLP and L2 normalisation; the score is a temperature-scaled cosine.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from recsys.retrieval.features import ItemFeatures


@dataclass
class TwoTowerConfig:
    dim: int = 64
    id_dim: int = 64
    hidden: int = 256
    history_len: int = 50
    temperature: float = 0.05
    dropout: float = 0.1
    use_user_id: bool = True


def mlp(d_in: int, hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, d_out)
    )


class ItemTower(nn.Module):
    def __init__(self, n_items: int, feats: ItemFeatures, cfg: TwoTowerConfig) -> None:
        super().__init__()
        self.id_emb = nn.Embedding(n_items, cfg.id_dim)
        nn.init.normal_(self.id_emb.weight, std=0.02)
        self.year_emb = nn.Embedding(feats.n_year_buckets, 8)
        self.register_buffer("genres", torch.from_numpy(feats.genres))
        self.register_buffer("year", torch.from_numpy(feats.year_bucket))
        self.register_buffer("text", torch.from_numpy(feats.text))
        d_in = cfg.id_dim + feats.genres.shape[1] + 8 + feats.text.shape[1]
        self.mlp = mlp(d_in, cfg.hidden, cfg.dim, cfg.dropout)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [
                self.id_emb(items),
                self.genres[items],
                self.year_emb(self.year[items]),
                self.text[items],
            ],
            dim=-1,
        )
        return F.normalize(self.mlp(x), dim=-1)


class UserTower(nn.Module):
    N_DENSE = 2  # log1p(#ratings before t), mean rating before t

    def __init__(self, n_users: int, item_id_emb: nn.Embedding, cfg: TwoTowerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.user_emb = nn.Embedding(n_users + 1, cfg.id_dim)  # row 0 = unknown user
        nn.init.normal_(self.user_emb.weight, std=0.02)
        self.item_id_emb = item_id_emb  # shared with the item tower
        self.mlp = mlp(cfg.id_dim * 2 + self.N_DENSE, cfg.hidden, cfg.dim, cfg.dropout)

    def forward(self, users: torch.Tensor, hist: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        """users: (B,) user_idx+1 (0 = unknown); hist: (B, L) item_idx, -1 = pad; dense: (B, 2)."""
        mask = (hist >= 0).float().unsqueeze(-1)
        h = self.item_id_emb(hist.clamp(min=0)) * mask
        pooled = h.sum(1) / mask.sum(1).clamp(min=1.0)
        u = self.user_emb(users) if self.cfg.use_user_id else torch.zeros_like(pooled)
        return F.normalize(self.mlp(torch.cat([u, pooled, dense], dim=-1)), dim=-1)


class TwoTower(nn.Module):
    def __init__(
        self, n_users: int, n_items: int, feats: ItemFeatures, cfg: TwoTowerConfig
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.item_tower = ItemTower(n_items, feats, cfg)
        self.user_tower = UserTower(n_users, self.item_tower.id_emb, cfg)

    @torch.no_grad()
    def all_item_embeddings(self, batch: int = 8192) -> torch.Tensor:
        n = self.item_tower.id_emb.num_embeddings
        dev = self.item_tower.id_emb.weight.device
        return torch.cat(
            [
                self.item_tower(torch.arange(s, min(s + batch, n), device=dev))
                for s in range(0, n, batch)
            ]
        )


def sampled_softmax_loss(
    user_vec: torch.Tensor,
    pos_items: torch.Tensor,
    item_vec: torch.Tensor,
    temperature: float,
    log_q: torch.Tensor | None,
    extra_items: torch.Tensor | None = None,
    extra_vec: torch.Tensor | None = None,
    extra_log_q: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-batch softmax with optional logQ correction and extra (e.g. uniform) negatives.

    logits[b, j] = <u_b, v_j> / T - log q_j. Column j == b is the positive. Other columns holding
    the same item as row b's positive are masked (they are not real negatives).
    """
    logits = user_vec @ item_vec.T / temperature
    cand = pos_items
    if log_q is not None:
        logits = logits - log_q[None, :]
    if extra_vec is not None:
        ex = user_vec @ extra_vec.T / temperature
        if extra_log_q is not None:
            ex = ex - extra_log_q[None, :]
        logits = torch.cat([logits, ex], dim=1)
        cand = torch.cat([pos_items, extra_items])
    b = len(pos_items)
    same = pos_items[:, None] == cand[None, :]
    same[torch.arange(b), torch.arange(b)] = False
    logits = logits.masked_fill(same, float("-inf"))
    return F.cross_entropy(logits, torch.arange(b, device=logits.device))
