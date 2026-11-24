"""Build a self-contained serving snapshot (models + features as of one cutoff) and load it into Redis.

Snapshot directory layout:
  user_tower.onnx, ranker.onnx       exported models
  item_emb.npy, faiss_{kind}.index   two-tower item embeddings + ANN indexes
  arrays.npz                         item-side arrays (features, genres, ALS factors, popularity lists)
  users.npz, users_map.parquet       user-side records (loaded into Redis)
  meta.json                          cutoff, feature order, config used
User features go to Redis (hash u:{movielens_user_id}); item feature vectors go to Redis (i:{item_idx}),
with an optional in-process cache (the M5 optimisation).

Training-side imports (torch, faiss, polars feature code) are inside build functions so the serving
image, which only runs `load_redis` and the engine, doesn't need torch.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl


def _ranker_with_norm(dcn):
    """DCN-v2 + input standardisation + final score, so serving feeds transformed raw features."""
    import torch
    from torch import nn

    class RankerWithNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = dcn.model
            self.register_buffer("mu", torch.from_numpy(dcn.mu))
            self.register_buffer("sd", torch.from_numpy(dcn.sd))
            self.w = float(dcn.cfg.rating_weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            p, r = self.model((x - self.mu) / self.sd)
            return torch.sigmoid(p) + self.w * r / 5.0

    return RankerWithNorm()


def export_onnx(module, args: tuple, path: Path, names: list[str]) -> None:
    import torch

    module = module.eval().cpu()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            module,
            args,
            str(path),
            input_names=names,
            output_names=["out"],
            dynamic_axes={n: {0: "batch"} for n in [*names, "out"]},
            dynamo=False,
        )


def build_snapshot(
    cfg, stage, dcn, out: Path, fallback_days: int, rerank: dict | None = None
) -> dict:
    """Export everything the online engine needs, as of `stage.ctx.cutoff`."""
    import faiss
    import torch

    from recsys.data.io import load_items
    from recsys.features.pit import item_features, user_features, user_genre_profile
    from recsys.ranking.data import FEATURES
    from recsys.retrieval.ann import build_index

    u_feats = [f for f in FEATURES if f.startswith("u_")]
    i_feats = [f for f in FEATURES if f.startswith("i_")]
    out.mkdir(parents=True, exist_ok=True)
    ctx, tt, als = stage.ctx, stage.retriever, stage.als
    thr = cfg.data.positive_threshold
    L = tt.cfg.model.history_len
    dummy = (
        torch.ones(2, dtype=torch.long),
        torch.zeros(2, L, dtype=torch.long),
        torch.zeros(2, 2),
    )
    export_onnx(tt.model.user_tower, dummy, out / "user_tower.onnx", ["users", "hist", "dense"])
    export_onnx(_ranker_with_norm(dcn), (torch.zeros(2, len(dcn.mu)),), out / "ranker.onnx", ["x"])
    np.save(out / "item_emb.npy", tt.item_emb)
    for kind, params in (("flat", {}), ("hnsw", {"M": 32})):
        faiss.write_index(
            build_index(tt.item_emb, kind, **params), str(out / f"faiss_{kind}.index")
        )

    items = load_items(cfg)
    ifeat = item_features(ctx.history, items, ctx.cutoff, thr).sort("item_idx")
    pop = tt.fallback.scores
    genres = tt.item_feats.genres
    genre_pop = np.stack(
        [
            np.argsort(-np.where(genres[:, g] > 0, pop, -1), kind="stable")[:500]
            for g in range(genres.shape[1])
        ]
    )
    np.savez(
        out / "arrays.npz",
        item_feats=ifeat.select(i_feats).cast(pl.Float32).to_numpy(),
        genres=genres,
        als_item=als.item_f,
        recent_pop_scores=pop,
        recent_pop=np.argsort(-pop, kind="stable")[:5000],
        genre_pop=genre_pop,
        movie_ids=items.sort("item_idx")["movie_id"].to_numpy(),
    )
    meta = {
        "cutoff": ctx.cutoff,
        "features": FEATURES,
        "u_feats": u_feats,
        "i_feats": i_feats,
        "genre_names": tt.item_feats.genre_names,
        "history_len": L,
        "fallback_days": fallback_days,
        "n_items": ctx.n_items,
        "dcn_rating_weight": dcn.cfg.rating_weight,
        "candidates": cfg.section("ranking").get("candidates", 200),
        "mmr_lambda": (rerank or {}).get("lambda", 1.0),
        "max_per_genre": (rerank or {}).get("max_per_genre"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    uf = user_features(ctx.history, ctx.cutoff, thr)
    U = np.full((ctx.n_users, len(u_feats)), np.nan, dtype=np.float32)
    U[uf["user_idx"].to_numpy()] = uf.select(u_feats).cast(pl.Float32).to_numpy()
    share, last = user_genre_profile(ctx.history, genres, ctx.n_users, ctx.cutoff, thr)
    np.savez(
        out / "users.npz",
        hist=tt.user_hist.astype(np.int32),
        dense=tt.user_dense,
        has_pos=tt.has_pos,
        ufeat=U,
        gshare=share,
        glast=last,
        als_user=als.user_f,
        als_has_pos=als.has_pos,
        seen_indptr=ctx.seen.indptr,
        seen_indices=ctx.seen.indices.astype(np.int32),
    )
    pl.read_parquet(cfg.processed / "users.parquet").sort("user_idx").write_parquet(
        out / "users_map.parquet"
    )
    return meta


def load_redis(snapshot: Path, redis_url: str, batch: int = 5000) -> int:
    """Push user and item feature records into Redis. Returns the number of users written."""
    import redis

    r = redis.Redis.from_url(redis_url)
    u = np.load(snapshot / "users.npz")
    a = np.load(snapshot / "arrays.npz")
    hist, dense, has_pos, ufeat = u["hist"], u["dense"], u["has_pos"], u["ufeat"]
    gshare, glast, als_user, als_has = u["gshare"], u["glast"], u["als_user"], u["als_has_pos"]
    indptr, indices = u["seen_indptr"], u["seen_indices"]
    ids = pl.read_parquet(snapshot / "users_map.parquet")["user_id"].to_numpy()
    pipe = r.pipeline(transaction=False)
    for i in range(len(ids)):
        pipe.hset(
            f"u:{ids[i]}",
            mapping={
                "idx": int(i),
                "hist": hist[i].tobytes(),
                "dense": dense[i].tobytes(),
                "has_pos": int(has_pos[i]),
                "ufeat": ufeat[i].tobytes(),
                "gshare": gshare[i].tobytes(),
                "glast": glast[i].tobytes(),
                "als": als_user[i].tobytes() if als_has[i] else b"",
                "seen": indices[indptr[i] : indptr[i + 1]].tobytes(),
            },
        )
        if (i + 1) % batch == 0:
            pipe.execute()
    item_feats = a["item_feats"]
    for j in range(len(item_feats)):
        pipe.set(f"i:{j}", item_feats[j].tobytes())
        if (j + 1) % batch == 0:
            pipe.execute()
    pipe.set("snapshot:cutoff", json.loads((snapshot / "meta.json").read_text())["cutoff"])
    pipe.execute()
    return len(ids)
