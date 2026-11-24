"""Item text embeddings: MiniLM over "title | genres | top tags", using only tags before a cutoff."""

from __future__ import annotations

import os

import numpy as np
import polars as pl

from recsys.config import Config
from recsys.data.io import load_items, scan_tags

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def item_texts(cfg: Config, cutoff: int, top_tags: int = 10) -> list[str]:
    """One string per item_idx. Tags are point-in-time: only tags applied before `cutoff`."""
    items = load_items(cfg)
    tags = (
        scan_tags(cfg)
        .filter(pl.col("ts") < cutoff)
        .group_by("item_idx", "tag")
        .len()
        .sort(["item_idx", "len", "tag"], descending=[False, True, False])
        .group_by("item_idx", maintain_order=True)
        .head(top_tags)
        .group_by("item_idx")
        .agg(pl.col("tag").str.join(", ").alias("tags"))
        .collect()
    )
    df = items.join(tags, on="item_idx", how="left").sort("item_idx")
    return [
        f"{t} | {', '.join(g or [])} | {tg or ''}".strip(" |")
        for t, g, tg in zip(
            df["title"].to_list(), df["genres"].to_list(), df["tags"].to_list(), strict=True
        )
    ]


def embed_texts(texts: list[str], batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)
