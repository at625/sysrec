"""FAISS index construction for inner-product (cosine, since embeddings are L2-normalised) search."""

from __future__ import annotations

import faiss
import numpy as np


def build_index(item_emb: np.ndarray, kind: str, **params: int) -> faiss.Index:
    """kind: "flat" (exact), "ivf" (nlist), or "hnsw" (M, ef_construction)."""
    x = np.ascontiguousarray(item_emb, dtype=np.float32)
    d = x.shape[1]
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "ivf":
        quant = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quant, d, params.get("nlist", 1024), faiss.METRIC_INNER_PRODUCT)
        index.train(x)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, params.get("M", 32), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = params.get("ef_construction", 200)
    else:
        raise ValueError(kind)
    index.add(x)
    return index


def set_search_param(index: faiss.Index, kind: str, value: int) -> None:
    if kind == "ivf":
        index.nprobe = value
    elif kind == "hnsw":
        index.hnsw.efSearch = value


def recall_vs_exact(approx: np.ndarray, exact: np.ndarray) -> float:
    """Mean |approx ∩ exact| / k over queries."""
    k = exact.shape[1]
    return float(
        np.mean([len(np.intersect1d(a, e)) / k for a, e in zip(approx, exact, strict=True)])
    )
