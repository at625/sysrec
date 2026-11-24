"""Precompute MiniLM item text embeddings for each evaluation cutoff (tags strictly before it)."""

from __future__ import annotations

import time

import numpy as np

from _common import base_parser, config_from
from recsys.data.split import windows
from recsys.features.text import embed_texts, item_texts


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument(
        "--fake", action="store_true", help="random vectors (tests/CI, no model download)"
    )
    args = p.parse_args()
    cfg = config_from(args)
    for w in windows(cfg).values():
        t0 = time.perf_counter()
        texts = item_texts(cfg, w.start)
        if args.fake:
            rng = np.random.default_rng(0)
            emb = rng.normal(size=(len(texts), 384)).astype(np.float32)
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        else:
            emb = embed_texts(texts)
        np.save(cfg.artifacts / f"text_emb_{w.start}.npy", emb)
        print(f"{w.name}: {emb.shape} in {time.perf_counter() - t0:.0f}s; e.g. {texts[0][:120]!r}")


if __name__ == "__main__":
    main()
