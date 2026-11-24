"""M5: online/offline parity. Runs the online engine (exact FAISS index) on the reference users and
compares top-10 lists to the offline batch pipeline for the same snapshot. Also reports how much the
HNSW index changes the lists. Writes results/m5_parity.json."""

from __future__ import annotations

import json

import numpy as np
import redis

from _common import base_parser, config_from
from recsys.serving.engine import Engine, EngineConfig
from recsys.utils.repro import write_result


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--redis", default="redis://localhost:6379/0")
    args = p.parse_args()
    cfg = config_from(args)
    snap = cfg.artifacts / "serving"
    ref = np.load(snap / "offline_reference.npz")
    rr = json.loads((cfg.results / "m4_rerank.json").read_text())["selected"]
    r = redis.Redis.from_url(args.redis)
    out = {}
    for index in ("flat", "hnsw"):
        eng = Engine(
            snap,
            r,
            EngineConfig(
                index=index,
                mmr_lambda=rr["lambda"],
                max_per_genre=rr["max_per_genre"],
                candidates=cfg.section("ranking")["candidates"],
            ),
        )
        exact, overlap = [], []
        for uid, want in zip(ref["user_ids"], ref["movie_ids"], strict=True):
            got = eng.recommend(int(uid), 10).items
            exact.append(got == want.tolist())
            overlap.append(len(set(got) & set(want.tolist())) / 10)
        out[index] = {
            "users": len(exact),
            "identical_lists": float(np.mean(exact)),
            "mean_top10_overlap": float(np.mean(overlap)),
        }
        print(index, out[index])
    write_result(cfg.results / "m5_parity.json", out, cfg)


if __name__ == "__main__":
    main()
