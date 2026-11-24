"""M5: build the serving snapshot as of the test-window start, compute offline reference lists for a
sample of users with the batch pipeline, and (optionally) load features into Redis.

  python scripts/build_serving.py [--redis redis://localhost:6379/0]
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from _common import base_parser, config_from  # noqa: E402
from recsys.data.split import eval_targets, windows  # noqa: E402
from recsys.ranking.dcn import DCNConfig, DCNRanker, DCNv2  # noqa: E402
from recsys.ranking.pipeline import fit_stage, ranking_frame  # noqa: E402
from recsys.rerank.mmr import rerank_all  # noqa: E402
from recsys.serving.snapshot import build_snapshot, load_redis  # noqa: E402
from recsys.utils.repro import set_seed  # noqa: E402


def load_dcn(path) -> DCNRanker:
    ck = torch.load(path, weights_only=False)
    c = ck["config"] | {"hidden": tuple(ck["config"]["hidden"])}
    r = DCNRanker(DCNConfig(**c), features=ck["features"], n_items=ck["n_items"])
    r.model = DCNv2(
        len(ck["mu"]),
        c["n_cross"],
        c["hidden"],
        c["dropout"],
        n_items=ck["n_items"] if c["use_item_id"] else 0,
    )
    r.model.load_state_dict(ck["state_dict"])
    r.model.eval()
    r.mu, r.sd = ck["mu"], ck["sd"]
    return r


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--redis", default=None)
    p.add_argument("--reference-users", type=int, default=300)
    args = p.parse_args()
    cfg = config_from(args)
    set_seed(cfg.seed)
    w = windows(cfg)["test"]
    out = cfg.artifacts / "serving"
    rr = json.loads((cfg.results / "m4_rerank.json").read_text())["selected"]
    sel = json.loads((cfg.results / "m1_baselines.json").read_text())["selection"]
    stage = fit_stage(cfg, w.start)
    dcn = load_dcn(cfg.artifacts / "ranking" / "dcn_test.pt")
    meta = build_snapshot(cfg, stage, dcn, out, sel["recent_days"], rerank=rr)

    # Offline reference: the batch pipeline (exact retrieval, polars features, torch DCN, MMR).
    tg = eval_targets(cfg, w)
    rng = np.random.default_rng(cfg.seed)
    users = np.sort(
        rng.choice(tg["user_idx"].to_numpy(), min(args.reference_users, tg.height), replace=False)
    )
    df, _ = ranking_frame(stage, users.astype(np.int64), cfg.section("ranking")["candidates"], None)
    df = df.with_columns(pl.Series("score", dcn.predict(df)))
    lists = rerank_all(
        df,
        users,
        "score",
        stage.retriever.item_emb,
        10,
        rr["lambda"],
        stage.retriever.item_feats.genres,
        rr["max_per_genre"],
    )
    umap = pl.read_parquet(out / "users_map.parquet")["user_id"].to_numpy()
    np.savez(
        out / "offline_reference.npz",
        user_ids=umap[users],
        items=lists,
        movie_ids=np.load(out / "arrays.npz")["movie_ids"][lists],
    )
    print(f"snapshot at {out} (cutoff {meta['cutoff']}); reference lists for {len(users)} users")
    if args.redis:
        print("loaded users into redis:", load_redis(out, args.redis))


if __name__ == "__main__":
    main()
