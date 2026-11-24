"""M2: train + evaluate the two-tower retriever and its ablations on val and test.

Variants (configs: two_tower + two_tower.ablations) are trained on history before each window
and evaluated with the M1 harness. Writes results/m2_two_tower.json.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from _common import base_parser, config_from
from recsys.data.split import windows
from recsys.eval.harness import build_context, evaluate
from recsys.retrieval.features import item_features
from recsys.retrieval.train_two_tower import TwoTowerRecommender, train_config_from_section
from recsys.utils.repro import PeakMemory, set_seed, write_result


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--only", nargs="*", help="only these variants (main = two_tower)")
    p.add_argument("--windows", nargs="*", default=None)
    p.add_argument("--out", default="m2_two_tower.json")
    p.add_argument(
        "--tuning", action="store_true", help="run two_tower.tuning variants on val only"
    )
    args = p.parse_args()
    cfg = config_from(args)
    set_seed(cfg.seed)
    sec = cfg.section("two_tower")
    variants = {"two_tower": {}} | {
        f"two_tower_{k}": v for k, v in sec.get("ablations", {}).items()
    }
    if args.tuning:
        variants = {"two_tower": {}} | {f"tune_{k}": v for k, v in sec.get("tuning", {}).items()}
        args.windows = ["val"]
        args.out = "m2_tuning.json"
    if args.only:
        variants = {k: v for k, v in variants.items() if k in args.only}
    abl_windows = set(sec.get("ablation_windows", ["val", "test"]))
    out: dict = {"val": {}, "test": {}, "train_configs": {}}
    with PeakMemory() as pm:
        for wname, w in windows(cfg).items():
            if args.windows and wname not in args.windows:
                continue
            ctx = build_context(cfg, w.start)
            feats = item_features(cfg, np.load(cfg.artifacts / f"text_emb_{w.start}.npy"))
            for name, ov in variants.items():
                if name != "two_tower" and wname not in abl_windows:
                    continue
                tc = train_config_from_section(sec, cfg.seed, ov)
                out["train_configs"][name] = dataclasses.asdict(tc)
                rec = TwoTowerRecommender(
                    feats, tc, name=name, checkpoint_dir=cfg.artifacts / "checkpoints"
                )
                r = evaluate(rec, ctx, w)
                out[wname][name] = dataclasses.asdict(r) | {
                    "epochs": rec.history_log,
                    "train_info": rec.train_info,
                }
                print(
                    wname,
                    name,
                    {
                        k: round(v, 4)
                        for k, v in r.metrics.items()
                        if k in ("recall@10", "recall@100", "ndcg@10", "recall@500")
                    },
                    flush=True,
                )
                if name == "two_tower":
                    from recsys.data.split import eval_targets

                    np.save(cfg.artifacts / f"two_tower_item_emb_{w.start}.npy", rec.item_emb)
                    ev_users = eval_targets(cfg, w)["user_idx"].to_numpy()
                    ev_users = ev_users[rec.has_pos[ev_users]]
                    np.save(
                        cfg.artifacts / f"two_tower_user_emb_{w.start}.npy",
                        rec.user_embeddings(ev_users),
                    )
    out["peak_rss_gb"] = round(pm.peak_gb, 2)
    out["wall_seconds"] = round(pm.seconds, 1)
    write_result(cfg.results / args.out, out, cfg)


if __name__ == "__main__":
    main()
