"""M3: build point-in-time ranking data, train LightGBM LambdaRank + DCN-v2, evaluate end to end.

For each eval window W (val, test):
  train snapshot: cutoff = W.start - label_days; labels = [cutoff, W.start)
  eval snapshot:  cutoff = W.start;               labels = W (used only for metrics)
Writes results/m3_ranking.json, ranking frames + model files to artifacts/ranking/.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import dataclasses  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from _common import base_parser, config_from  # noqa: E402
from recsys.data.split import eval_targets, windows  # noqa: E402
from recsys.eval.metrics import ranking_metrics  # noqa: E402
from recsys.ranking.data import FEATURES  # noqa: E402
from recsys.ranking.dcn import DCNConfig, DCNRanker  # noqa: E402
from recsys.ranking.lgbm import LGBMRanker  # noqa: E402
from recsys.ranking.pipeline import DAY, fit_stage, label_users, ranking_frame, rerank_lists  # noqa: E402
from recsys.utils.repro import PeakMemory, set_seed, write_result  # noqa: E402

KS = (10, 50, 100)


def metrics_for(lists: np.ndarray, targets: list[np.ndarray]) -> dict[str, float]:
    return ranking_metrics(lists, targets, [k for k in KS if k <= lists.shape[1]])


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--windows", nargs="*", default=["val", "test"])
    args = p.parse_args()
    cfg = config_from(args)
    set_seed(cfg.seed)
    sec = cfg.section("ranking")
    K = sec["candidates"]
    out_dir = cfg.artifacts / "ranking"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    with PeakMemory() as pm:
        for wname in args.windows:
            w = windows(cfg)[wname]
            res: dict = {}
            t0 = time.perf_counter()
            # ---- training snapshot (label window just before W) ----
            train_cut = w.start - sec["label_days"] * DAY
            stage = fit_stage(cfg, train_cut)
            users, labels = label_users(cfg, train_cut, w.start)
            rng = np.random.default_rng(cfg.seed)
            if len(users) > sec["max_train_users"]:
                users = np.sort(rng.choice(users, sec["max_train_users"], replace=False))
            train_df, _ = ranking_frame(stage, users, K, labels)
            assert train_df["label_pos"].sum() > 0
            del stage
            # ---- eval snapshot ----
            stage = fit_stage(cfg, w.start)
            tg = eval_targets(cfg, w)
            ev_users = tg["user_idx"].to_numpy().astype(np.int64)
            targets = [np.asarray(t) for t in tg["targets"].to_list()]
            _, ev_labels = label_users(cfg, w.start, w.end)
            eval_df, cands = ranking_frame(stage, ev_users, K, ev_labels)
            train_df.write_parquet(out_dir / f"train_{wname}.parquet")
            eval_df.write_parquet(out_dir / f"eval_{wname}.parquet")
            res["data"] = {
                "train_cutoff": train_cut,
                "train_users": len(users),
                "train_rows": train_df.height,
                "train_pos_rate": float(train_df["label_pos"].mean()),
                "train_rows_with_rating": int(train_df["label_rating"].is_not_null().sum()),
                "eval_users": len(ev_users),
                "eval_rows": eval_df.height,
                "candidate_recall@K": float(
                    np.mean(
                        [np.isin(t, c).mean() for t, c in zip(targets, cands.items, strict=True)]
                    )
                ),
                "build_seconds": time.perf_counter() - t0,
            }
            print(wname, res["data"], flush=True)

            # retrieval-only ordering of the same candidates
            res["retrieval_only"] = metrics_for(cands.items, targets)
            lists = {"retrieval_only": cands.items}

            # validation split of training users for LightGBM early stopping
            tr_users = train_df["user_idx"].unique().to_numpy()
            hold = set(rng.choice(tr_users, max(1, len(tr_users) // 10), replace=False).tolist())
            is_hold = train_df["user_idx"].is_in(list(hold))
            lgbm = LGBMRanker(**sec["lgbm"], seed=cfg.seed)
            t1 = time.perf_counter()
            lgbm.fit(train_df.filter(~is_hold), train_df.filter(is_hold))
            s = lgbm.predict(eval_df)
            lists["lgbm"] = rerank_lists(eval_df, s, ev_users, K)
            res["lgbm"] = metrics_for(lists["lgbm"], targets) | {
                "fit_seconds": time.perf_counter() - t1,
                "best_iteration": lgbm.booster.best_iteration,
            }
            res["lgbm_importance_gain"] = lgbm.importance()
            lgbm.booster.save_model(str(out_dir / f"lgbm_{wname}.txt"))

            dcfg = DCNConfig(**sec["dcn"], seed=cfg.seed)
            dcn = DCNRanker(dcfg, n_items=stage.ctx.n_items)
            t1 = time.perf_counter()
            dcn.fit(train_df)
            res["dcn_train_log"] = dcn.log
            p_pos, p_rating = dcn.predict_heads(eval_df)
            for rw in sec["rating_weight_grid"]:
                lst = rerank_lists(eval_df, p_pos + rw * p_rating / 5.0, ev_users, K)
                res[f"dcn_rw{rw}"] = metrics_for(lst, targets)
                if rw == dcfg.rating_weight:
                    lists["dcn"] = lst
            res["dcn"] = res[f"dcn_rw{dcfg.rating_weight}"] | {
                "fit_seconds": time.perf_counter() - t1
            }
            rated = eval_df.filter(pl.col("label_rating").is_not_null())
            if rated.height:
                _, pr = dcn.predict_heads(rated)
                res["dcn_rating_rmse_on_rated_candidates"] = float(
                    np.sqrt(np.mean((pr - rated["label_rating"].to_numpy()) ** 2))
                )
            import torch

            torch.save(
                {
                    "state_dict": dcn.model.state_dict(),
                    "mu": dcn.mu,
                    "sd": dcn.sd,
                    "features": dcn.features,
                    "config": dataclasses.asdict(dcfg),
                    "n_items": stage.ctx.n_items,
                },
                out_dir / f"dcn_{wname}.pt",
            )

            # Feature-group ablation (LightGBM, retrained without each group)
            res["ablation_lgbm_ndcg@10"] = {"all": res["lgbm"]["ndcg@10"]}
            for g, feats in sec.get("ablation_groups", {}).items():
                keep = [f for f in FEATURES if f not in feats]
                m = LGBMRanker(features=keep, **sec["lgbm"], seed=cfg.seed)
                m.fit(train_df.filter(~is_hold), train_df.filter(is_hold))
                lst = rerank_lists(eval_df, m.predict(eval_df), ev_users, K)
                res["ablation_lgbm_ndcg@10"][f"without_{g}"] = metrics_for(lst, targets)["ndcg@10"]
            eval_df.with_columns(
                pl.Series("score_lgbm", s.astype(np.float32)),
                pl.Series("score_dcn_pos", p_pos.astype(np.float32)),
                pl.Series("score_dcn_rating", p_rating.astype(np.float32)),
                pl.Series(
                    "score_dcn", (p_pos + dcfg.rating_weight * p_rating / 5.0).astype(np.float32)
                ),
            ).write_parquet(out_dir / f"eval_{wname}.parquet")
            for name, lst in lists.items():
                np.save(out_dir / f"lists_{wname}_{name}.npy", lst)
            np.save(out_dir / f"users_{wname}.npy", ev_users)
            results[wname] = res
            print(
                wname,
                json.dumps(
                    {
                        k: {
                            m: round(v, 4)
                            for m, v in res[k].items()
                            if m in ("ndcg@10", "recall@10")
                        }
                        for k in ("retrieval_only", "lgbm", "dcn")
                    }
                ),
                flush=True,
            )
    results["peak_rss_gb"] = round(pm.peak_gb, 2)
    results["wall_seconds"] = round(pm.seconds, 1)
    write_result(cfg.results / "m3_ranking.json", results, cfg)


if __name__ == "__main__":
    main()
