"""Data report: sizes, sparsity, rating distribution, long tail, activity over time, split stats.

Writes docs/data.md, docs/figures/data_*.png and results/m0_data_report.json.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from _common import base_parser, config_from  # noqa: E402

from recsys.data.io import scan_ratings  # noqa: E402
from recsys.data.split import eval_targets, windows  # noqa: E402
from recsys.utils.repro import PeakMemory, write_result  # noqa: E402


def iso(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d %H:%M UTC")


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--docs", default="docs")
    args = p.parse_args()
    cfg = config_from(args)
    docs = Path(args.docs)
    figs = docs / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    thr = cfg.data.positive_threshold

    with PeakMemory() as pm:
        r = scan_ratings(cfg)
        basic = (
            r.select(
                pl.len().alias("n"),
                pl.col("user_idx").n_unique().alias("users"),
                pl.col("item_idx").n_unique().alias("items"),
                pl.col("ts").min().alias("tmin"),
                pl.col("ts").max().alias("tmax"),
                (pl.col("rating") >= thr).mean().alias("pos_frac"),
            )
            .collect()
            .row(0, named=True)
        )
        rating_dist = r.group_by("rating").len().sort("rating").collect()
        user_cnt = r.group_by("user_idx").len().select("len").collect()["len"].to_numpy()
        item_cnt = r.group_by("item_idx").len().select("len").collect()["len"].to_numpy()
        per_month = (
            r.with_columns(pl.from_epoch("ts").dt.truncate("1mo").alias("m"))
            .group_by("m")
            .len()
            .sort("m")
            .collect()
        )

        split = {}
        train_items = set(
            r.filter(pl.col("ts") < cfg.data.val_start)
            .select(pl.col("item_idx").unique())
            .collect()["item_idx"]
            .to_list()
        )
        train_users = set(
            r.filter(pl.col("ts") < cfg.data.val_start)
            .select(pl.col("user_idx").unique())
            .collect()["user_idx"]
            .to_list()
        )
        split["train"] = {
            "start": iso(basic["tmin"]),
            "end_exclusive": iso(cfg.data.val_start),
            "interactions": r.filter(pl.col("ts") < cfg.data.val_start)
            .select(pl.len())
            .collect()
            .item(),
            "users": len(train_users),
            "items": len(train_items),
        }
        seen_items, seen_users = set(train_items), set(train_users)
        for name, w in windows(cfg).items():
            wf = r.filter((pl.col("ts") >= w.start) & (pl.col("ts") < w.end))
            n = wf.select(pl.len()).collect().item()
            items_w = set(wf.select(pl.col("item_idx").unique()).collect()["item_idx"].to_list())
            tg = eval_targets(cfg, w)
            eval_users = set(tg["user_idx"].to_list())
            # "History" for this window = everything before its start.
            hist = r.filter(pl.col("ts") < w.start)
            hist_items = set(
                hist.select(pl.col("item_idx").unique()).collect()["item_idx"].to_list()
            )
            hist_users = set(
                hist.select(pl.col("user_idx").unique()).collect()["user_idx"].to_list()
            )
            cold_items = items_w - hist_items
            tgt = tg.explode("targets")
            cold_target_frac = (
                float(tgt["targets"].is_in(list(cold_items)).mean()) if tgt.height else 0.0
            )
            split[name] = {
                "start": iso(w.start),
                "end_exclusive": iso(w.end),
                "interactions": n,
                "eval_users_with_positive": len(eval_users),
                "eval_users_warm": len(eval_users & hist_users),
                "eval_users_cold": len(eval_users - hist_users),
                "positive_targets": tgt.height,
                "items_in_window": len(items_w),
                "cold_start_items": len(cold_items),
                "target_share_on_cold_items": round(cold_target_frac, 4),
            }
            seen_items |= items_w
            seen_users |= set(
                wf.select(pl.col("user_idx").unique()).collect()["user_idx"].to_list()
            )

    sparsity = 1 - basic["n"] / (basic["users"] * basic["items"])
    report = {
        "interactions": basic["n"],
        "users": basic["users"],
        "items_rated": basic["items"],
        "sparsity": sparsity,
        "positive_fraction": basic["pos_frac"],
        "time_range": [iso(basic["tmin"]), iso(basic["tmax"])],
        "rating_distribution": dict(
            zip([str(x) for x in rating_dist["rating"]], rating_dist["len"].to_list(), strict=True)
        ),
        "user_activity": {q: float(np.percentile(user_cnt, q * 100)) for q in [0.5, 0.9, 0.99]}
        | {"min": int(user_cnt.min()), "max": int(user_cnt.max())},
        "item_popularity": {q: float(np.percentile(item_cnt, q * 100)) for q in [0.5, 0.9, 0.99]}
        | {"min": int(item_cnt.min()), "max": int(item_cnt.max())},
        "top1pct_items_share_of_interactions": float(
            np.sort(item_cnt)[::-1][: max(1, len(item_cnt) // 100)].sum() / item_cnt.sum()
        ),
        "split": split,
        "peak_rss_gb": round(pm.peak_gb, 2),
        "wall_seconds": round(pm.seconds, 1),
    }
    write_result(cfg.results / "m0_data_report.json", report, cfg)

    # Plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar([str(x) for x in rating_dist["rating"]], rating_dist["len"])
    ax[0].set_title("Rating distribution")
    for arr, lab in [(user_cnt, "users"), (item_cnt, "items")]:
        s = np.sort(arr)[::-1]
        ax[1].loglog(np.arange(1, len(s) + 1), s, label=lab)
    ax[1].set_title("Long tail (rank vs #ratings, log-log)")
    ax[1].legend()
    ax[2].plot(per_month["m"].to_list(), per_month["len"].to_list())
    for w in windows(cfg).values():
        ax[2].axvline(dt.datetime.fromtimestamp(w.start, dt.UTC), ls="--", c="r")
    ax[2].set_title("Ratings per month (dashed = val/test start)")
    fig.tight_layout()
    fig.savefig(figs / "data_overview.png", dpi=110)

    s = report["split"]
    lines = [
        "# Data report: MovieLens-32M",
        "",
        "Generated by `python scripts/data_report.py` (raw numbers in `results/m0_data_report.json`).",
        "",
        "| | |",
        "|---|---|",
        f"| Interactions | {report['interactions']:,} |",
        f"| Users | {report['users']:,} |",
        f"| Items with >= 1 rating | {report['items_rated']:,} (catalog in movies.csv: see items.parquet) |",
        f"| Sparsity | {report['sparsity']:.5f} |",
        f"| Positive fraction (rating >= {thr}) | {report['positive_fraction']:.3f} |",
        f"| Time range | {report['time_range'][0]} to {report['time_range'][1]} |",
        f"| Median / p90 / p99 ratings per user | {report['user_activity'][0.5]:.0f} / {report['user_activity'][0.9]:.0f} / {report['user_activity'][0.99]:.0f} |",
        f"| Median / p90 / p99 ratings per item | {report['item_popularity'][0.5]:.0f} / {report['item_popularity'][0.9]:.0f} / {report['item_popularity'][0.99]:.0f} |",
        f"| Share of interactions on top 1% of items | {report['top1pct_items_share_of_interactions']:.3f} |",
        "",
        "![overview](figures/data_overview.png)",
        "",
        "## Time-based split",
        "",
        "Cutoffs live in `configs/default.yaml` (`data.val_start`, `data.test_start`, `data.test_end`).",
        "",
        "| Window | Start | End (exclusive) | Interactions | Eval users (>=1 positive) | warm / cold users | Cold-start items | Target share on cold items |",
        "|---|---|---|---|---|---|---|---|",
        f"| train | {s['train']['start']} | {s['train']['end_exclusive']} | {s['train']['interactions']:,} | | | | |",
    ]
    for name in ["val", "test"]:
        x = s[name]
        lines.append(
            f"| {name} | {x['start']} | {x['end_exclusive']} | {x['interactions']:,} | {x['eval_users_with_positive']:,} | "
            f"{x['eval_users_warm']:,} / {x['eval_users_cold']:,} | {x['cold_start_items']:,} | {x['target_share_on_cold_items']:.3f} |"
        )
    lines += [
        "",
        "*Warm* users have at least one interaction before the window starts; *cold* users appear for the "
        "first time inside the window. *Cold-start items* are items rated in the window that were never "
        "rated before the window start. Neither can be scored by ID embeddings alone; see `docs/design.md`.",
        "",
        f"Report computed with peak RSS {report['peak_rss_gb']} GB in {report['wall_seconds']} s.",
        "",
    ]
    (docs / "data.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
