"""M6: offline replay "A/B" simulation on the held-out test window.

Variants: (A) recent-popular, (B) ALS, (C) two-tower only, (D) two-tower + DCN-v2 ranker,
(E) D + MMR diversity re-ranking. Every variant is scored on the same users (test-window users with
>= 1 positive), so comparisons are paired. A and B are refit here; C/D/E lists come from
scripts/train_rankers.py and scripts/run_rerank.py (same snapshot). Writes results/m6_ab_sim.json,
docs/figures/ab_sim.png. This is an offline replay, NOT a live experiment (see docs/design.md, M6).
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import base_parser, config_from  # noqa: E402
from recsys.data.split import eval_targets, windows  # noqa: E402
from recsys.eval.bootstrap import bootstrap_ci, paired_bootstrap  # noqa: E402
from recsys.eval.harness import build_context  # noqa: E402
from recsys.eval.metrics import catalog_coverage, per_user_metrics  # noqa: E402
from recsys.rerank.mmr import intra_list_diversity  # noqa: E402
from recsys.retrieval.baselines import ALSRecommender, PopularityRecommender  # noqa: E402
from recsys.utils.repro import set_seed, write_result  # noqa: E402

VARIANTS = {
    "A": "recent-popular",
    "B": "ALS",
    "C": "two-tower",
    "D": "two-tower + DCN-v2",
    "E": "two-tower + DCN-v2 + MMR",
}
METRICS = ["ndcg@10", "recall@10", "hitrate@10"]


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()
    cfg = config_from(args)
    set_seed(cfg.seed)
    w = windows(cfg)["test"]
    rd = cfg.artifacts / "ranking"
    tg = eval_targets(cfg, w)
    users = tg["user_idx"].to_numpy()
    assert np.array_equal(np.load(rd / "users_test.npy"), users), (
        "M3 artifacts are for different users"
    )
    targets = [np.asarray(t) for t in tg["targets"].to_list()]

    sel = json.loads((cfg.results / "m1_baselines.json").read_text())["selection"]
    ctx = build_context(cfg, w.start)
    pop = PopularityRecommender(days=sel["recent_days"])
    pop.fit(ctx)
    als = ALSRecommender(
        **sel["als"],
        iterations=cfg.section("baselines")["als_iterations"],
        fallback_days=sel["recent_days"],
        seed=cfg.seed,
    )
    als.fit(ctx)
    lists = {
        "A": pop.recommend(users, 10),
        "B": als.recommend(users, 10),
        "C": np.load(rd / "lists_test_retrieval_only.npy")[:, :10],
        "D": np.load(rd / "lists_test_dcn.npy")[:, :10],
        "E": np.load(rd / "lists_test_dcn_mmr.npy")[:, :10],
    }
    emb = np.load(cfg.artifacts / f"two_tower_item_emb_{w.start}.npy")

    per_user = {v: per_user_metrics(lst, targets, 10) for v, lst in lists.items()}
    table: dict = {}
    for v, lst in lists.items():
        table[v] = {"name": VARIANTS[v]} | {
            m: bootstrap_ci(per_user[v][m], args.n_boot, seed=cfg.seed) for m in METRICS
        }
        table[v]["ild@10"] = intra_list_diversity(lst, emb)
        table[v]["catalog_coverage@10"] = catalog_coverage(lst, ctx.n_items, 10)
    pairs = [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),
        ("A", "D"),
        ("A", "E"),
        ("B", "E"),
    ]
    tests = {
        f"{a}->{b}": {
            m: paired_bootstrap(per_user[a][m], per_user[b][m], args.n_boot, seed=cfg.seed)
            for m in METRICS
        }
        for a, b in pairs
    }
    for v, row in table.items():
        print(
            v,
            row["name"],
            {m: round(row[m]["mean"], 4) for m in METRICS},
            "ILD",
            round(row["ild@10"], 3),
        )
    for k, t in tests.items():
        print(k, {m: (round(x["diff"], 4), round(x["p_value"], 4)) for m, x in t.items()})
    write_result(
        cfg.results / "m6_ab_sim.json",
        {
            "window": "test",
            "n_users": len(users),
            "n_boot": args.n_boot,
            "variants": table,
            "paired_tests": tests,
            "note": "offline replay simulation, not a live A/B test",
        },
        cfg,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m in zip(axes, ["ndcg@10", "recall@10"], strict=True):
        means = [table[v][m]["mean"] for v in lists]
        err = [
            [table[v][m]["mean"] - table[v][m]["ci_low"] for v in lists],
            [table[v][m]["ci_high"] - table[v][m]["mean"] for v in lists],
        ]
        ax.bar(list(lists), means, yerr=err, capsize=4, color="#4C72B0")
        ax.set_title(f"{m} on test window (95% bootstrap CI, n={len(users):,} users)", fontsize=9)
        ax.set_xticks(range(len(lists)), [f"{v}\n{VARIANTS[v]}" for v in lists], fontsize=7)
    fig.tight_layout()
    fig.savefig(cfg.figures / "ab_sim.png", dpi=110)


if __name__ == "__main__":
    main()
