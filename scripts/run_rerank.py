"""M4: MMR diversity re-ranking on top of the DCN-v2 ranker's candidate scores.

Sweeps MMR lambda (with and without a max-N-per-genre rule) on the val window, picks the default
operating point, then reports the same sweep on test. Writes results/m4_rerank.json and
docs/figures/rerank_tradeoff.png.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from _common import base_parser, config_from  # noqa: E402
from recsys.data.split import eval_targets, windows  # noqa: E402
from recsys.eval.metrics import catalog_coverage, ranking_metrics  # noqa: E402
from recsys.rerank.mmr import genre_coverage, intra_list_diversity, rerank_all  # noqa: E402
from recsys.retrieval.features import item_features  # noqa: E402
from recsys.utils.repro import write_result  # noqa: E402


def evaluate_lists(lists, targets, emb, genres, n_items) -> dict[str, float]:
    m = ranking_metrics(lists, targets, [10])
    return {
        "ndcg@10": m["ndcg@10"],
        "recall@10": m["recall@10"],
        "ild@10": intra_list_diversity(lists, emb),
        "genre_coverage@10": genre_coverage(lists, genres),
        "catalog_coverage@10": catalog_coverage(lists, n_items, 10),
    }


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--score", default="score_dcn")
    args = p.parse_args()
    cfg = config_from(args)
    sec = cfg.section("rerank")
    lams = sec.get("lambdas", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3])
    caps = sec.get("genre_caps", [None, 3])
    rd = cfg.artifacts / "ranking"
    out: dict = {"score": args.score, "lambdas": lams, "genre_caps": caps}
    for wname, w in windows(cfg).items():
        df = pl.read_parquet(rd / f"eval_{wname}.parquet")
        users = np.load(rd / f"users_{wname}.npy")
        tg = eval_targets(cfg, w)
        assert np.array_equal(tg["user_idx"].to_numpy(), users)
        targets = [np.asarray(t) for t in tg["targets"].to_list()]
        emb = np.load(cfg.artifacts / f"two_tower_item_emb_{w.start}.npy")
        genres = item_features(cfg, np.zeros((len(emb), 1), np.float32)).genres
        rows = []
        for cap in caps:
            for lam in lams:
                lists = rerank_all(df, users, args.score, emb, 10, lam, genres, cap)
                r = {"lambda": lam, "max_per_genre": cap} | evaluate_lists(
                    lists, targets, emb, genres, len(emb)
                )
                rows.append(r)
                print(wname, r, flush=True)
                if lam == sec.get("default_lambda") and cap == sec.get("default_max_per_genre"):
                    np.save(rd / f"lists_{wname}_dcn_mmr.npy", lists)
        out[wname] = rows

    fig, ax = plt.subplots(figsize=(7, 5))
    for cap in caps:
        pts = [r for r in out["val"] if r["max_per_genre"] == cap]
        ax.plot(
            [r["ild@10"] for r in pts],
            [r["ndcg@10"] for r in pts],
            "o-",
            label=f"val, genre cap={cap}",
        )
        for r in pts:
            ax.annotate(f"λ={r['lambda']}", (r["ild@10"], r["ndcg@10"]), fontsize=7)
    ax.set_xlabel("intra-list diversity@10 (1 - mean cosine, two-tower item embeddings)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("MMR re-ranking: accuracy vs diversity (val window)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.figures / "rerank_tradeoff.png", dpi=110)
    write_result(cfg.results / "m4_rerank.json", out, cfg)


if __name__ == "__main__":
    main()
