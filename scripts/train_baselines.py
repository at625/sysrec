"""M1: evaluate popularity, recent popularity and ALS baselines on val (tuning) and test.

Hyperparameters are selected on val; for test every model is refit on all history before the
test window (train + val). Writes results/m1_baselines.json.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # implicit's ALS does its own threading

import dataclasses  # noqa: E402

from _common import base_parser, config_from
from recsys.data.split import windows  # noqa: E402
from recsys.eval.harness import KS, build_context, evaluate  # noqa: E402
from recsys.retrieval.baselines import ALSRecommender, PopularityRecommender  # noqa: E402
from recsys.utils.repro import PeakMemory, set_seed, write_result  # noqa: E402


def main() -> None:
    cfg = config_from(base_parser(__doc__).parse_args())
    set_seed(cfg.seed)
    b = cfg.section("baselines")
    sel = b.get("select_metric", "recall@100")
    w = windows(cfg)
    out: dict = {"val": {}, "test": {}, "selection": {}}

    with PeakMemory() as pm:
        ctx = build_context(cfg, w["val"].start)
        print(f"val context: {ctx.history.height:,} interactions")
        res = evaluate(PopularityRecommender(), ctx, w["val"])
        out["val"]["popular"] = dataclasses.asdict(res)
        best_days, best = None, -1.0
        for d in b["recent_days_grid"]:
            r = evaluate(PopularityRecommender(days=d), ctx, w["val"])
            out["val"][r.model] = dataclasses.asdict(r)
            print(r.model, sel, round(r.metrics[sel], 4))
            if r.metrics[sel] > best:
                best_days, best = d, r.metrics[sel]
        best_als, best = None, -1.0
        for i, params in enumerate(b["als_grid"]):
            m = ALSRecommender(
                **params, iterations=b["als_iterations"], fallback_days=best_days, seed=cfg.seed
            )
            r = evaluate(m, ctx, w["val"])
            key = f"als[{i}]"
            out["val"][key] = dataclasses.asdict(r) | {"params": params}
            print(key, params, sel, round(r.metrics[sel], 4), f"fit {r.fit_seconds:.0f}s")
            if r.metrics[sel] > best:
                best_als, best = params, r.metrics[sel]
        out["selection"] = {"metric": sel, "recent_days": best_days, "als": best_als}
        del ctx

        ctx = build_context(cfg, w["test"].start)
        print(f"test context: {ctx.history.height:,} interactions")
        for m in [
            PopularityRecommender(),
            PopularityRecommender(days=best_days),
            ALSRecommender(
                **best_als, iterations=b["als_iterations"], fallback_days=best_days, seed=cfg.seed
            ),
        ]:
            r = evaluate(m, ctx, w["test"])
            out["test"][r.model] = dataclasses.asdict(r)
            print(
                "test",
                r.model,
                {
                    k: round(v, 4)
                    for k, v in r.metrics.items()
                    if k.startswith(("recall", "ndcg@10"))
                },
            )
    out["peak_rss_gb"] = round(pm.peak_gb, 2)
    out["wall_seconds"] = round(pm.seconds, 1)
    out["ks"] = list(KS)
    write_result(cfg.results / "m1_baselines.json", out, cfg)


if __name__ == "__main__":
    main()
