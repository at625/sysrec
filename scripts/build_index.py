"""M2: FAISS benchmark. Exact IndexFlatIP vs IVF vs HNSW on two-tower item embeddings.

Queries are the two-tower user embeddings of test-window users. Reports recall@k vs exact search
and single-query latency (batch size 1, one thread) p50/p99. Writes results/m2_faiss.json and
docs/figures/faiss_tradeoff.png, and saves the chosen serving index to artifacts/.
"""

from __future__ import annotations

import time

import faiss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import base_parser, config_from  # noqa: E402
from recsys.retrieval.ann import build_index, recall_vs_exact, set_search_param  # noqa: E402
from recsys.utils.repro import hardware_label, write_result  # noqa: E402


def latency(index: faiss.Index, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    times, out = [], []
    for row in q:
        t0 = time.perf_counter()
        _, ids = index.search(row[None, :], k)
        times.append((time.perf_counter() - t0) * 1000)
        out.append(ids[0])
    return np.array(times), np.stack(out)


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--queries", type=int, default=2000)
    args = p.parse_args()
    cfg = config_from(args)
    sec = cfg.section("faiss")
    cut = cfg.data.test_start
    items = np.load(cfg.artifacts / f"two_tower_item_emb_{cut}.npy")
    users = np.load(cfg.artifacts / f"two_tower_user_emb_{cut}.npy")
    rng = np.random.default_rng(cfg.seed)
    q = users[rng.choice(len(users), size=min(args.queries, len(users)), replace=False)]
    faiss.omp_set_num_threads(1)
    k = min(args.k, len(items))

    flat = build_index(items, "flat")
    t_flat, exact = latency(flat, q, k)
    rows = [
        {
            "index": "flat",
            "param": None,
            "recall_vs_exact": 1.0,
            "p50_ms": float(np.percentile(t_flat, 50)),
            "p99_ms": float(np.percentile(t_flat, 99)),
            "build_s": 0.0,
        }
    ]
    for kind, build_params, search_name, search_values in [
        ("ivf", {"nlist": n}, "nprobe", sec.get("ivf_nprobe", [1, 4, 16, 64]))
        for n in sec.get("ivf_nlist", [256, 1024])
    ] + [
        ("hnsw", {"M": m}, "efSearch", sec.get("hnsw_ef", [16, 32, 64, 128, 256]))
        for m in sec.get("hnsw_M", [32])
    ]:
        build_params = {
            key: min(v, len(items) // 39) if key == "nlist" else v
            for key, v in build_params.items()
        }
        faiss.omp_set_num_threads(4)
        t0 = time.perf_counter()
        idx = build_index(items, kind, **build_params)
        build_s = time.perf_counter() - t0
        faiss.omp_set_num_threads(1)
        for v in search_values:
            set_search_param(idx, kind, v)
            t, ids = latency(idx, q, k)
            rows.append(
                {
                    "index": kind,
                    "build_params": build_params,
                    "param": {search_name: v},
                    "recall_vs_exact": recall_vs_exact(ids, exact),
                    "p50_ms": float(np.percentile(t, 50)),
                    "p99_ms": float(np.percentile(t, 99)),
                    "build_s": build_s,
                }
            )
            print(rows[-1], flush=True)
    write_result(
        cfg.results / "m2_faiss.json",
        {"k": k, "n_items": len(items), "n_queries": len(q), "threads": 1, "rows": rows},
        cfg,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    for label in sorted({(r["index"], str(r.get("build_params", ""))) for r in rows}):
        pts = [r for r in rows if (r["index"], str(r.get("build_params", ""))) == label]
        ax.plot(
            [r["p50_ms"] for r in pts],
            [r["recall_vs_exact"] for r in pts],
            "o-",
            label=f"{label[0]} {label[1]}".strip(),
        )
    ax.set_xscale("log")
    ax.set_xlabel("p50 latency per query (ms, 1 thread, batch 1)")
    ax.set_ylabel(f"recall@{k} vs exact")
    ax.set_title(f"FAISS trade-off, {len(items):,} items\n{hardware_label()}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.figures / "faiss_tradeoff.png", dpi=110)


if __name__ == "__main__":
    main()
