"""Plot results/m5_latency.json: p50/p99 end-to-end latency vs offered load for each serving config."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    d = json.loads(Path("results/m5_latency.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 5))
    hw = ""
    for label, r in d["runs"].items():
        hw = r.get("hardware_label", hw)
        ok = [
            x for x in r["runs"] if x["achieved_rps"] >= 0.95 * x["offered_rps"] and not x["errors"]
        ]
        if not ok:
            continue
        rates = [x["offered_rps"] for x in ok]
        (line,) = ax.plot(rates, [x["e2e_ms"]["p99"] for x in ok], "o-", label=f"{label} p99")
        ax.plot(
            rates,
            [x["e2e_ms"]["p50"] for x in ok],
            "o--",
            color=line.get_color(),
            alpha=0.6,
            label=f"{label} p50",
        )
    ax.set_xlabel(
        "offered load (requests/s, open loop); points shown only where the service kept up"
    )
    ax.set_ylabel("end-to-end latency (ms)")
    ax.set_yscale("log")
    ax.set_title(
        f"/recommend latency vs load\n{hw}; load generator on the same machine", fontsize=9
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("docs/figures/latency.png", dpi=110)


if __name__ == "__main__":
    main()
