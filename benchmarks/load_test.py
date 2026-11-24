"""Open-loop load generator for /recommend.

Fires requests at a fixed rate (Poisson arrivals) regardless of response time, so queueing shows up
in latency instead of silently lowering the offered load. Records client-side end-to-end latency and
the server-reported per-stage timings. Appends a run to results/m5_latency.json under --label.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.utils.repro import git_commit, hardware, hardware_label, lib_versions  # noqa: E402


async def run_rate(
    client: httpx.AsyncClient,
    url: str,
    users: np.ndarray,
    rate: float,
    seconds: float,
    rng: np.random.Generator,
) -> dict:
    lat, stages, errors, tasks = [], {}, 0, []

    async def one(uid: int) -> None:
        nonlocal errors
        t0 = time.perf_counter()
        try:
            r = await client.get(url, params={"user_id": int(uid), "k": 10})
            r.raise_for_status()
            body = r.json()
            lat.append((time.perf_counter() - t0) * 1000)
            for s, ms in body["timings_ms"].items():
                stages.setdefault(s, []).append(ms)
        except Exception:  # noqa: BLE001
            errors += 1

    start = time.perf_counter()
    t_next = start
    while t_next - start < seconds:
        now = time.perf_counter()
        if now < t_next:
            await asyncio.sleep(t_next - now)
        tasks.append(asyncio.create_task(one(rng.choice(users))))
        t_next += rng.exponential(1.0 / rate)
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - start
    pct = lambda a: {f"p{q}": float(np.percentile(a, q)) for q in (50, 95, 99)} if a else {}  # noqa: E731
    return {
        "offered_rps": rate,
        "duration_s": seconds,
        "completed": len(lat),
        "errors": errors,
        "achieved_rps": len(lat) / wall,
        "e2e_ms": pct(lat),
        "stages_ms": {s: pct(v) for s, v in stages.items()},
    }


async def main_async(args: argparse.Namespace) -> dict:
    users = np.load(args.users)
    rng = np.random.default_rng(0)
    limits = httpx.Limits(max_connections=args.max_connections)
    out = []
    async with httpx.AsyncClient(base_url=args.url, timeout=30, limits=limits) as c:
        for _ in range(20):  # warm-up
            await c.get("/recommend", params={"user_id": int(users[0]), "k": 10})
        health = (await c.get("/health")).json()
        for rate in args.rates:
            res = await run_rate(c, "/recommend", users, rate, args.seconds, rng)
            print(json.dumps(res), flush=True)
            out.append(res)
    return {"health": health, "runs": out}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--users", default="artifacts/serving/load_users.npy")
    p.add_argument("--rates", type=float, nargs="+", default=[5, 10, 20, 40, 80])
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--max-connections", type=int, default=200)
    p.add_argument("--label", required=True)
    p.add_argument("--note", default="")
    p.add_argument("--out", default="results/m5_latency.json")
    args = p.parse_args()
    res = asyncio.run(main_async(args))
    path = Path(args.out)
    data = json.loads(path.read_text()) if path.exists() else {"runs": {}}
    data["runs"][args.label] = res | {
        "note": args.note,
        "hardware": hardware(),
        "hardware_label": hardware_label(),
        "libraries": lib_versions(),
        "git_commit": git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
