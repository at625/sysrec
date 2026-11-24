"""CLI: push a serving snapshot's features into Redis (used by the compose `loader` service)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from recsys.serving.snapshot import load_redis


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", default="artifacts/serving")
    p.add_argument("--redis", default="redis://localhost:6379/0")
    a = p.parse_args()
    t0 = time.perf_counter()
    n = load_redis(Path(a.snapshot), a.redis)
    print(f"loaded {n} users + item features into {a.redis} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
