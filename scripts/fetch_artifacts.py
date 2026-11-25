"""Download and unpack the trained artifacts published as a GitHub Release, so nothing has to be retrained.

  python scripts/fetch_artifacts.py                 # serving snapshot only (enough for docker compose)
  python scripts/fetch_artifacts.py --all           # + ranking frames/models, embeddings, two-tower checkpoints

Bundles live in Git LFS on the `artifacts` branch of the repo (main stays small to clone):
  sysrec-serving-snapshot.tar.gz      artifacts/serving/        ONNX models, FAISS indexes, Redis feature snapshot
  sysrec-pipeline-artifacts.tar.gz    artifacts/ranking/, text + two-tower embeddings (M3, M4, M6 inputs)
  sysrec-two-tower-checkpoints.tar    artifacts/checkpoints/    the three main two-tower models (reused by
                                                                train_rankers.py / build_serving.py instead of retraining)
Alternative: `git fetch origin artifacts && git lfs pull` in a checkout of that branch.
Each file is checked against scripts/artifacts_v1.sha256 before unpacking. Processed Parquet is not
bundled: `python scripts/download_data.py && python scripts/prepare_data.py` rebuilds it in seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "at625/sysrec"
BRANCH = "artifacts"
SERVING = "sysrec-serving-snapshot.tar.gz"
EXTRA = ["sysrec-pipeline-artifacts.tar.gz", "sysrec-two-tower-checkpoints.tar"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--all", action="store_true", help="also fetch pipeline artifacts and checkpoints (~1.6 GB)"
    )
    p.add_argument(
        "--cache", default=str(ROOT / "downloads"), help="where downloaded bundles are kept"
    )
    args = p.parse_args()
    expected = dict(
        reversed(line.split())
        for line in (ROOT / "scripts" / "artifacts_v1.sha256").read_text().splitlines()
    )
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    for name in [SERVING, *(EXTRA if args.all else [])]:
        dst = cache / name
        if not dst.exists() or sha256(dst) != expected[name]:
            url = f"https://github.com/{REPO}/raw/{BRANCH}/{name}"  # redirects to the LFS object
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dst)
        digest = sha256(dst)
        if digest != expected[name]:
            raise SystemExit(f"checksum mismatch for {name}: {digest}")
        with tarfile.open(dst) as t:
            for m in t.getmembers():  # only regular files/dirs under artifacts/
                if (
                    not m.name.startswith("artifacts/")
                    or ".." in Path(m.name).parts
                    or not (m.isfile() or m.isdir())
                ):
                    raise SystemExit(f"unexpected member in {name}: {m.name}")
            t.extractall(ROOT, filter="data")
        print(f"{name}: sha256 OK, unpacked into artifacts/")


if __name__ == "__main__":
    main()
