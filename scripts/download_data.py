"""Download MovieLens-32M from GroupLens and verify the published MD5 checksum."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
EXPECTED_MD5 = "d472be332d4daa821edc399621853b57"  # from ml-32m.zip.md5 on files.grouplens.org


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/raw")
    p.add_argument(
        "--cafile",
        default=os.environ.get("RECSYS_CAFILE"),
        help="optional CA bundle (GroupLens chains to the emSign root, which some "
        "minimal trust stores lack; certifi's bundle has it)",
    )
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / "ml-32m.zip"
    if not zpath.exists() or md5(zpath) != EXPECTED_MD5:
        import ssl

        ctx = ssl.create_default_context(cafile=args.cafile) if args.cafile else None
        if ctx is None:
            try:
                import certifi

                ctx = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                ctx = ssl.create_default_context()
        print(f"downloading {URL} ...")
        with urllib.request.urlopen(URL, context=ctx) as r, zpath.open("wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    digest = md5(zpath)
    if digest != EXPECTED_MD5:
        raise SystemExit(f"checksum mismatch: got {digest}, expected {EXPECTED_MD5}")
    print(f"md5 OK ({digest})")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    print(f"extracted to {out / 'ml-32m'}")


if __name__ == "__main__":
    main()
