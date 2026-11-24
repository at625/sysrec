"""Generate the tiny synthetic dataset and run prepare on it (used by tests and CI)."""

from __future__ import annotations

from _common import base_parser, config_from

from recsys.data.prepare import prepare
from recsys.data.synthetic import generate


def main() -> None:
    p = base_parser(__doc__)
    p.set_defaults(config="configs/synthetic.yaml")
    cfg = config_from(p.parse_args())
    generate(cfg.data.raw_dir, seed=cfg.seed)
    print(prepare(cfg.data.raw_dir, cfg.data.processed_dir))


if __name__ == "__main__":
    main()
