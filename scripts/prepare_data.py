"""CSV -> Parquet with contiguous int32 IDs. Logs wall time and peak memory to results/."""

from __future__ import annotations

from _common import base_parser, config_from
from recsys.data.prepare import prepare
from recsys.utils.repro import PeakMemory, write_result


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from(args)
    with PeakMemory() as pm:
        stats = prepare(cfg.data.raw_dir, cfg.data.processed_dir)
    stats.update(peak_rss_gb=round(pm.peak_gb, 2), wall_seconds=round(pm.seconds, 1))
    print(stats)
    write_result(cfg.results / "m0_prepare.json", {"prepare": stats}, cfg)


if __name__ == "__main__":
    main()
