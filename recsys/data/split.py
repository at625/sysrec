"""Global time-based split.

  train: ts <  val_start
  val:   val_start  <= ts < test_start
  test:  test_start <= ts < test_end

When evaluating a window, the model may use every interaction strictly before the window start
("history"), and is scored on the positives inside the window ("targets"). For the val window the
history is train; for the test window the history is train + val. Nothing at or after a window's
start is ever visible to the model that is scored on that window.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from recsys.config import Config
from recsys.data.io import scan_ratings


@dataclass(frozen=True)
class Window:
    name: str
    start: int  # inclusive
    end: int  # exclusive


def windows(cfg: Config) -> dict[str, Window]:
    d = cfg.data
    return {
        "val": Window("val", d.val_start, d.test_start),
        "test": Window("test", d.test_start, d.test_end),
    }


def train_frame(cfg: Config) -> pl.LazyFrame:
    return scan_ratings(cfg).filter(pl.col("ts") < cfg.data.val_start)


def history_before(cfg: Config, cutoff: int) -> pl.LazyFrame:
    """All interactions strictly before `cutoff` (the data a model may train on)."""
    return scan_ratings(cfg).filter(pl.col("ts") < cutoff)


def window_frame(cfg: Config, w: Window) -> pl.LazyFrame:
    return scan_ratings(cfg).filter((pl.col("ts") >= w.start) & (pl.col("ts") < w.end))


def positives(lf: pl.LazyFrame, threshold: float) -> pl.LazyFrame:
    return lf.filter(pl.col("rating") >= threshold)


def eval_targets(cfg: Config, w: Window) -> pl.DataFrame:
    """Per-user list of positive items in the window, for users with >= 1 positive.

    An item rated positively twice in the window is counted once.
    """
    return (
        positives(window_frame(cfg, w), cfg.data.positive_threshold)
        .group_by("user_idx")
        .agg(pl.col("item_idx").unique().sort().alias("targets"))
        .sort("user_idx")
        .collect()
    )
