"""M0: data preparation and the time-based split."""

from __future__ import annotations

import hashlib

import polars as pl

from recsys.config import Config
from recsys.data.io import load_items, scan_ratings
from recsys.data.split import eval_targets, history_before, train_frame, window_frame, windows


def test_prepare_types_and_contiguous_ids(syn_cfg: Config) -> None:
    r = scan_ratings(syn_cfg).collect()
    assert r.schema["user_idx"] == pl.Int32 and r.schema["item_idx"] == pl.Int32
    assert r.schema["rating"] == pl.Float32 and r.schema["ts"] == pl.Int64
    items = load_items(syn_cfg)
    assert items["item_idx"].to_list() == list(range(items.height))
    assert r["item_idx"].max() < items.height
    assert r["ts"].is_sorted()


def test_prepare_preserves_rows(syn_cfg: Config) -> None:
    raw = pl.read_csv(f"{syn_cfg.data.raw_dir}/ratings.csv")
    assert scan_ratings(syn_cfg).select(pl.len()).collect().item() == raw.height


def test_windows_are_ordered_and_disjoint(syn_cfg: Config) -> None:
    d = syn_cfg.data
    assert d.val_start < d.test_start < d.test_end
    w = windows(syn_cfg)
    assert w["val"].end == w["test"].start


def test_no_test_interaction_in_training(syn_cfg: Config) -> None:
    train = train_frame(syn_cfg).collect()
    test = window_frame(syn_cfg, windows(syn_cfg)["test"]).collect()
    assert train["ts"].max() < syn_cfg.data.val_start
    assert test["ts"].min() >= syn_cfg.data.test_start
    key = ["user_idx", "item_idx", "ts"]
    assert train.join(test, on=key, how="inner").height == 0


def test_history_strictly_before_window(syn_cfg: Config) -> None:
    for w in windows(syn_cfg).values():
        h = history_before(syn_cfg, w.start).collect()
        assert h["ts"].max() < w.start


def test_eval_targets_only_positives_in_window(syn_cfg: Config) -> None:
    w = windows(syn_cfg)["test"]
    tg = eval_targets(syn_cfg, w).explode("targets")
    wf = (
        window_frame(syn_cfg, w)
        .filter(pl.col("rating") >= syn_cfg.data.positive_threshold)
        .collect()
    )
    pairs = set(zip(wf["user_idx"], wf["item_idx"], strict=True))
    assert all((u, i) in pairs for u, i in zip(tg["user_idx"], tg["targets"], strict=True))


def test_split_is_reproducible(syn_cfg: Config) -> None:
    def digest() -> str:
        t = train_frame(syn_cfg).collect().sort(["ts", "user_idx", "item_idx"])
        return hashlib.sha256(t.write_ipc(None).getvalue()).hexdigest()

    assert digest() == digest()


def test_user_subsample_is_stable(syn_cfg: Config) -> None:
    from recsys.config import load_config

    sub = load_config(
        syn_cfg.path, {"data.processed_dir": syn_cfg.data.processed_dir, "data.user_fraction": 0.5}
    )
    a = scan_ratings(sub).select(pl.col("user_idx").unique().sort()).collect()
    b = scan_ratings(sub).select(pl.col("user_idx").unique().sort()).collect()
    assert a.equals(b)
    n_all = scan_ratings(syn_cfg).select(pl.col("user_idx").n_unique()).collect().item()
    assert 0.3 * n_all < a.height < 0.7 * n_all
