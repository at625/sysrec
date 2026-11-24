"""Shared fixtures: a tiny synthetic dataset prepared once per test session."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.config import Config, load_config  # noqa: E402
from recsys.data.prepare import prepare  # noqa: E402
from recsys.data.synthetic import generate  # noqa: E402


@pytest.fixture(scope="session")
def syn_cfg(tmp_path_factory: pytest.TempPathFactory) -> Config:
    base = tmp_path_factory.mktemp("syn")
    cfg = load_config(
        ROOT / "configs" / "synthetic.yaml",
        {
            "data.raw_dir": str(base / "raw"),
            "data.processed_dir": str(base / "processed"),
            "artifacts_dir": str(base / "artifacts"),
            "results_dir": str(base / "results"),
        },
    )
    generate(cfg.data.raw_dir, seed=cfg.seed)
    prepare(cfg.data.raw_dir, cfg.data.processed_dir)
    return cfg
