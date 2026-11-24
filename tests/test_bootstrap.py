from __future__ import annotations

import numpy as np

from recsys.eval.bootstrap import bootstrap_ci, paired_bootstrap


def test_ci_contains_mean() -> None:
    x = np.random.default_rng(0).normal(1.0, 1.0, size=2000)
    ci = bootstrap_ci(x)
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]
    assert ci["ci_low"] < 1.0 < ci["ci_high"]


def test_paired_detects_shift_and_null() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=1000)
    b = a + 0.1  # same users, constant lift: paired test must be decisive
    r = paired_bootstrap(a, b)
    assert r["p_value"] < 0.01 and r["ci_low"] > 0
    r0 = paired_bootstrap(a, a + rng.normal(scale=1e-3, size=1000) - 0.0)
    assert r0["ci_low"] <= 0 <= r0["ci_high"] or abs(r0["diff"]) < 1e-3
