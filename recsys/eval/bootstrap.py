"""Bootstrap confidence intervals over users, and paired bootstrap tests between variants."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    """Percentile bootstrap CI of the mean of per-user values."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    """Paired test of mean(b) - mean(a) on the same users.

    Resamples users jointly, so per-user difficulty cancels. The two-sided p-value is the bootstrap
    estimate of P(diff has the opposite sign of the observed diff), doubled and capped at 1.
    """
    assert a.shape == b.shape
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    obs = d.mean()
    p = min(1.0, 2 * float(np.mean(means <= 0) if obs > 0 else np.mean(means >= 0)))
    return {
        "diff": float(obs),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": p,
        "relative_lift": float(obs / a.mean()) if a.mean() else float("nan"),
    }
