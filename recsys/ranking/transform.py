"""Feature transform shared by offline training and online serving (no torch dependency)."""

from __future__ import annotations

import numpy as np

LOG_FEATURES = {
    "u_n_ratings",
    "u_n_pos",
    "u_n_last30d",
    "i_n_ratings",
    "i_n_pos",
    "i_pos_7d",
    "i_pos_30d",
    "i_pos_365d",
    "retrieval_rank",
    "u_days_since_last",
    "u_days_since_first",
    "i_days_since_first_rating",
    "x_days_since_similar",
}


def transform_features(cols: dict[str, np.ndarray], features: list[str]) -> np.ndarray:
    """Raw feature columns -> model input: signed log1p for count-like features, NaN -> 0 plus a
    missing-indicator column per feature. Shared by offline training and online serving."""
    out = []
    for f in features:
        c = np.asarray(cols[f], dtype=np.float32).copy()
        missing = np.isnan(c)
        if f in LOG_FEATURES:
            c = np.sign(c) * np.log1p(np.abs(c))
        out.append(np.where(missing, 0.0, c))
        out.append(missing.astype(np.float32))
    return np.stack(out, 1).astype(np.float32)
