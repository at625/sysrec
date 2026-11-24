"""LightGBM LambdaRank baseline ranker (one query group per user)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from recsys.ranking.data import FEATURES


def _groups(df: pl.DataFrame) -> np.ndarray:
    # rows must be contiguous per user
    return df.group_by("user_idx", maintain_order=True).len()["len"].to_numpy()


class LGBMRanker:
    name = "lgbm_lambdarank"

    def __init__(self, features: list[str] | None = None, **params) -> None:
        self.features = features or FEATURES
        self.params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [10],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambdarank_truncation_level": 20,
            "verbose": -1,
            "seed": 0,
        } | params
        self.num_rounds = self.params.pop("num_rounds", 300)

    def fit(self, train: pl.DataFrame, valid: pl.DataFrame | None = None) -> None:
        train = train.sort("user_idx", maintain_order=True)
        ds = lgb.Dataset(
            train.select(self.features).to_numpy(),
            train["label_pos"].to_numpy(),
            group=_groups(train),
            feature_name=self.features,
            free_raw_data=False,
        )
        sets, cbs = [ds], []
        if valid is not None:
            valid = valid.sort("user_idx", maintain_order=True)
            sets.append(
                lgb.Dataset(
                    valid.select(self.features).to_numpy(),
                    valid["label_pos"].to_numpy(),
                    group=_groups(valid),
                    reference=ds,
                )
            )
            cbs.append(lgb.early_stopping(30, verbose=False))
        self.booster = lgb.train(self.params, ds, self.num_rounds, valid_sets=sets, callbacks=cbs)

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        return self.booster.predict(
            df.select(self.features).to_numpy(), num_iteration=self.booster.best_iteration or None
        )

    def importance(self) -> dict[str, float]:
        gain = self.booster.feature_importance("gain")
        return dict(zip(self.features, (gain / gain.sum()).tolist(), strict=True))
