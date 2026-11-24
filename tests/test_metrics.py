"""Metric functions against hand-computed examples."""

from __future__ import annotations

import math

import numpy as np
import pytest

from recsys.eval.metrics import catalog_coverage, popularity_bias, ranking_metrics


def test_single_user_hand_computed() -> None:
    # ranks:      1  2  3  4
    topk = np.array([[5, 1, 7, 2]])
    targets = [np.array([1, 2, 9])]  # hits at ranks 2 and 4; 3 targets
    m = ranking_metrics(topk, targets, ks=[2, 4])
    assert m["recall@2"] == pytest.approx(1 / 3)
    assert m["recall@4"] == pytest.approx(2 / 3)
    assert m["hitrate@2"] == 1.0
    assert m["mrr@4"] == pytest.approx(0.5)
    dcg4 = 1 / math.log2(3) + 1 / math.log2(5)
    idcg4 = 1 + 1 / math.log2(3) + 1 / math.log2(4)  # min(3, 4) = 3 ideal hits
    assert m["ndcg@4"] == pytest.approx(dcg4 / idcg4)
    dcg2 = 1 / math.log2(3)
    idcg2 = 1 + 1 / math.log2(3)
    assert m["ndcg@2"] == pytest.approx(dcg2 / idcg2)


def test_perfect_and_empty_rankings() -> None:
    topk = np.array([[1, 2, 3], [7, 8, 9]])
    targets = [np.array([1, 2]), np.array([4])]
    m = ranking_metrics(topk, targets, ks=[3])
    assert m["recall@3"] == pytest.approx(0.5)  # user0: 1.0, user1: 0.0
    assert m["ndcg@3"] == pytest.approx(0.5)
    assert m["mrr@3"] == pytest.approx(0.5)
    assert m["hitrate@3"] == pytest.approx(0.5)


def test_mrr_first_hit_only() -> None:
    m = ranking_metrics(np.array([[0, 3, 4]]), [np.array([3, 4])], ks=[3])
    assert m["mrr@3"] == pytest.approx(0.5)


def test_coverage_and_popularity_bias() -> None:
    topk = np.array([[0, 1], [0, 2]])
    assert catalog_coverage(topk, n_items=10, k=2) == pytest.approx(0.3)
    pop = np.arange(10, dtype=float)
    assert popularity_bias(topk, pop, k=2) == pytest.approx((0 + 1 + 0 + 2) / 4)


def test_requires_targets() -> None:
    with pytest.raises(AssertionError):
        ranking_metrics(np.array([[1]]), [np.array([], dtype=int)], ks=[1])
