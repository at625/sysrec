from __future__ import annotations

import numpy as np

from recsys.rerank.mmr import genre_coverage, intra_list_diversity, mmr


def _setup():
    # items 0,1 near-duplicates; item 2 orthogonal
    emb = np.array([[1, 0], [0.99, 0.01], [0, 1]], dtype=np.float32)
    rel = np.array([1.0, 0.9, 0.5])
    return emb, rel


def test_lambda_one_is_relevance_order() -> None:
    emb, rel = _setup()
    assert mmr(np.array([0, 1, 2]), rel, emb, 3, lam=1.0).tolist() == [0, 1, 2]


def test_low_lambda_promotes_diverse_item() -> None:
    emb, rel = _setup()
    assert mmr(np.array([0, 1, 2]), rel, emb, 2, lam=0.3).tolist() == [0, 2]


def test_genre_cap() -> None:
    emb, rel = _setup()
    genres = np.array([[1, 0], [1, 0], [0, 1]])
    out = mmr(np.array([0, 1, 2]), rel, emb, 2, lam=1.0, item_genres=genres, max_per_genre=1)
    assert out.tolist() == [0, 2]


def test_diversity_metrics() -> None:
    emb = np.array([[1, 0], [0, 1]], dtype=np.float32)
    assert intra_list_diversity(np.array([[0, 1]]), emb) == 1.0
    assert genre_coverage(np.array([[0, 1]]), np.array([[1, 0, 0], [0, 1, 0]])) == 2 / 3
