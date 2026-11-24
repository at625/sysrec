"""Tiny synthetic MovieLens-format dataset for unit tests and CI.

Users and items get latent factors; users pick items with probability proportional to
exp(u.v + popularity), so collaborative signal exists and models can beat random.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Animation"]
TAGS = ["funny", "dark", "classic", "twist ending", "visually stunning", "slow", "quirky"]


def generate(
    out_dir: str | Path,
    n_users: int = 300,
    n_items: int = 200,
    per_user: int = 40,
    t_min: int = 1_550_000_000,
    t_max: int = 1_620_000_000,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dim = 4
    U = rng.normal(size=(n_users, dim))
    V = rng.normal(size=(n_items, dim))
    pop = rng.normal(scale=1.0, size=n_items)

    movie_ids = np.sort(rng.choice(np.arange(1, n_items * 5), size=n_items, replace=False))
    genre_of = [
        GENRES[int(np.argmax(V[i] @ rng.normal(size=(dim, len(GENRES)))))] for i in range(n_items)
    ]
    second = rng.choice(GENRES, size=n_items)
    years = rng.integers(1950, 2020, size=n_items)
    pl.DataFrame(
        {
            "movieId": movie_ids,
            "title": [f"Movie {m} ({y})" for m, y in zip(movie_ids, years, strict=True)],
            "genres": [g if g == s else f"{g}|{s}" for g, s in zip(genre_of, second, strict=True)],
        }
    ).write_csv(out / "movies.csv")

    rows = []
    for u in range(n_users):
        start = rng.integers(t_min, t_max - 5_000_000)
        span = rng.integers(1_000_000, 30_000_000)
        k = int(rng.integers(per_user // 2, per_user * 2))
        logits = U[u] @ V.T + pop
        p = np.exp(logits - logits.max())
        p /= p.sum()
        items = rng.choice(n_items, size=min(k, n_items), replace=False, p=p)
        ts = np.sort(rng.integers(start, min(start + span, t_max), size=len(items)))
        for i, t in zip(items, ts, strict=True):
            score = U[u] @ V[i] / 2 + rng.normal(scale=0.5)
            rating = float(np.clip(np.round((3.0 + score) * 2) / 2, 0.5, 5.0))
            rows.append((u + 1, int(movie_ids[i]), rating, int(t)))
    pl.DataFrame(rows, schema=["userId", "movieId", "rating", "timestamp"], orient="row").write_csv(
        out / "ratings.csv"
    )

    tag_rows = []
    for _ in range(n_users * 2):
        u, i = int(rng.integers(n_users)), int(rng.integers(n_items))
        tag_rows.append(
            (u + 1, int(movie_ids[i]), str(rng.choice(TAGS)), int(rng.integers(t_min, t_max)))
        )
    pl.DataFrame(
        tag_rows, schema=["userId", "movieId", "tag", "timestamp"], orient="row"
    ).write_csv(out / "tags.csv")
    return out
