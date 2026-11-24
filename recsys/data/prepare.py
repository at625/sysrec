"""Convert MovieLens CSVs to typed Parquet with contiguous integer IDs.

Outputs (in cfg.data.processed_dir):
  ratings.parquet  user_idx:int32, item_idx:int32, rating:float32, ts:int64   (sorted by ts)
  items.parquet    item_idx:int32, movie_id:int32, title:str, year:int16 (0 = unknown), genres:list[str]
  users.parquet    user_idx:int32, user_id:int32
  tags.parquet     user_idx:int32, item_idx:int32, tag:str, ts:int64
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

YEAR_RE = r"\((\d{4})\)\s*$"


def prepare(raw_dir: str | Path, out_dir: str | Path) -> dict[str, int]:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    movies = pl.read_csv(raw_dir / "movies.csv", schema_overrides={"movieId": pl.Int32})
    items = (
        movies.sort("movieId")
        .with_row_index("item_idx")
        .select(
            pl.col("item_idx").cast(pl.Int32),
            pl.col("movieId").alias("movie_id"),
            pl.col("title").str.strip_chars(),
            pl.col("title")
            .str.strip_chars()
            .str.extract(YEAR_RE, 1)
            .cast(pl.Int16)
            .fill_null(0)
            .alias("year"),
            pl.when(pl.col("genres") == "(no genres listed)")
            .then(pl.lit(""))
            .otherwise(pl.col("genres"))
            .str.split("|")
            .list.eval(pl.element().filter(pl.element() != ""))
            .alias("genres"),
        )
    )
    items.write_parquet(out_dir / "items.parquet")
    item_map = items.select("movie_id", "item_idx").lazy()

    ratings_raw = pl.scan_csv(
        raw_dir / "ratings.csv",
        schema={
            "userId": pl.Int32,
            "movieId": pl.Int32,
            "rating": pl.Float32,
            "timestamp": pl.Int64,
        },
    )
    users = (
        ratings_raw.select(pl.col("userId").unique().sort())
        .collect()
        .with_row_index("user_idx")
        .select(pl.col("user_idx").cast(pl.Int32), pl.col("userId").alias("user_id"))
    )
    users.write_parquet(out_dir / "users.parquet")
    user_map = users.lazy()

    ratings = (
        ratings_raw.join(user_map, left_on="userId", right_on="user_id")
        .join(item_map, left_on="movieId", right_on="movie_id")
        .select("user_idx", "item_idx", "rating", pl.col("timestamp").alias("ts"))
        .sort("ts", "user_idx", "item_idx")
    )
    ratings.sink_parquet(out_dir / "ratings.parquet")

    tags_path = raw_dir / "tags.csv"
    if tags_path.exists():
        tags = (
            pl.scan_csv(
                tags_path,
                schema={
                    "userId": pl.Int32,
                    "movieId": pl.Int32,
                    "tag": pl.Utf8,
                    "timestamp": pl.Int64,
                },
            )
            .join(user_map, left_on="userId", right_on="user_id", how="left")
            .join(item_map, left_on="movieId", right_on="movie_id")
            .select(
                pl.col("user_idx"),
                "item_idx",
                pl.col("tag").str.strip_chars().str.to_lowercase(),
                pl.col("timestamp").alias("ts"),
            )
            .filter(pl.col("tag").is_not_null() & (pl.col("tag") != ""))
        )
        tags.sink_parquet(out_dir / "tags.parquet")

    n_ratings = pl.scan_parquet(out_dir / "ratings.parquet").select(pl.len()).collect().item()
    return {"n_users": users.height, "n_items": items.height, "n_ratings": n_ratings}
