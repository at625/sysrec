# Design notes

One section per milestone: what was built, why, alternatives considered, and trade-offs.

## M0: Data and the time-based split

**What.** Raw MovieLens-32M CSVs are converted once into typed Parquet (`recsys/data/prepare.py`).
User and movie IDs are remapped to contiguous `int32` indices (`user_idx`, `item_idx`) so they can be
used directly as embedding rows and array offsets. Ratings are `float32`, timestamps `int64`.
The full conversion peaks at about 3.2 GB RSS using Polars' lazy scan and streaming sink.

**Split.** A single global cutoff split (not per-user leave-last-out):

| window | range |
|---|---|
| train | everything before 2023-04-01 |
| val | 2023-04-01 to 2023-07-01 |
| test | 2023-07-01 to 2023-10-13 |

When a model is scored on a window it may use only interactions strictly before the window's start.
For validation that is the train set; for the final test numbers it is train + val (models are refit
on that history with hyperparameters chosen on val). A global cutoff is what a deployed system
actually faces: it is trained on the past and serves the future. Leave-last-out splits leak future
information across users (another user's later ratings shape the popularity and item embeddings used
to predict an earlier event), which inflates metrics.

**Evaluation population.** Users with at least one positive (rating >= 4) inside the window. Some
of them have no earlier history ("cold" users); they are kept in the evaluation and handled by the
popularity fallback, and metrics are also reported for warm users only. See `docs/data.md`.

**Alternatives considered.** Leave-one-out (standard in papers, but leaks time); random split (worse
leakage); per-user temporal split (still mixes calendar time across users). The trade-off of a global
cutoff is that the eval windows are small (a few thousand users) because MovieLens activity has
declined over time, so confidence intervals matter; M6 reports bootstrap CIs.

**Subsampling.** `data.user_fraction` keeps a stable hash-based user subset, so every model sees the
same users if a step has to be scaled down.
