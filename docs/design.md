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

## M1: Evaluation harness and baselines

**What.** `recsys/eval/metrics.py` implements Recall, NDCG, HitRate and MRR at K in {10, 50, 100, 500},
catalog coverage, and popularity bias (mean history-popularity of recommended items).
`recsys/eval/harness.py` fits a model on everything before a window's start, scores the **full
catalog** (87,585 items) for each evaluation user, masks every item the user interacted with before
the window, and takes the top K. Metric definitions are spelled out in the module docstring and
unit-tested against hand-computed examples (`tests/test_metrics.py`).

**Why full-catalog.** Sampled metrics (rank the target among 100 random negatives) change the
ordering of models, not just the scale (Krichene & Rendle 2020). Scoring 87k items for ~5k users is
cheap, so there is no reason to sample.

**Baselines.**
- Global popularity: positives over all history.
- Recent popularity: positives in the last N days before the cutoff; N in {7, 30, 90, 365} picked on
  val (30 days won).
- ALS (implicit library, confidence weighting on positives). Grid over factors, regularisation,
  alpha, and **how much history to train on**. Training on only the last 3 years beat training on
  all 28 years (val Recall@100 0.289 vs 0.277), because tastes and the catalog drift: a model fit
  mostly on 1990s-2010s ratings under-weights what people watch in 2023.

**Finding worth knowing.** Under a global time split, recent popularity is a very strong baseline,
and on test ALS only ties it on Recall@100 and loses on NDCG@10. This is a well-known effect of
realistic temporal evaluation (much of the next-quarter activity goes to new and trending titles),
and it is the bar the two-tower model has to clear. ALS does recommend far less popular items
(popularity bias and coverage in `results/m1_baselines.json`).

**Cold users** (no positives before the window) get the recent-popularity list from every model, so
model differences come from warm users; `metrics_warm` in the results isolates them.

**Protocol.** Hyperparameters are chosen on val (models fit on train). For test, the chosen config is
refit on train + val. Nothing is tuned on test.
