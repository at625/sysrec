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

## M2: Two-tower retrieval and ANN

**Model** (`recsys/retrieval/two_tower.py`).
- *User tower*: user-ID embedding (row 0 is the unknown-user bucket) + mean-pooled ID embeddings
  of the user's last 50 positives + two point-in-time stats (log #ratings so far, mean rating so
  far) -> 256-unit MLP -> 64-d, L2-normalised.
- *Item tower*: item-ID embedding (shared with the history pooling) + genre multi-hot + release-year
  bucket embedding + frozen 384-d MiniLM embedding of "title | genres | top-10 tags" -> MLP -> 64-d,
  L2-normalised. Tags are only those applied before the cutoff.
- Score = cosine / temperature (0.05).

**Training** (`recsys/retrieval/train_two_tower.py`). Each positive (rating >= 4) in the last 3
years before the cutoff is one example (about 1.8M for val). Its history is the user's *earlier*
positives (`build_sequences` + `gather_history`), so the target never leaks into its own input.
Loss = in-batch sampled softmax (batch 4096) with logQ correction, where q is the item's frequency
among training targets; other in-batch columns holding the same item as the row's positive are
masked so they don't count as negatives. Adam, lr 1e-3, 5 epochs, about 2.0 to 2.5 minutes per epoch
on the 4-vCPU cloud machine (per-epoch times in `results/m2_two_tower.json`).

**Results (test window; numbers from `results/m2_two_tower.json`, `results/m1_baselines.json`).**

| model | Recall@100 | NDCG@10 |
|---|---|---|
| recent popular (30d) | 0.284 | 0.172 |
| ALS (3y history) | 0.279 | 0.159 |
| two-tower | **0.298** | 0.162 |

The two-tower model beats ALS and recent popularity on Recall@100, the metric that matters for a
candidate generator, but not on NDCG@10: it is a good *recall* model and a mediocre *ordering*
model, which is exactly the job split between retrieval and ranking.

**Ablations** (`results/m2_ablations.json`, test Recall@100):
- logQ off: 0.132 (from 0.298). Without the correction, in-batch negatives are sampled in proportion
  to popularity, so the model learns to push popular items down and recommends the long tail. This is
  the single most important trick in the model.
- Mixed negatives (in-batch + 4096 uniform per batch, with the mixture's q in the correction):
  0.301, a small gain at about 2x the item-tower compute per step. Kept off by default for speed.

**Val tuning** (`results/m2_tuning.json`, run with 3 epochs as the base): 5 epochs +0.003 Recall@100
(adopted); 6 years of targets +0.004 but 2x training time (not adopted); temperature 0.1 -0.006;
dropping the user-ID embedding -0.005. No setting was evaluated on test before being chosen.

**FAISS** (`results/m2_faiss.json`, `docs/figures/faiss_tradeoff.png`; 87,585 items, 64-d, 2,000
test-user queries, batch 1, 1 thread, on the cloud Xeon). Exact `IndexFlatIP` takes ~0.85 ms p50.
HNSW (M=32) at efSearch=128 reaches 0.988 recall@100 vs exact at 0.07 ms p50 (about 12x faster);
IVF-256 at nprobe=16 gets 0.977 at 0.05 ms. At this catalog size exact search is already cheap, so
the ANN choice matters less than it would at 10M items; the benchmark is there to show the trade-off
curve and the knob (efSearch / nprobe) you'd turn. Serving uses HNSW with efSearch raised to cover
the over-fetch needed to drop already-seen items.

**Alternatives considered.** Attention pooling or a SASRec user tower (stretch goal); training on
all 28 years of history (slower, and stale tastes hurt ALS); sampled softmax with a separate
popularity-based sampler instead of in-batch (more machinery for similar results).
