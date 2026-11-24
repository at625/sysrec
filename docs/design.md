# Design notes

One section per milestone: what was built, why, alternatives considered, and trade-offs. All
numbers come from `results/*.json`; hardware for timings is a 4-vCPU Intel Xeon cloud VM without a GPU.

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

## M3: Point-in-time features and ranking

**Training data** (`recsys/ranking/data.py`, `scripts/train_rankers.py`). The ranker that is
evaluated on window W is trained on a *snapshot* just before W:

| | retrieval models + features computed on | labels from |
|---|---|---|
| ranker for val | history < 2022-12-31 | 2022-12-31 .. 2023-04-01 |
| ranker for test | history < 2023-04-01 | 2023-04-01 .. 2023-07-01 (the val window) |
| test evaluation | history < 2023-07-01 | test window (metrics only) |

For every user with a positive in the label window, the two-tower model's top 200 candidates become
examples (about 0.8 to 0.9M rows per snapshot), labelled positive if the user rated the item >= 4 in the
label window. The negatives are therefore *hard*: items the retriever already thinks are relevant.
Retrieval only covers 42% of the test positives in its top 200 (`candidate_recall@K`), which caps
what any ranker can do.

**Features** (`recsys/features/pit.py`), all computed from history strictly before the snapshot
cutoff, and tested by perturbing every row at or after the cutoff and checking the feature frame is
byte-identical (`tests/test_features.py`):
retrieval score and rank; ALS user-item dot product; user activity (counts, mean rating, recency,
last-30-day activity); item popularity over 7/30/365 days and all time; smoothed average rating;
days since the item's first rating; release age; user-genre affinity (mean and max share of the
user's positives in the item's genres); days since the user last liked an item of the same genre.
They are written to Parquet (`artifacts/ranking/*.parquet`) and, for serving, to Redis.

**Rankers.**
- LightGBM LambdaRank, one query per user, early stopping on 10% of training users.
- DCN-v2 (3 full-rank cross layers + 256-128 MLP) with two heads: P(positive) (BCE) and predicted
  rating (MSE, only on candidates the user actually rated in the label window). Inputs: signed
  log1p for count features, standardisation, explicit missing-indicator per feature. Final score
  = P(pos) + w * rating / 5, with w chosen on val from {0, 0.1, 0.2, 0.5, 1.0}.

**Results** (`results/m3_ranking.json`, test window, same 4,726 users, top 10 from the same 200 candidates):

| ordering | NDCG@10 | Recall@10 |
|---|---|---|
| retrieval only (two-tower) | 0.162 | 0.070 |
| LightGBM LambdaRank | 0.206 | 0.095 |
| DCN-v2 multi-task | **0.210** | **0.098** |

Both rankers beat retrieval-only ordering by about 30% relative NDCG@10, and both beat the recent-popular
baseline (0.172).

**Honest findings.**
- *The rating head does not help ranking here.* Val picked w = 0 (rating head ignored at inference),
  and a single-task DCN (no rating loss) scores 0.188 on val (same as multi-task) and 0.212 on test. The
  rating task is still useful as a product signal (RMSE 0.86 on rated candidates), but it isn't an
  accuracy lever for the top-10. A likely reason: only ~8% of candidate rows have a rating label.
- *Feature importance* (LightGBM gain, test ranker): retrieval rank 23%, ALS score 11%, 7-day and
  30-day popularity 7% and 6%. Group ablations (retrain without a group, test NDCG@10): removing user
  activity costs the most (0.206 to 0.185), then item popularity (to 0.195) and retrieval features (to 0.197).
  Popularity *trend* features matter because the eval window is dominated by what is new this quarter.
- *Why such a big jump from ranking?* I checked for leakage first: features only read
  `ctx.history` (asserted), the perturbation test passes, and the labels come from a window the
  features never see. The gain is plausible because the two-tower model has no notion of *time*
  (it doesn't know what is trending this month), while the ranker sees recent popularity and
  recency. This is also why recent-popular is such a strong baseline.

## M4: Diversity re-ranking

**What** (`recsys/rerank/mmr.py`). Greedy MMR on the DCN scores (min-max normalised per user):
pick the item maximising lambda * relevance - (1 - lambda) * max cosine similarity to already-picked
items, using two-tower item embeddings. Optional business rule: at most N items per genre in the
top 10. Metrics: intra-list diversity (1 - mean pairwise cosine), genre coverage per list, catalog
coverage, NDCG@10 and Recall@10.

**Operating point** (chosen on val by a pre-declared rule in `configs/default.yaml`: the most diverse
setting whose val NDCG@10 is within 1% of no re-ranking): **lambda = 0.8, no genre cap**. On val it
raises ILD@10 from 0.318 to 0.350 (+10%) with NDCG@10 unchanged (0.1877 vs 0.1877). On test (M6) NDCG@10
moves by -0.0003 (paired 95% CI [-0.0009, 0.0005]) while ILD@10 rises from 0.300 to 0.321.
The genre cap (max 3 per genre) costs about 9% NDCG@10 on val for +0.08 genre coverage, so it is
implemented and configurable but off by default. Curve: `docs/figures/rerank_tradeoff.png`.

**Trade-off.** Offline NDCG can only penalise diversity: it rewards predicting what users did when
shown a non-diverse system. The real value of diversity (exploration, satisfaction, less filter-bubble)
needs online measurement, so the chosen point is "free diversity" by the offline metric.

## M5: Online serving

**Design** (`recsys/serving/`; walkthrough in `docs/code-walkthrough.md`). A snapshot is built as of
the test-window start: ONNX user tower + ONNX DCN-v2 (with standardisation baked in), FAISS
flat + HNSW indexes, item arrays, and per-user records. A one-shot `loader` container pushes user
records (200,948 users: history, stats, genre profile, ALS vector, seen items) and item feature
vectors into Redis (`maxmemory` 1.5 GB; measured usage 380 MB). The FastAPI service runs retrieval -> feature fetch ->
ranking -> MMR, exposing `/health` and Prometheus `/metrics` with a per-stage histogram.

**Cold start.** Unknown user IDs get recent-popular items, or genre-popular items with
`?genres=Comedy,Horror`. Known users with no positives get recent-popular *candidates* that are
still ranked. Items unseen before the cutoff have no ID-embedding training signal but still get
embeddings from their genre/year/text features; unknown item IDs never reach the model because the
history is stored as validated indices in the snapshot.

**Parity** (`results/m5_parity.json`). With the exact index, online top-10 lists equal the offline
batch pipeline's lists for **300/300** sampled users; with HNSW, 99% identical, 99.9% top-10 overlap.
Parity is by construction: online code calls the same feature functions and transform as offline.

**Load test** (`results/m5_latency.json`, `docs/figures/latency.png`; open-loop Poisson load,
30 s per rate, cloud VM with 4 vCPU Intel Xeon @ 2.3 GHz, load generator on the same VM, so absolute
numbers are pessimistic):

| config | max sustained rate tested | p50 / p95 / p99 at that rate |
|---|---|---|
| baseline: item features from Redis, 1 worker | 100 req/s (saturates at ~119 at 200 offered) | 9.1 / 24.6 / 37.8 ms |
| #1 item features cached in-process, 1 worker | 200 req/s | 11.2 / 46.8 / 79.7 ms |
| #2 + 3 workers, 1 BLAS thread each | 300 req/s (saturates at 400) | 5.7 / 12.6 / 19.6 ms |

Optimisation #1 removed the Redis `MGET` of 200 item keys per request (item_features stage p50 from
1.06 ms to 0.03 ms), lifting single-worker throughput from ~119 to >= 200 req/s. The first attempt at
#2 (3 workers) made things *worse*: each worker's OpenBLAS used 4 threads, so 12 threads fought over 4
cores and MMR's small matrix product went from 0.4 ms to 11 to 60 ms. Pinning BLAS to one thread per
worker fixed it (recorded as `attempt2_3workers_blas_oversubscribed` in the results).

## M6: Offline replay "A/B" simulation

`scripts/run_ab_sim.py` scores five variants on the same 4,726 test users, with 2,000-sample
bootstrap 95% CIs over users and paired bootstrap tests (`results/m6_ab_sim.json`,
`docs/figures/ab_sim.png`):

| variant | NDCG@10 [95% CI] | Recall@10 [95% CI] | ILD@10 |
|---|---|---|---|
| A recent-popular | 0.172 [0.165, 0.178] | 0.076 [0.072, 0.081] | 0.369 |
| B ALS | 0.159 [0.153, 0.166] | 0.067 [0.063, 0.071] | 0.383 |
| C two-tower | 0.162 [0.155, 0.168] | 0.070 [0.066, 0.074] | 0.313 |
| D two-tower + DCN-v2 | 0.210 [0.203, 0.218] | 0.098 [0.093, 0.103] | 0.300 |
| E D + MMR | 0.210 [0.203, 0.218] | 0.098 [0.093, 0.103] | 0.321 |

Paired: D vs A +0.039 NDCG@10 (+22.5% relative, CI [0.034, 0.043], p < 0.001); D vs C +0.049
(+30%); E vs D -0.0003 (CI [-0.0009, 0.0005], not significant).

**What this is not.** It is a replay of logged behaviour, not an experiment. It cannot capture:
feedback loops (users only rate what the old system showed them; a new system changes that);
position bias and presentation effects; whether users actually like more diverse lists; novelty
effects; long-term retention. Users and items are not randomised, and the logged data reflects
MovieLens' own recommender. A real launch would need an online A/B test with guardrail metrics.
