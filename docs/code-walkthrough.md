# Code walkthrough

Two tours: one request through the online system, then the offline pipeline that produces the
models and features it uses. File references are clickable in most editors.

## Part 1: one request, `GET /recommend?user_id=42&k=10`

1. **HTTP layer**: `recsys/serving/app.py:recommend`. A sync FastAPI endpoint, so it runs in the
   thread pool; the engine's heavy calls (ONNX Runtime, FAISS) release the GIL. It records
   end-to-end latency and per-stage latency in Prometheus histograms (`/metrics`).
2. **User features from Redis**: `recsys/serving/engine.py:Engine.fetch_user`. One `HGETALL u:42`
   returns the user's snapshot record: contiguous index, last-50 positive items, two dense stats,
   activity features, genre profile (share + last time per genre), ALS user vector, and the list of
   already-seen items. All are raw bytes decoded with `np.frombuffer` (no JSON parsing).
   - Unknown user: `Engine.fallback` returns recent-popular items, or genre-popular items when the
     request passes `genres=Comedy,Horror`.
3. **Retrieval**: `Engine.retrieve`.
   - The user tower runs in ONNX Runtime (`user_tower.onnx`, exported from
     `recsys/retrieval/two_tower.py:UserTower`).
   - FAISS (`faiss_hnsw.index`, or exact `faiss_flat.index`) returns the top `K + |seen|` items by
     inner product; seen items are dropped and the top K=200 kept. Over-fetching by `|seen|` makes
     the result identical to the offline "score everything, mask seen, take top K".
   - A user with no positives skips the tower and uses recent-popular candidates, exactly like
     `TwoTowerRecommender.recommend` offline.
4. **Item features**: `Engine.fetch_item_features`. One `MGET` of 200 `i:{item}` keys, or a numpy
   row-gather when `CACHE_ITEM_FEATURES=1` (the optimisation measured in M5).
5. **Ranking**: `Engine.rank`. Builds the same 20 feature columns as the offline
   `recsys/ranking/data.py:build_frame`, using the *same* functions
   (`recsys/features/pit.py:user_item_cross`, `recsys/ranking/transform.py:transform_features`),
   then runs `ranker.onnx` (DCN-v2 + standardisation + head combination, exported in
   `recsys/serving/snapshot.py:_ranker_with_norm`).
6. **Re-ranking**: `Engine.rerank` calls `recsys/rerank/mmr.py:mmr` with the default lambda and
   max-3-per-genre rule chosen in M4.
7. **Response**: movie IDs, scores, and per-stage timings.

Parity: `scripts/build_serving.py` computes the same users' lists with the offline batch code and
stores them in `offline_reference.npz`; `scripts/check_parity.py` and
`tests/test_serving_integration.py::test_online_matches_offline` compare.

## Part 2: the offline pipeline

| Step | Command | Main code |
|---|---|---|
| Download + checksum | `scripts/download_data.py` | |
| CSV -> Parquet, int32 IDs | `scripts/prepare_data.py` | `recsys/data/prepare.py` |
| Data report | `scripts/data_report.py` | |
| Split | (config) | `recsys/data/split.py` |
| Baselines | `scripts/train_baselines.py` | `recsys/eval/harness.py`, `recsys/retrieval/baselines.py` |
| Text embeddings | `scripts/build_text_embeddings.py` | `recsys/features/text.py` |
| Two-tower | `scripts/train_two_tower.py` | `recsys/retrieval/{features,two_tower,train_two_tower}.py` |
| FAISS benchmark | `scripts/build_index.py` | `recsys/retrieval/ann.py` |
| Rankers | `scripts/train_rankers.py` | `recsys/features/pit.py`, `recsys/ranking/*` |
| Re-ranking | `scripts/run_rerank.py` | `recsys/rerank/mmr.py` |
| Serving snapshot | `scripts/build_serving.py` | `recsys/serving/snapshot.py` |
| A/B replay | `scripts/run_ab_sim.py` | `recsys/eval/bootstrap.py` |

Key ideas to look at:

- **The harness contract** (`recsys/eval/harness.py`): `build_context(cfg, cutoff)` returns only
  history strictly before the cutoff; `evaluate` refuses a context whose cutoff differs from the
  window start. Models never receive the evaluation window.
- **Point-in-time training sequences** (`recsys/retrieval/features.py:build_sequences`): each
  training target's history is the user's earlier positives, and the user stats count only ratings
  with a strictly smaller timestamp.
- **Sampled softmax with logQ** (`recsys/retrieval/two_tower.py:sampled_softmax_loss`): in-batch
  negatives, duplicate-positive masking, `- log q` correction, optional uniform negatives.
- **Ranking snapshots** (`recsys/ranking/data.py` docstring): retrieval and features at cutoff c,
  labels from `[c, c + 91 days)`; the ranker evaluated on a window is trained on the snapshot just
  before it.
- **Caching**: two-tower checkpoints are keyed by a hash of config + data + cutoff
  (`TwoTowerRecommender._checkpoint_path`), so later stages reuse exactly the model earlier stages
  evaluated, and an interrupted run resumes from the last finished epoch.
