# Resume bullets

Every number is copied from a results file produced by a script in this repo. Hardware for all
latency/throughput numbers: cloud VM, 4 vCPU Intel Xeon @ 2.3 GHz, no GPU.

1. **Built a four-stage recommender (two-tower retrieval with FAISS ANN, DCN-v2 ranker, MMR
   re-ranking) on MovieLens-32M (32M ratings, 87.6K items), achieving +22.5% NDCG@10 over a
   recent-popularity baseline on a held-out future time window (0.172 to 0.210, paired bootstrap
   p < 0.001, 4,726 users).**
   Backed by `results/m6_ab_sim.json` (variants A and D, `paired_tests["A->D"]["ndcg@10"]`).

2. **Trained a two-tower retrieval model in PyTorch with in-batch sampled softmax and logQ
   sampling-bias correction, reaching 0.298 full-catalog Recall@100 vs 0.279 for ALS; showed logQ
   correction more than doubles recall (0.132 without it), and served it through FAISS HNSW at 98.8%
   of exact-search recall with a 12x faster query (0.07 ms vs 0.85 ms).**
   Backed by `results/m2_two_tower.json`, `results/m1_baselines.json`, `results/m2_ablations.json`
   (test), `results/m2_faiss.json` (hnsw efSearch=128 row vs flat row).

3. **Deployed the pipeline as a FastAPI + Redis + ONNX Runtime service with Docker Compose and
   Prometheus per-stage latency metrics; raised sustained throughput from ~119 to 300 requests/s at
   19.6 ms p99 by caching item features and fixing BLAS thread oversubscription across workers, with
   online results matching the offline pipeline for 300/300 sampled users.**
   Backed by `results/m5_latency.json` (runs `baseline_hnsw_redis_items_1worker` at 200 offered req/s
   achieved 118.9; `opt2_item_cache_3workers_1blas_thread` at 300 req/s) and `results/m5_parity.json`
   (`flat.identical_lists = 1.0`).

Shorter variants if space is tight:
- Built a multi-stage recommender on MovieLens-32M that improves NDCG@10 by 22.5% over a popularity
  baseline on future data (time-based split, full-catalog metrics, bootstrap CIs).
- Served retrieval + ranking + re-ranking at 300 req/s with 19.6 ms p99 on 4 vCPUs (FastAPI, Redis,
  ONNX Runtime, FAISS, Docker).
