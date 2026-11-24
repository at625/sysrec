# Project: an end-to-end, multi-stage recommender system

## Who I am and why this exists

I'm a CS student (graduating May 2027) recruiting for new-grad SWE / MLE / AI roles. This project goes on my resume. **You (Claude Code) are building the entire project end to end.** Work autonomously through every milestone below, and leave behind clear documentation so I can understand and explain every design decision in interviews.

The goal: a production-style recommender that mirrors how large tech companies actually build recommendations. That means **candidate retrieval (two-tower + ANN) → ranking (multi-task deep model) → re-ranking (diversity/business rules) → low-latency serving**, with a rigorous offline evaluation harness and a simulated A/B test. Correctness of evaluation matters more than flashy models: no data leakage, time-based splits, full-catalog metrics.

## My environment

- Machine: **MacBook Air 13-inch (M4, 2025)**, Apple Silicon (arm64), fanless
- Chip / accelerator: **Apple M4**. No NVIDIA GPU, so no CUDA. Use PyTorch's **MPS** backend (Apple GPU), with CPU fallback.
- Memory: **16 GB unified RAM** (shared by CPU, GPU, Docker, and everything else)
- OS: **macOS** (check the exact version with `sw_vers` during M0)
- Disk: check free space with `df -h` during M0; the project needs roughly 30 GB free for raw data, Parquet, features, embeddings, indexes, and Docker images. Warn me if there's less.
- Python: **3.11**, managed with `uv` (install it with Homebrew if it's missing)
- Docker: **yes**, Docker Desktop for Mac (it must be running for M5)
- Package manager: Homebrew (ask me before installing anything with `brew` that needs my password)

## Platform notes: Apple Silicon, 16 GB RAM (follow these throughout)

- **Device selection:** write one `get_device()` helper that picks `mps` if `torch.backends.mps.is_available()`, otherwise `cpu`, overridable from config. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` so any unsupported ops fall back to CPU. If MPS gives wrong or unstable results for some op, use CPU for that part and document it. Use float32 on MPS (avoid float64, which MPS doesn't support).
- **Memory budget is the main constraint.** 16 GB is shared with macOS and Docker, so aim to keep the Python process under about 8–10 GB:
  - Use Polars **lazy** queries and streaming where possible; avoid materializing big pandas copies.
  - Store IDs as `int32`, ratings as `float32` or `int8`, and timestamps as `int64` only where needed. Remap user and item IDs to contiguous integers.
  - Ranking training data (examples × retrieval candidates) can explode in size. Cap it by sampling training examples and limiting candidates per example (configurable), and write it to Parquet in chunks.
  - Use a small development subsample (e.g., 10–20% of users, configurable) for fast iteration, then run final reported results on the full dataset if it fits. If the full dataset doesn't fit in a step, use the largest subsample that does **consistently for every model**, and state it clearly in the README.
- **Libraries on macOS arm64:** use `faiss-cpu` from pip (there's no FAISS GPU on Mac). LightGBM may need `libomp` (`brew install libomp`) if the wheel fails to import. ONNX Runtime and TorchScript both work on arm64. Prefer prebuilt wheels; don't compile big C++ projects from source.
- **Long runs:** the MacBook Air has no fan and slows down under sustained load, so expect training to be slower than the benchmarks you'd see online. Keep epochs and embedding sizes modest, checkpoint every epoch so runs can resume, and run long jobs with `caffeinate -i` so the Mac doesn't sleep. Log wall-clock times so speed results are honest about the hardware.
- **Docker on Mac:** Docker Desktop has its own memory limit. Keep the Compose stack lean (API + Redis), with Redis `maxmemory` set, and note the Docker Desktop memory setting I need if the default is too low. Images must be built for `linux/arm64`.
- **CI** runs on GitHub's Linux runners, so keep code platform-independent (no macOS-only paths or MPS-only code paths in tests; tests run on CPU).
- **Latency numbers** measured on a MacBook Air are laptop numbers. Label them with the hardware in results and in the README.

## How I want you to work (follow these every session)

1. **Build it all, autonomously.** Work through M0 → M7 in order without waiting for my approval between steps. Write the code, run it, debug it, and fix failures yourself. Stop and ask me only if you're blocked on something only I can do: hardware, credentials, installing system packages that need my password, or a decision that can't be undone.
2. **Track progress in `PROGRESS.md`.** Keep a checklist of milestones and sub-tasks, updated as you go, with notes on what's done, what's next, and any open problems. At the start of every session, read `PROGRESS.md` and resume from where you left off.
3. **Tests gate each milestone.** A milestone is done only when its acceptance criteria below pass. Don't mark a milestone complete or move on while anything is failing.
4. **Never fabricate or estimate numbers.** Every metric in the README or in my resume bullets must come from a script that actually ran and wrote raw results to `results/*.json`, with the dataset version, split definition, random seed, hardware, library versions, and git commit hash recorded. If something hasn't been measured, write "not measured."
5. **No data leakage, ever.** All splits are by time. Every feature must be computed **point-in-time correct**: only data from before the prediction timestamp. Write tests that check this. If a metric looks suspiciously good, investigate leakage before celebrating.
6. **Don't copy existing recsys codebases** (e.g., TorchRec examples, RecBole, Merlin). Use the papers below for concepts, and write original, readable code. Libraries for building blocks are fine (PyTorch, FAISS, LightGBM, Polars, Redis).
7. **Prefer clarity over cleverness.** Type hints, docstrings on non-obvious functions, small modules, and config-driven experiments (YAML or dataclasses), so every result can be reproduced with one command. Make a small, descriptive git commit after each working step.
8. **Document as you build.** At the end of each milestone, add a section to `docs/design.md` explaining what you built, why, the alternatives you considered, and the trade-offs.
9. If something is too slow or too big for my hardware, scale it down sensibly (subsample users, smaller embeddings, fewer epochs), note the change in `PROGRESS.md`, and keep going. Always report results on the same scaled dataset for every model so comparisons stay fair.

## Reference reading (use for concepts, not code)

- Covington et al., *Deep Neural Networks for YouTube Recommendations* (RecSys 2016): the retrieval → ranking architecture
- Yi et al., *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations* (RecSys 2019): two-tower with in-batch negatives and logQ correction
- Wang et al., *DCN V2: Improved Deep & Cross Network* (WWW 2021)
- Naumov et al., *Deep Learning Recommendation Model (DLRM)* (2019)
- Krichene & Rendle, *On Sampled Metrics for Item Recommendation* (KDD 2020): why we use full-catalog metrics
- Kang & McAuley, *Self-Attentive Sequential Recommendation (SASRec)* (ICDM 2018): for the stretch goal
- Carbonell & Goldstein, *The Use of MMR, Diversity-Based Reranking* (SIGIR 1998)

## Tech stack

- Data: **Polars** (or DuckDB) + Parquet for the offline pipeline
- Models: **PyTorch** for the two-tower and ranker; **LightGBM** (LambdaRank) as a strong ranking baseline; the `implicit` library or your own matrix factorization for a classic CF baseline
- Text features: a small sentence-embedding model (e.g., `all-MiniLM-L6-v2`) to embed item titles/tags, computed once offline
- ANN: **FAISS** (compare exact `IndexFlatIP` vs `IVF`/`HNSW`)
- Online feature store: **Redis**
- Serving: **FastAPI** (async), model exported with TorchScript or ONNX Runtime; Docker Compose for API + Redis
- Observability: a Prometheus `/metrics` endpoint with per-stage latency histograms
- Tooling: `pytest`, `ruff`, pinned `requirements.txt` (or `uv`), `matplotlib` for plots, GitHub Actions running unit tests on CPU with a tiny synthetic dataset

## Data

- Primary dataset: **MovieLens-32M** from GroupLens (ratings with timestamps, movie titles/genres, user tags). Write `scripts/download_data.py` to fetch it and verify the checksum if one is published.
- Treat it as **implicit feedback** for retrieval: a positive interaction is a rating ≥ 4 (make the threshold a config value). Keep the raw rating as a label for the ranker's second task.
- **Time-based split:** pick global cutoff timestamps so that train < validation < test in time (e.g., the last ~N weeks of data for test, the weeks before that for validation). Evaluate on users who have at least one positive in the eval window. Document the exact cutoffs.
- Items that appear only after the training cutoff are cold-start items; report how many there are and how the system handles them.
- Build a tiny synthetic dataset generator for fast unit tests and CI.

## Repo layout (suggested; adjust if you have a good reason)

```
recsys/
  CLAUDE.md
  PROGRESS.md
  README.md
  configs/                  # YAML experiment configs
  docs/                     # design.md, code-walkthrough.md, interview-prep.md, resume-bullets.md
  scripts/                  # download_data.py, build_features.py, train_*.py, build_index.py
  recsys/
    data/                   # loading, splits, negative sampling, datasets/dataloaders
    features/               # point-in-time feature computation, feature definitions
    retrieval/              # baselines, two-tower model, training loop, FAISS index
    ranking/                # LightGBM baseline, DCN-v2 multi-task ranker
    rerank/                 # MMR diversity, business rules
    eval/                   # metrics, evaluation harness, A/B simulation, bootstrap CIs
    serving/                # FastAPI app, Redis feature client, model loading, fallbacks
  benchmarks/               # load generator, latency benchmarks, plotting
  results/                  # raw JSON results (committed)
  tests/
  docker-compose.yml
```

## Milestones and acceptance criteria

### M0: Setup + data
- Repo scaffold, pinned deps, `ruff` + `pytest`, GitHub Actions running CPU unit tests on the synthetic dataset.
- Environment check script (`scripts/env_check.py`): prints macOS version, free disk, RAM, Python/torch versions, whether MPS works (with a tiny matmul on MPS vs CPU), and whether FAISS, LightGBM, Polars, and ONNX Runtime import correctly. Confirms Docker is running.
- Data download script, conversion to Parquet, and a data report (`docs/data.md`): number of users, items, and interactions; sparsity; rating distribution; activity long tail; interactions over time.
- Time-based train/val/test split with the cutoffs recorded in config.
- **Done when:** the env check passes, data loads within the memory budget (log peak memory), the split is reproducible from the config, and tests confirm that no test-window interaction appears in training data.

### M1: Evaluation harness + baselines
- Metrics: Recall@K, NDCG@K, HitRate@K, and MRR for K ∈ {10, 50, 100, 500}, plus catalog coverage and popularity bias (mean popularity of recommended items). **Full-catalog ranking, no sampled negatives.** Exclude items the user already interacted with in training.
- Baselines: (a) global popularity, (b) recent popularity (last N days before the cutoff), (c) matrix factorization / ALS on implicit feedback.
- **Done when:** metric functions are unit-tested against hand-computed examples, and all baselines are evaluated on validation and test with results saved to `results/`.

### M2: Two-tower retrieval + ANN index
- **User tower:** user ID embedding (with hashing or an "unknown" bucket) plus a pooled embedding of the user's recent positive items before the cutoff (mean or attention pooling), plus simple user features.
- **Item tower:** item ID embedding plus genre multi-hot, release year bucket, and the precomputed text embedding of title + top tags.
- Train with **in-batch softmax negatives and logQ (sampling-bias) correction**, using temperature-scaled dot product with L2-normalized outputs. Try mixed negatives (in-batch + uniformly sampled) as an ablation.
- Build FAISS indexes over all item embeddings. Benchmark exact vs IVF vs HNSW: recall-vs-exact and query latency (p50/p99) at several settings.
- **Done when:** the two-tower model is trained and evaluated with the M1 harness against all baselines, with results for the logQ on/off and negative-sampling ablations, plus a FAISS recall/latency trade-off plot. If two-tower doesn't beat ALS on Recall@100, debug it (learning rate, temperature, leakage, feature bugs) and document what you found. Don't hide it.

### M3: Feature pipeline + ranking model
- **Ranking training data:** for each (user, timestamp) training example, take the top-K candidates from the retrieval model (not random items) as negatives, so the ranker learns to separate items that look similar. Label positives from actual future interactions.
- **Features (all point-in-time correct):** user activity stats, user-genre affinity, item popularity over several time windows, item average rating, item age, recency of the user's last interaction with similar items, the retrieval score, and the user-item embedding dot product.
- Write features offline to Parquet; write a loader that pushes the latest feature snapshot into Redis for serving.
- **Rankers:** (a) LightGBM LambdaRank baseline, (b) **DCN-v2 multi-task model** with two heads, P(positive) and predicted rating, combined into a final score with a configurable weight.
- Evaluate end to end: retrieve top-K, rank, and measure NDCG@10 / Recall@10 on the test window.
- **Done when:** point-in-time correctness tests pass (features for an example never use data at or after its timestamp), both rankers beat retrieval-only ordering on NDCG@10 (or the failure is documented and explained), and a feature-importance / ablation table is saved.

### M4: Re-ranking
- MMR-style diversity re-ranking using item embedding similarity, plus a configurable rule such as max N items per genre in the top 10.
- Metrics: intra-list diversity, genre coverage, and catalog coverage, alongside NDCG.
- **Done when:** a plot shows the NDCG vs diversity trade-off curve across MMR λ values, and a default operating point is chosen with the justification documented.

### M5: Online serving
- FastAPI service: `GET /recommend?user_id=...&k=...` runs ANN retrieval → Redis feature fetch → batched ranker inference (TorchScript/ONNX) → re-rank → response with item IDs and scores.
- Cold-start fallbacks: an unknown user gets recent-popular items (or genre-popular, if genre preferences are provided); unknown items in history are skipped safely.
- Docker Compose brings up the API and Redis with one command. Add a `/health` endpoint and Prometheus metrics with latency per stage.
- Load test with a load generator at several request rates; report throughput and p50/p95/p99 end to end and per stage. Optimize the biggest bottleneck at least once (e.g., batching, caching user embeddings, a smaller FAISS `nprobe`) and record before/after numbers.
- **Done when:** integration tests hit the running service, the online results for a sample of users match the offline pipeline's results for the same model snapshot, and latency results are saved.

### M6: Simulated A/B test
- Compare variants on the held-out test window: (A) recent-popular, (B) ALS, (C) two-tower only, (D) two-tower + ranker, (E) D + diversity re-ranking.
- Report each metric with **bootstrap 95% confidence intervals** over users, and use paired tests (e.g., paired bootstrap) between variants.
- Clearly document that this is an **offline replay simulation, not a live A/B test**, and what it can't tell you (feedback loops, position bias, user response to diversity).
- **Done when:** `scripts/run_ab_sim.py` reproduces the comparison table and a plot from scratch.

### M7: Write-up
- README: the problem, an architecture diagram (retrieval → ranking → re-ranking → serving), how to run everything (including one-command reproduction), a results table with CIs, latency results, limitations, and future work.
- `docs/design.md`: key design decisions and trade-offs, finalized and cleaned up.
- `docs/code-walkthrough.md`: a guided tour following one request through the system (API → retrieval → features → ranker → re-rank → response), plus a tour of the offline training pipeline, pointing to specific files and functions.
- `docs/interview-prep.md`: for each milestone, the core concept explained simply, 5 likely interview questions with strong answers grounded in this codebase and its measured results (include ML system design questions like "design YouTube recommendations" mapped onto this project), and the honest limitations and "what I'd do next."
- `docs/resume-bullets.md`: 3 resume bullets in the format `Built X ... achieving Y ...`, using **only measured numbers** from `results/`, each with a note on which results file backs it.

## Stretch goals (only after M7 is done)
- Re-run the final training on a rented cloud GPU (if I choose to), to report full-dataset results with larger models
- SASRec-style transformer user tower over the interaction sequence, compared against mean pooling
- Port the feature pipeline to PySpark (local mode) and compare runtime
- Streaming real-time features (e.g., Redpanda/Kafka consumer updating Redis) with a "user just watched X" demo
- Kubernetes manifests with horizontal autoscaling for the API
- Model and data drift monitoring with a simple dashboard
- A second dataset (e.g., an Amazon Reviews 2023 category) to show the pipeline generalizes

## Explicitly out of scope
- Live user traffic or a real online A/B test
- A frontend beyond a minimal demo page (optional)
- Chasing state-of-the-art leaderboard numbers. The goal is a correct, well-evaluated, explainable system with honest measurements.
