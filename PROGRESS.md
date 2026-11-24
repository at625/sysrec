# PROGRESS

Read this first at the start of every session and resume from "Next".

## Hardware actually used (overrides the MacBook notes in CLAUDE.md)

Built in a **Claude Code cloud container**, not the MacBook Air: Intel Xeon @ 2.1 to 2.3 GHz,
**4 vCPU, ~15.7 GB RAM, no GPU** (no CUDA, no MPS), ~50 GB free disk, Ubuntu 24.04, Python 3.11 (uv
venv), Docker 29 (daemon started manually with `dockerd`). Every result file records the exact
hardware in `meta.hardware`. `recsys/utils/device.py:get_device()` picks CUDA > MPS > CPU, so the same
code uses the Apple GPU on the Mac.

Environment quirks:
- GroupLens' TLS chain ends at the emSign root, missing from this container's trust store;
  `scripts/download_data.py` uses certifi's bundle (which has it). MD5 is verified either way.
- Hugging Face downloads need `HF_HUB_DISABLE_XET=1` behind this proxy.
- PyPI's Linux torch wheel ships CUDA libs (~6 GB venv); CI installs the CPU-only wheel.
- Docker builds here go through the sandbox proxy: `DOCKER_BUILD_HTTPS_PROXY=http://127.0.0.1:32801`
  and the proxy CA copied to `docker/ca/proxy.crt` (gitignored). On a Mac neither is needed.

## Milestones

- [x] **M0 Setup + data**: scaffold, pinned deps, ruff + pytest, CI; env check; download with MD5;
  Parquet (32,000,204 ratings, peak RSS 3.2 GB); `docs/data.md`; split in config; leakage tests.
- [x] **M1 Evaluation + baselines**: full-catalog metrics (hand-computed tests); popularity,
  recent-popular (30d chosen on val), ALS (3-year history chosen on val). `results/m1_baselines.json`.
- [x] **M2 Two-tower + ANN**: test Recall@100 0.298 vs ALS 0.279. Ablations: logQ off 0.132, mixed
  negatives 0.301. Val sweep in `results/m2_tuning.json`. FAISS flat/IVF/HNSW benchmark + plot.
- [x] **M3 Features + rankers**: point-in-time tests pass; test NDCG@10 retrieval-only 0.162,
  LightGBM 0.206, DCN-v2 0.210. Importance + group ablations + single-task ablation saved.
- [x] **M4 Re-ranking**: lambda sweep x genre cap; default lambda=0.8, no cap (val rule); plot.
- [x] **M5 Serving**: compose (redis + loader + api); integration tests pass; parity 300/300 (exact),
  99% (HNSW); load test baseline ~119 req/s -> 300 req/s at p99 19.6 ms after caching + BLAS fix.
- [x] **M6 Simulated A/B**: `scripts/run_ab_sim.py`; D vs A +22.5% NDCG@10, p < 0.001.
- [x] **M7 Write-up**: README, design.md, code-walkthrough.md, interview-prep.md, resume-bullets.md.

## Scale-downs and deviations (applied to every model equally)

- Full dataset used everywhere (no user subsampling needed).
- Two-tower training targets limited to the last 3 years before each cutoff (history still uses all
  years). A 6-year variant was +0.004 val Recall@100 at 2x cost (`results/m2_tuning.json`).
- Ranker training capped at 20,000 users per snapshot (actual: 4,585 val / 4,076 test, so no cap hit),
  200 candidates per user.
- DCN-v2 head weight chosen on val = 0.0 (rating head unused at inference; documented in design.md).
- Load generator runs on the same 4-vCPU VM as the service, so latency numbers are pessimistic.

## Next (stretch goals, only if asked)

- Candidate mixer to raise candidate recall (currently 42% at 200).
- SASRec user tower; streaming features; second dataset.

## Open problems

- None blocking. Note: `results/m2_tuning.json` was run when the base config used 3 epochs; the
  `two_tower` row there is the 3-epoch model.
