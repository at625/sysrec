# PROGRESS

Read this first at the start of every session and resume from "Next".

## Hardware actually used (overrides the MacBook notes in CLAUDE.md)

The build runs in a **Claude Code cloud container**, not the MacBook Air:
Intel Xeon @ 2.1 to 2.3 GHz, **4 vCPU, ~15.7 GB RAM, no GPU** (no CUDA, no MPS), ~50 GB free disk,
Ubuntu 24.04, Python 3.11 (uv venv). Every timing/latency result records the exact hardware in
`results/*.json -> meta.hardware`. `recsys/utils/device.py:get_device()` picks CUDA > MPS > CPU, so
the same code uses the Apple GPU on the Mac.

Environment quirks found during setup:
- GroupLens' TLS chain ends at the emSign root, which this container's trust store lacks;
  `scripts/download_data.py` uses certifi's bundle (which has it). MD5 is verified either way.
- Hugging Face downloads need `HF_HUB_DISABLE_XET=1` behind this proxy.
- PyPI's Linux torch wheel ships CUDA libs (big venv, ~6 GB); CI installs the CPU-only wheel.
- Docker CLI exists but the daemon is not running at M0; revisit at M5.

## Milestones

- [x] **M0 Setup + data**
  - [x] Scaffold, pinned `requirements.txt`, ruff + pytest, GitHub Actions (CPU, synthetic data)
  - [x] `scripts/env_check.py` (PASSED; Docker daemon not running, reported as optional)
  - [x] `scripts/download_data.py` (MD5 d472be332d4daa821edc399621853b57 verified)
  - [x] `scripts/prepare_data.py`: CSV -> Parquet, int32 IDs (32,000,204 ratings; peak RSS 3.2 GB, 5 s)
  - [x] `scripts/data_report.py` -> `docs/data.md`
  - [x] Time split in `configs/default.yaml`; leakage + reproducibility tests pass
- [ ] M1 Evaluation harness + baselines
- [ ] M2 Two-tower retrieval + ANN
- [ ] M3 Features + rankers
- [ ] M4 Re-ranking
- [ ] M5 Serving
- [ ] M6 Simulated A/B
- [ ] M7 Write-up

## Scale-downs and deviations (applied to every model equally)

- None yet. Full dataset fits in memory for preparation and reporting.

## Next

- M1: metrics + harness, popularity, recent popularity, ALS.

## Open problems

- None.
