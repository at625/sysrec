Trained artifacts for sysrec, built on a 4-vCPU cloud VM (no GPU) at commit bf7de78, so the system can be served or the pipeline resumed without retraining.

| file | unpacks to | what |
|---|---|---|
| sysrec-serving-snapshot.tar.gz (256 MB) | `artifacts/serving/` | ONNX user tower + DCN-v2 ranker, FAISS flat/HNSW indexes, Redis feature snapshot (cutoff 2023-07-01). Enough for `docker compose up`. |
| sysrec-pipeline-artifacts.tar.gz (519 MB) | `artifacts/ranking/`, `artifacts/*.npy` | Ranking train/eval frames, LightGBM + DCN-v2 models, final lists, MiniLM text embeddings, two-tower item/user embeddings. Inputs for `run_rerank.py`, `run_ab_sim.py`. |
| sysrec-two-tower-checkpoints.tar (1.1 GB) | `artifacts/checkpoints/` | The three main two-tower models (cutoffs 2022-12-31, 2023-04-01, 2023-07-01). `train_rankers.py` and `build_serving.py` load these instead of retraining. |

Restore: `python scripts/fetch_artifacts.py` (serving only) or `python scripts/fetch_artifacts.py --all`. SHA-256 sums are in `scripts/artifacts_v1.sha256`.

Derived from MovieLens 32M (Harper & Konstan, 2015) and redistributed under the same license: non-commercial use only. See MODEL_DATA_LICENSE.txt.
