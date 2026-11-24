#!/usr/bin/env bash
# One-command reproduction of every result in results/ (a few hours on a 4-vCPU cloud VM).
# Requires: Python 3.11 venv with requirements.txt; Docker for the serving/load-test part.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/env_check.py
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/data_report.py
python scripts/train_baselines.py                      # M1
HF_HUB_DISABLE_XET=1 python scripts/build_text_embeddings.py
python scripts/train_two_tower.py --tuning             # M2 val sweep
python scripts/train_two_tower.py --only two_tower --out m2_two_tower.json
python scripts/build_index.py
python scripts/train_two_tower.py --only two_tower two_tower_no_logq two_tower_mixed_negatives --out m2_ablations.json
python scripts/train_rankers.py                        # M3
python scripts/run_rerank.py                           # M4
python scripts/build_serving.py                        # M5 snapshot + offline reference
python scripts/run_ab_sim.py                           # M6
if docker info >/dev/null 2>&1; then
  INDEX=flat docker compose up -d --build
  until curl -sf localhost:8000/health >/dev/null; do sleep 2; done
  SERVICE_URL=http://localhost:8000 pytest -m integration
  python scripts/check_parity.py
  docker compose down
else
  echo "Docker not running: skipping M5 compose, integration tests and load tests"
fi
