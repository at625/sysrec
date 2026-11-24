#!/usr/bin/env bash
# M2 main model -> M3 rankers -> M4 re-ranking -> M2 FAISS benchmark -> M2 ablations
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/train_two_tower.py --only two_tower --out m2_two_tower.json
python scripts/train_rankers.py
python scripts/run_rerank.py
python scripts/build_index.py
python scripts/train_two_tower.py --only two_tower two_tower_no_logq two_tower_mixed_negatives --out m2_ablations.json
