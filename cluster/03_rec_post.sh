#!/bin/bash
# Stage 3/3 — rec env. Normalize sentiment, trust signals, graph, baselines,
# ST-HGRec, ablations. Copies everything worth keeping to RUN_DIR at the end,
# outside this job's ephemeral working directory.
# Usage: CLUSTER_USER=yourname RUN_ID=myrun bash cluster/03_rec_post.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source cluster/config.sh
cd "$REPO_ROOT"

source env-rec/bin/activate  # created by 01_rec_pre.sh — must have already run

python -m src.sentiment.normalize_aspects
python -m src.data.trust_signals
python -m src.graph.build_graph

python -m src.eval.run_baselines --config configs/popularity.yaml
python -m src.eval.run_baselines --config configs/bpr.yaml
python -m src.eval.run_baselines --config configs/lightgcn.yaml
python -m src.eval.run_baselines --config configs/ngcf.yaml

python -m src.eval.run_sthgrec --config configs/sthgrec.yaml --check-determinism
python -m src.eval.run_sthgrec --config configs/sthgrec.yaml
python -m src.eval.ablations --config configs/sthgrec.yaml

deactivate

# --- copy everything worth keeping out to persistent storage ---
mkdir -p "$RUN_DIR"
cp -r results "$RUN_DIR/"
cp data/interim/absa_normalized.csv data/interim/trust_signals.csv "$RUN_DIR/" 2>/dev/null || true

echo "Stage 3 (rec-post) done. Full pipeline complete."
echo "Results saved to: $RUN_DIR"
ls -la "$RUN_DIR"
