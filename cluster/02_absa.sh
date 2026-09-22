#!/bin/bash
# Stage 2/3 — absa env, separate from env-rec (never mix). The long pole:
# checkpointed, safe to interrupt and resubmit this same script to resume.
# Usage: CLUSTER_USER=yourname RUN_ID=myrun bash cluster/02_absa.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source cluster/config.sh
cd "$REPO_ROOT"

if [ ! -d env-absa ]; then
    python -m venv env-absa
    source env-absa/bin/activate
    pip install -r environments/requirements-absa.txt
    pip install --force-reinstall torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    python -m spacy download en_core_web_sm
else
    source env-absa/bin/activate
fi

python -m src.sentiment.run_absa --device auto --batch-size 256

deactivate

mkdir -p "$RUN_DIR"
cp data/interim/absa_full.csv "$RUN_DIR/" 2>/dev/null || true
echo "Stage 2 (absa) done. RUN_ID=$RUN_ID  RUN_DIR=$RUN_DIR"
echo "Next: bash cluster/03_rec_post.sh  (with the same RUN_ID)"
