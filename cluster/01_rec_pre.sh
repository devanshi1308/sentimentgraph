#!/bin/bash
# Stage 1/3 — rec env. Raw data + atomic files. Run before 02_absa.sh.
# Usage (Altair Shell Script job, or `bash cluster/01_rec_pre.sh` over SSH):
#   CLUSTER_USER=yourname RUN_ID=myrun bash cluster/01_rec_pre.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source cluster/config.sh
cd "$REPO_ROOT"

if [ ! -d env-rec ]; then
    python -m venv env-rec
    source env-rec/bin/activate
    pip install -r environments/requirements-rec.txt
    pip install --force-reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
else
    source env-rec/bin/activate
fi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python -m src.data.download        # skips files that already exist in data/raw/
python -m src.data.build_atomic

deactivate

echo "Stage 1 (rec-pre) done. RUN_ID=$RUN_ID  RUN_DIR=$RUN_DIR"
echo "Next: bash cluster/02_absa.sh  (with the same RUN_ID)"
