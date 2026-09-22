#!/bin/bash
# Submits all three pipeline stages as one dependency chain: 02 waits for 01
# to finish successfully, 03 waits for 02. Requires SSH access to a node with
# `qsub` (PBS Professional CLI) — if you only have the Altair Access web
# portal, use the Shell Script app on each cluster/0N_*.sh instead; see
# cluster/README.md.
#
# Usage: CLUSTER_USER=yourname bash cluster/submit_all.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source cluster/config.sh   # validates CLUSTER_USER, computes/prints RUN_ID, makes RUN_DIR

mkdir -p "$RUN_DIR/pbslogs"
V="CLUSTER_USER=${CLUSTER_USER},RUN_ID=${RUN_ID}"

jid1=$(qsub -v "$V" -o "$RUN_DIR/pbslogs/01.out" -e "$RUN_DIR/pbslogs/01.err" cluster/01_rec_pre.pbs)
echo "01_rec_pre  -> $jid1"

jid2=$(qsub -v "$V" -W depend=afterok:"$jid1" -o "$RUN_DIR/pbslogs/02.out" -e "$RUN_DIR/pbslogs/02.err" cluster/02_absa.pbs)
echo "02_absa     -> $jid2 (after $jid1)"

jid3=$(qsub -v "$V" -W depend=afterok:"$jid2" -o "$RUN_DIR/pbslogs/03.out" -e "$RUN_DIR/pbslogs/03.err" cluster/03_rec_post.pbs)
echo "03_rec_post -> $jid3 (after $jid2)"

echo
echo "RUN_ID=$RUN_ID"
echo "Results will land in: $RUN_DIR"
echo "Check status with: qstat -u \$USER"
