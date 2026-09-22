#!/bin/bash
# Shared settings for every cluster/*.sh stage script. Edit CLUSTER_USER once;
# everything else derives from it. Sourced, never executed directly.

CLUSTER_USER="${CLUSTER_USER:-CHANGE_ME}"

REPO_ROOT="/data/${CLUSTER_USER}/sentimentgraph"

# Where finished results are copied, OUTSIDE the ephemeral per-job working
# directory Altair/PBS gives each job. Survives the job ending / being
# cleaned up. One subfolder per pipeline run, grouped by RUN_ID so the three
# staged jobs (rec-pre / absa / rec-post) that make up one run land together.
RESULTS_ROOT="/data/${CLUSTER_USER}/st-hgrec-results"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"

if [ "$CLUSTER_USER" = "CHANGE_ME" ]; then
    echo "cluster/config.sh: set CLUSTER_USER (export CLUSTER_USER=yourname, or edit this file) before running any stage script." >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
