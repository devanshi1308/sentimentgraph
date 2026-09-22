# Running on the SVKM AI/ML cluster

Scripts here run the pipeline on the SVKM AI/ML cluster (PBS Professional,
fronted by the Altair Access web portal at `https://10.126.1.10:4443`; see
`Altair access -User Manual.pdf`). The cluster's GPU node has 2x NVIDIA H100
80GB — for this repo's model sizes, one GPU (or one MIG slice) is plenty; see
`GPU.md` for why (embedding table ~15 MB, adjacencies well under 1 GB).

The pipeline is split into **three stages** because it needs two Python envs
that must never share a process (`env-rec` / `env-absa` — see `README.md`):

| Script | Env | What it does |
| --- | --- | --- |
| `01_rec_pre.sh` | env-rec | build both venvs, download raw data, build atomic files |
| `02_absa.sh` | env-absa | full ABSA extraction (the long pole) |
| `03_rec_post.sh` | env-rec | normalize sentiment, trust signals, graph, baselines, ST-HGRec, ablations — then copies `results/` out |

All three read `cluster/config.sh` for `CLUSTER_USER` (your cluster username —
determines the `/data/<user>/...` paths) and `RUN_ID` (groups one pipeline
run's output together; defaults to a timestamp if unset).

**Where results actually end up:** each stage's job working directory is
ephemeral (Altair/PBS gives every job its own `pbs.<jobid>.../` folder that
you shouldn't rely on surviving). `03_rec_post.sh` explicitly `cp -r`s
`results/` plus the normalized-aspect and trust-signal CSVs to
`/data/<user>/st-hgrec-results/<run-id>/` at the end — that's the persistent,
external location. Don't rely on Altair's "Output Directory" form field alone
for this; use it in addition (see below), not instead.

## Path A — SSH access to the cluster (recommended if you have it)

```bash
git clone https://github.com/devanshi1308/sentimentgraph.git /data/<you>/sentimentgraph
cd /data/<you>/sentimentgraph
CLUSTER_USER=<you> bash cluster/submit_all.sh
```

That submits all three stages as one PBS dependency chain (02 waits for 01,
03 waits for 02) and prints the job IDs plus the results path. Track with
`qstat -u $USER`; `tail -f` the `.out`/`.err` files it prints for live logs.

To resubmit just the long ABSA stage after an interruption (it's
checkpointed — safe to resume):
```bash
qsub -v CLUSTER_USER=<you>,RUN_ID=<same-run-id> cluster/02_absa.pbs
```

**Before your first run:** confirm the queue name and GPU resource syntax
(`-q workq`, `select=1:ngpus=1`) in the `.pbs` files with the cluster admin —
those are placeholders from the Altair Access web form's shown defaults, not
verified against this cluster's actual PBS config.

## Path B — Altair Access web portal only (no SSH)

### Path B1 — Jupyter notebook (recommended if you only have the portal)

`sthgrec_pipeline.ipynb` runs the whole pipeline as notebook cells instead of
Shell Script jobs — easier to watch, debug, and rerun one stage at a time:

1. **Applications → Jupyter** (Container Image `pytorch_pbs:23.06-py3`,
   GPUs per Node `1`). Submit, wait for Running, open via **Custom Actions →
   Show Jupyter URL**.
2. In Jupyter's file browser: **Upload** → select `cluster/sthgrec_pipeline.ipynb`
   from your local clone of this repo → open it.
3. Edit the `CLUSTER_USER = "CHANGE_ME"` cell near the top to your Altair
   Access username, then **Run → Run All Cells** (or run them one at a time —
   each stage is its own cell/section, so you can stop and resume).
4. The last cell prints where results landed under
   `/data/<you>/st-hgrec-results/<run-id>/` — download that folder from the
   **Files** tab when done.

### Path B2 — Shell Script jobs (no interactive session needed)

1. Log in at `https://10.126.1.10:4443`.
2. Get the repo onto cluster storage: submit a throwaway **Jupyter** job
   (any container, e.g. `pytorch_pbs:23.06-py3`), open its terminal, and
   `git clone https://github.com/devanshi1308/sentimentgraph.git` into
   `/data/<you>/`. Terminate that job once cloned.
3. For each stage, in order — **Applications → Shell Script**:
   - Container Image: `pytorch_pbs:23.06-py3`
   - Number of GPUs per Node: `1` (stage 1 doesn't need one, but it's cheap
     to leave uniform)
   - Job Script: the corresponding `cluster/0N_*.sh` file
   - **Output Directory**: set to `/data/<you>/st-hgrec-results` (not
     `[Default]`) — this is Altair's own answer to "save results elsewhere";
     the script's own `cp` at the end of stage 3 is the guaranteed path
   - Before submitting stage 1, export `CLUSTER_USER` and `RUN_ID` so the
     script can see them — if the form has no plain env-var field, hardcode
     them at the top of your uploaded copy of `cluster/config.sh` instead of
     relying on the `CHANGE_ME` default
   - Submit Job, wait for it to reach Completed in the Jobs list, then submit
     the next stage the same way, reusing the same `RUN_ID`

## Notes

- Run `bash cluster/01_rec_pre.sh` once per account — it creates `env-rec` in
  the repo and skips recreating it on later runs (`if [ ! -d env-rec ]`), so
  reruns of stage 3 alone don't rebuild it.
- `set -euo pipefail` in every script means the job stops at the first error
  instead of silently continuing — check the `.err` log if a stage fails.
- These are plain bash, so nothing here applies to a Windows machine; that's
  covered separately in `GPU.md` / `README.md` for local (non-cluster) runs.
