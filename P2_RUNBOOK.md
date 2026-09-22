# P2 Runbook — Regenerate all artifacts on the corrected data spine

**Who runs this:** Devanshi, on the machine with the envs + disk (this is Phase P2 of
`IMPLEMENTATION_PROMPT.md`). **What it does:** regenerates every pipeline artifact on
the P1-corrected code — official benchmark split preserved, no test leakage, normalized
sentiment. **Why numbers will change:** the old baseline table was measured on a random
8:1:1 re-split; the new one uses the official *temporal* split, which is harder.
**Expect every metric to drop. That is correct, not a regression.**

Read this top-to-bottom once before starting. Steps are ordered; each has an
acceptance check. Anything failing acceptance → stop and ping Muaaz.

---

## 0. Sync + clean stale state  (5 min)

```bash
cd st-hgrec
git switch devanshi && git pull
```

Delete stale artifacts that the new code either replaced or would misread:

```bash
# The combined .inter no longer exists in the pipeline — remove the stale copy
rm -f data/processed/video_games/video_games.inter

# CRITICAL: the ABSA checkpoint indexes rows of the OLD review population.
# The population changed (now train-split users only), so a resume from the old
# checkpoint would silently misalign rows. Start fresh:
rm -f data/interim/absa_full.csv data/interim/absa_full_progress.txt

# The old baseline table mixes random-split rows; the runner APPENDS, so archive it
mv results/baselines/summary.csv results/baselines/summary_randomsplit_archive.csv 2>/dev/null || true

# Old graph + old sample are superseded (both regenerate)
rm -f data/processed/video_games/graph.gpickle data/interim/absa_sample.csv
```

Windows (PowerShell) equivalent:

```powershell
Remove-Item -ErrorAction SilentlyContinue data/processed/video_games/video_games.inter
Remove-Item -ErrorAction SilentlyContinue data/interim/absa_full.csv, data/interim/absa_full_progress.txt
Move-Item -ErrorAction SilentlyContinue results/baselines/summary.csv results/baselines/summary_randomsplit_archive.csv
Remove-Item -ErrorAction SilentlyContinue data/processed/video_games/graph.gpickle, data/interim/absa_sample.csv
```

Note: `data/processed/`, `data/interim/*.csv|*.txt`, and `*.gpickle` are now
**gitignored** — never `git add -f` any of them. The only artifact you'll commit at the
end is `results/baselines/summary.csv` (and small figures).

---

## 1. Rebuild BOTH envs from the updated pins  (10–20 min)

Requirements changed in P0/P1: rec env gained `matplotlib`, `httpx`, `pyarrow`
(+ transitives); absa env gained `pyarrow`, `update-checker`. Rebuild clean:

```bash
# rec env
rm -rf env-rec && uv venv env-rec --python 3.11    # 3.11: ray==2.6.3 ships cp311 wheels
source env-rec/bin/activate
uv pip install -r environments/requirements-rec.txt
python -c "import recbole, matplotlib, httpx, pyarrow, networkx, pandas; print('rec env OK')"
deactivate

# absa env (separate — never mix)
rm -rf env-absa && uv venv env-absa --python 3.11
source env-absa/bin/activate
uv pip install -r environments/requirements-absa.txt
python -c "import pyabsa, pyarrow, update_checker, pandas; print('absa env OK')"
deactivate
```

Windows (PowerShell) — same steps, `rm -rf` -> `Remove-Item -Recurse -Force`,
`source .../activate` -> `.../Activate.ps1`:

```powershell
# rec env
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue env-rec
uv venv env-rec --python 3.11
env-rec\Scripts\Activate.ps1
uv pip install -r environments/requirements-rec.txt
python -c "import recbole, matplotlib, httpx, pyarrow, networkx, pandas; print('rec env OK')"
deactivate

# absa env (separate — never mix)
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue env-absa
uv venv env-absa --python 3.11
env-absa\Scripts\Activate.ps1
uv pip install -r environments/requirements-absa.txt
python -c "import pyabsa, pyarrow, update_checker, pandas; print('absa env OK')"
deactivate
```

**Acceptance:** both `OK` lines print. If a resolver conflict appears, capture the full
error and ping Muaaz — do not hand-upgrade pins.

*(If `uv` isn't installed: macOS/Linux — `curl -LsSf https://astral.sh/uv/install.sh | sh`;
Windows — `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`; or use
`python3.11 -m venv` + `pip install` equivalently on either OS.)*

---

## 2. Raw data  (skip if `data/raw/` already populated; else 30–60 min)

```bash
source env-rec/bin/activate   # Windows: env-rec\Scripts\Activate.ps1
python -m src.data.download
```

`stream_download` skips files that already exist, so re-running is safe.

**Acceptance:** the stats block prints (users, items, interactions ≈2.5–2.6M reviews,
density, meta rows). Save the printout — it's a PRD §5 artifact.

---

## 3. Atomic files — the split-preserving build  (5–10 min)

```bash
python -m src.data.build_atomic
```

This now writes **three** interaction files (`video_games.train.inter`,
`.valid.inter`, `.test.inter`) preserving the official 5-core timestamp split verbatim,
plus `.item`/`.user` built from the union of all splits.

**Acceptance (this is the fix #1 runtime gate):**

```bash
# Row counts of the atomic files must equal the official split CSVs exactly
for s in train valid test; do
  echo "$s: official=$(($(wc -l < data/raw/Video_Games.$s.csv)-1))  atomic=$(($(wc -l < data/processed/video_games/video_games.$s.inter)-1))"
done
```

Cross-platform (works identically on macOS/Linux/Windows — run it instead of
the bash loop above if you're not in bash):

```bash
python -c "
from pathlib import Path
for s in ('train', 'valid', 'test'):
    official = sum(1 for _ in open(f'data/raw/Video_Games.{s}.csv', encoding='utf-8')) - 1
    atomic = sum(1 for _ in open(f'data/processed/video_games/video_games.{s}.inter', encoding='utf-8')) - 1
    print(f'{s}: official={official}  atomic={atomic}')
"
```

Every pair must match ±0. Record the three numbers in your report-back.

---

## 4. Baselines on the honest split  (CPU: hours · GPU: much less)

```bash
python -m src.eval.run_baselines --config configs/popularity.yaml
python -m src.eval.run_baselines --config configs/bpr.yaml
python -m src.eval.run_baselines --config configs/lightgcn.yaml
python -m src.eval.run_baselines --config configs/ngcf.yaml
```

RecBole's startup log now prints the benchmark-file load — **check the logged
train/valid/test sizes match step 3's numbers** (that's the mechanism working).

**Acceptance (fix #1 complete):** fresh `results/baselines/summary.csv` with 4 rows;
sane ordering (LightGCN ≥ BPR/NGCF > Pop); numbers lower than the archived
random-split table (expected). Re-run one model to confirm seed-2020 stability.

---

## 5. Trust signals  (rec env, ~10 min)

```bash
python -m src.data.trust_signals
deactivate
```

(still in the rec env activated in step 1 — Windows: `env-rec\Scripts\Activate.ps1`)

**Acceptance:** `data/interim/trust_signals.csv` exists; the printed population is the
**train-split** user count; top/bottom-10 tables + describe() look sane (trust_score in
(0,1), no NaN). Save the printout.

---

## 6. Full ABSA run  (absa env — the long one: hours on GPU, days on CPU)

```bash
source env-absa/bin/activate   # Windows: env-absa\Scripts\Activate.ps1
python -m src.sentiment.run_absa
```

Batched + checkpointed: safe to interrupt and re-run (it resumes from
`data/interim/absa_full_progress.txt` — which you deleted in step 0, so it starts
clean against the new train population). Run it in `tmux`/`screen` or with `nohup`
on macOS/Linux; on Windows just leave the terminal (or Windows Terminal tab) open,
or launch it detached with
`Start-Process powershell -ArgumentList '-NoExit','-Command','python -m src.sentiment.run_absa'`.

**Acceptance:** completes over the full train-population review set;
`data/interim/absa_full.csv` has columns `user_id,item_id,aspect,sentiment`.

---

## 7. Normalize aspects  (either env, ~1 min)

```bash
python -m src.sentiment.normalize_aspects
deactivate
```

**Acceptance:** `data/interim/absa_normalized.csv` written; the coverage report prints
(matched %, per-canonical counts, top-20 unmatched terms).
**Send the full coverage report to Muaaz** — the unmatched-terms list drives the
vocabulary curation pass, and per-canonical balance feeds the paper.

---

## 8. Build the graph  (rec env, ~10–20 min)

```bash
source env-rec/bin/activate   # Windows: env-rec\Scripts\Activate.ps1
python -m src.graph.build_graph
```

**Acceptance (fix #3 runtime gate):**
- sentiment-edge count is on the order of the normalized-CSV row count (thousands+,
  not 4); skipped-row count is reported (nonzero is fine — those are non-train pairs)
- `applied trust_score to N user nodes` with **N > 0** (needs step 5's CSV)
- reload sanity check prints `matches original: True`

Optionally regenerate the figure (now needs matplotlib, pinned in step 1):

```bash
python -m src.graph.visualize_graph   # -> results/figures/graph_sentiment_edges.png
deactivate
```

---

## 9. Commit + report back  (5 min)

```bash
git status              # should show ONLY results/ changes (tables/figures)
git add results/baselines/summary.csv
# figure: only add if <1 MB — drop dpi in visualize_graph.py if larger
git commit -m "results: baselines regenerated on official benchmark split (P2)"
git push origin devanshi
```

Then report to Muaaz:
1. the step-3 split row counts (3 numbers)
2. the new baseline table (4 rows)
3. trust summary stats
4. the ABSA coverage report (full)
5. graph summary block (node/edge counts per type)
6. wall-clock per step + any deviation from this runbook

That report closes the runtime half of fixes #1 and #3 and is the Gate G2 evidence.

---

## If something breaks

| Symptom | Likely cause | Do |
|---|---|---|
| RecBole `FileNotFoundError` on `.inter` | step 3 not run / wrong `data_path` | re-run step 3 from repo root |
| RecBole ignores the split / logs an 8:1:1 split | config missing `benchmark_filename` — wrong branch? | `git log -1` must be ≥ `7cab0fc` |
| ABSA crash on `update_checker` import | absa env built from old pins | rebuild env (step 1) |
| `pd.read_parquet` ImportError | pyarrow missing → env built from old pins | rebuild env (step 1) |
| ABSA resume looks misaligned | old checkpoint survived step 0 | delete the two files in step 0, restart step 6 |
| OOM building the graph | close other jobs; the build is vectorized but the pickle dump is memory-hungry | ping Muaaz if it persists on your box |

Never mix the two envs. Never `git add -f` ignored data files. Ping Muaaz before
changing anything under `configs/` — every number must trace to a committed config.
