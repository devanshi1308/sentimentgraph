# fixes.md

Tracked issues from the codebase review against `Implementation_PRD.docx`
(branch `devanshi`, as of commit `5454c17`). Ordered by severity. Each item has:
what it is, where it lives, why it matters (PRD reference), and how to fix it.

Legend: 🔴 blocker for the model phase · 🟠 correctness/hygiene · 🟡 polish

---

## 🔴 1. The official 5-core benchmark split is being discarded

> **STATUS: ✅ CODE-RESOLVED (P1, commit `216f9fa`)** — `build_atomic` now writes
> `video_games.{train,valid,test}.inter`; all configs load them via
> `benchmark_filename`, verified against RecBole 1.2.1 source (`Dataset.build()`
> early-returns, bypassing `eval_args` splitting; only `mode: full` stays live).
> **Runtime acceptance (row counts == official split, seed-stable reruns) lands in
> P2** on a machine with the rec env + data.

**Where:** `src/data/build_atomic.py` (`build_inter`), all `configs/*.yaml`.

**What happens.** `build_inter()` downloads the official timestamp-based
`train` / `valid` / `test` CSVs and then concatenates all three back into a
single `video_games.inter` with no split label:

```python
combined = pd.concat(dfs, ignore_index=True)   # train + valid + test merged
```

None of the configs set `eval_args`, so RecBole falls back to its default
evaluation split — a **random** re-split (`{'split': {'RS': [0.8, 0.1, 0.1]},
'order': 'RO', ...}`). The carefully-constructed temporal benchmark split is
thrown away, and the reported baseline numbers in `results/baselines/summary.csv`
are measured on a random re-partition, not the standard test set.

**Why it matters.** PRD §3 explicitly says: *"use the official 5-core benchmark
split … Using those instead of rolling our own k-core is less work and more
reproducible."* PRD §5 lists reproducibility as core. Right now the numbers
neither match the published benchmark nor are reproducible against it, which
undercuts the "solid and reproducible, not best-in-class" goal and the
threats-to-validity story.

**Fix (recommended: preserve the split via a label column).**
1. In `build_inter`, tag each row with its split and keep the tag:
   ```python
   for name, path in split_paths.items():
       df = pd.read_csv(path)
       df["split"] = {"train": 0, "valid": 1, "test": 2}[name]
       dfs.append(df)
   ```
   Emit it as a RecBole field, e.g. `split:float`, in the `.inter`.
2. In every config, tell RecBole to use the preset split instead of re-splitting:
   ```yaml
   eval_args:
     split: {'LS': 'valid_and_test'}   # or the preset/leave-one split you intend
     order: TO                         # temporal order, not random
     group_by: user
     mode: full
   load_col:
     inter: [user_id, item_id, rating, timestamp]
   ```
   If sticking with RecBole's built-in splitter, at minimum set `order: TO` so
   the split respects timestamps. The cleanest reproduction is to feed RecBole
   the exact benchmark partition rather than any re-split.
3. Re-run all four baselines and regenerate `summary.csv`.

**Acceptance.** The row counts RecBole logs for train/valid/test match the
official split file sizes, and numbers are stable across reruns with `seed 2020`.

---

## 🔴 2. RecBole-GNN and PyTorch Geometric are not installed

> **STATUS: ✅ RESOLVED (prep phase; Muaaz sign-off 2026-07-18)** — investigation
> (`HARNESS_DECISION.md`) found RecBole-GNN unmaintained (last commit 2023-10), not on
> PyPI, and pinned to `recbole==1.1.1` vs our locked `1.2.1`. **Decision: Option B** —
> baselines stay on vanilla RecBole; `torch-geometric==2.6.1` (pure-Python wheel) is
> pinned for use inside `sthgrec.py` only; PRD §2 amended accordingly. Custom
> `torch.sparse` (Option C) is the sanctioned fallback. Trust representation (fix #4)
> decided the same day: **Option 4A** (propagation weight; `STHGREC_DESIGN.md` §3).

**Where:** `environments/requirements-rec.txt` (only `recbole==1.2.1`);
`src/eval/run_baselines.py` imports `LightGCN`/`NGCF` from
`recbole.model.general_recommender`.

**What happens.** The PRD locks **RecBole-GNN** as the single harness and notes
PyG "arrives as a RecBole-GNN dependency." The pinned rec env has neither
`recbole-gnn` nor `torch-geometric` / `torch-scatter` / `torch-sparse`. The
baselines currently run on vanilla RecBole's LightGCN/NGCF, which works — but
step 8 (ST-HGRec, the heterogeneous LightGCN, "Option D") needs PyG, and it
isn't there.

**Why it matters.** PRD §2: *"Why RecBole-GNN: it ships LightGCN and NGCF with
sparse-tensor acceleration … so every baseline and our default backbone run
through one harness."* PRD §6.3 / step 8: ST-HGRec is where "we build the
heterogeneous ST-HGRec extension … we only touch [PyG] directly" then. Without
RecBole-GNN/PyG the backbone can't be built, and the baselines don't share the
same harness the paper claims.

**Decision needed.** Either:
- **(a)** Adopt RecBole-GNN as intended: add `recbole-gnn` + matching
  `torch-geometric` (and the `torch-scatter`/`torch-sparse` wheels for
  `torch==2.5.1`) to `requirements-rec.txt`, switch the LightGCN/NGCF imports to
  the RecBole-GNN versions, and re-verify the baseline numbers; **or**
- **(b)** Consciously document that vanilla RecBole is the harness for baselines
  and that PyG will be added only at step 8 — and update the PRD/stack note so
  the "one harness" claim stays honest.

Recommend (a) before the model phase so baselines and ST-HGRec share one stack.

---

## 🔴 3. ABSA output is not wired into the graph (pipeline disconnected)

> **STATUS: ✅ CODE-RESOLVED (P1, commits `7319872`, `5505a8b`)** — sentiment flow is
> now `absa_full.csv` → `normalize_aspects.py` → `absa_normalized.csv` → `build_graph`
> (CLI-overridable). Graph + trust + ABSA population all read `train.inter` only
> (test-leakage guard). Edge construction vectorized; skipped rows counted; aspect
> nodes from kept rows only. **Full-scale run (real edge counts) lands in P2.**

**Where:** `src/sentiment/run_absa.py` writes
`data/interim/absa_full.csv`; `src/graph/build_graph.py` reads
`data/interim/absa_sample.csv` (`SENTIMENT_PATH`).

**What happens.** `run_absa.py` was rewritten to do a full-population run
(~899,964 reviews of the 5-core users, batched + checkpointed) and now writes
`absa_full.csv`. But `build_graph.py` still points at the old 19-row
`absa_sample.csv`. The full-scale sentiment output never reaches the graph.
`absa_full.csv` isn't on disk yet (the long run hasn't completed/committed), so
the committed `graph.gpickle` is still built from 19 sample rows — hence
`visualize_graph.py`'s own title, *"The 4 sentiment/aspect edges."* The graph's
sentiment layer is effectively a placeholder.

**Why it matters.** PRD step 7 is "attach sentiment … edges" and the whole thesis
rests on aspect-level sentiment being in the graph. A 4-edge sentiment layer
can't support ST-HGRec or the explanation study.

**Fix.**
1. Point `build_graph.py` at the real output. Prefer making it a constant that
   matches `run_absa.py`, or parameterize via CLI:
   ```python
   SENTIMENT_PATH = REPO_ROOT / "data" / "interim" / "absa_full.csv"
   ```
2. Run the full ABSA job to completion (it's resumable via
   `absa_full_progress.txt`), then rebuild the graph.
3. Re-generate the visualization and confirm the sentiment/aspect edge count is
   now realistic (thousands+), not 4.

**Acceptance.** `build_graph` reports a sentiment-edge count on the same order as
the ABSA rows, and the graph's aspect nodes reflect the full aspect vocabulary.

---

## 🟠 4. Trust "edges" are node attributes, and the committed graph has none

> **STATUS: ✅ RESOLVED (P4, commits `91477ed`, `a1ae636`)** — **Option 4A implemented**:
> trust scales the user's interaction-edge weights inside `graph_tensors` (missing
> trust → 1.0), verified by independent gate audit to touch only interaction edges and
> to restore exact equality under `use_trust_edges: false`. Analysis in
> `STHGREC_DESIGN.md` §3. The graph-side `trust_applied > 0` full-scale check still
> runs in P2.

**Where:** `src/graph/build_graph.py` (`build_graph`, trust block);
`src/data/trust_signals.py` output `data/interim/trust_signals.csv` (absent).

**What happens.** Trust is applied as a **node attribute**
(`G.nodes[u]["trust_score"] = ...`), not as typed edges. And
`trust_signals.csv` was never generated before the current graph was built, so
`graph.gpickle` has `trust_applied = 0` — the code logs *"none yet — Day 6
task."* The trust signal is absent from the shipped artifact entirely.

**Why it matters.** PRD step 7 says "attach sentiment + trust **edges**," and
trust is a first-class signal for the "trust-aware" contribution. If ST-HGRec is
meant to message-pass over trust, an attribute on a node is not the same as a
typed edge in the heterogeneous graph.

**Fix.**
1. Run `trust_signals.py` to produce `trust_signals.csv` (it already scopes to
   the 5-core user population).
2. Decide the trust representation for ST-HGRec:
   - keep `trust_score` as a **user node feature** (fine if the model consumes
     node features), **or**
   - materialize **typed edges** (e.g. `user --trusts--> user`, or
     `user --trust_weight--> item`) if message-passing over trust is intended.
   Document the choice so it matches the model design.
3. Rebuild the graph and confirm `trust_applied` > 0.

---

## 🟠 5. Large binaries committed to git

> **STATUS: ✅ RESOLVED (P0, commit `ed9e236`)** — `graph.gpickle` (~90 MB),
> `absa_sample.csv`, and the 1.3 MB figure untracked (kept on disk; figure moved to
> `results/figures/`). `.gitignore` now covers `data/processed/`, `*.gpickle`, and
> `data/interim/*.{csv,txt}`. Verified by G0 gate: no tracked file >24 KB.
> **Deferred sub-item (decided at G0):** purging the ~90 MB blobs already in history
> is deferred to a pre-submission quiet moment (filter-repo + coordinated force-push;
> Devanshi re-clones once).

**Where:** `data/processed/video_games/graph.gpickle` (~90 MB, and growing:
89 → 94 → 90 MB across commits); `results/graph_sentiment_edges.png` (1.3 MB).
`.gitignore` covers `*.pt/.pth/.bin/.ckpt` and `data/raw/` but **not**
`*.gpickle` or `data/processed/`.

**What happens.** A ~90 MB pickle is committed and rewritten every time the graph
changes, permanently bloating history. The PNG is a smaller but similar case.

**Why it matters.** PRD §9: *".gitignore ignores data/raw and large model
files"* and *"raw data is gitignored; everything else is public by submission
time"* — the intent is that heavy, regenerable artifacts don't live in git.
`graph.gpickle` is regenerable from code + `absa_full.csv` + `trust_signals.csv`,
so it belongs out of tracking.

**Fix.**
1. Add to `.gitignore`:
   ```
   # regenerable heavy artifacts
   data/processed/
   *.gpickle
   ```
   (Keep `.gitkeep` placeholders so the dirs survive.)
2. Untrack without deleting locally:
   ```
   git rm --cached data/processed/video_games/graph.gpickle
   ```
3. It's already in history (multiple ~90 MB blobs). If repo size matters before
   submission, purge with `git filter-repo --path
   data/processed/video_games/graph.gpickle --invert-paths` (coordinate — it
   rewrites history and needs a force-push).
4. Figures: small PNGs in `results/figures/` are arguably fine to commit, but be
   deliberate. If kept, move it there (see #9).

---

## 🟠 6. The fixed aspect list was abandoned

> **STATUS: ✅ RESOLVED (P1, commits `a0a3b24`, `5505a8b`)** — lead chose
> **extract-all, normalize-after**: `run_absa` stays open-vocab; `aspects.py` is now a
> 10-canonical vocabulary (PRD four first, priority-ordered) with word-boundary/prefix
> matching (adversarially tested — `hardware`/`scoreboard`/`salesman`-class false
> positives eliminated); `normalize_aspects.py` maps + reports coverage and top
> unmatched terms for the lead's curation loop.

**Where:** `src/sentiment/aspects.py` (`ASPECTS`) — no longer imported anywhere;
`src/sentiment/run_absa.py` (old version filtered to `ASPECTS`, new one doesn't).

**What happens.** The rewritten `run_absa.py` writes every open-vocabulary aspect
PyABSA emits, with no filtering to the locked list
(`gameplay, graphics, price, difficulty`). `aspects.py` is now dead code (only
self-referenced).

**Why it matters.** PRD step 5: *"PyABSA on ~300 reviews; **fix aspect list**;
eyeball 20,"* and `aspects.py`'s own docstring says the list is *"fixed after
eyeballing … so the sentiment table stays comparable across runs."* Dropping it
silently changes the aspect vocabulary from a controlled set to whatever PyABSA
produces, which affects graph structure, ablations, and the explanation aspects.

**Fix — pick one and document it:**
- **(a) Re-apply the fixed list** as a filter/normalizer in `run_absa.py` (map
  raw aspects → canonical `ASPECTS`, drop the rest), restoring PRD step 5; **or**
- **(b) Intentionally go open-vocabulary** (extract all now, curate later) — then
  delete or repurpose `aspects.py`, and note the decision so "fix aspect list"
  isn't silently unmet.

Either way, remove the orphaned/dead code state.

---

## 🟡 7. `matplotlib` missing from the rec environment

> **STATUS: ✅ RESOLVED (P0, commit `52fff89`)** — pinned `matplotlib==3.11.1`,
> `httpx==0.28.1`, `pyarrow==25.0.0` + 8 transitives (httpx/pyarrow were also missing —
> found by full import scan). `uv pip compile` resolves the file cleanly (66 pkgs,
> no conflicts). Verified by G0 gate: every third-party import in rec-side `src/` maps
> to a pin. See new item #11 for the absa-env counterpart.

**Where:** `src/graph/visualize_graph.py` does `import matplotlib.pyplot as plt`;
`environments/requirements-rec.txt` pins `plotly` but **not** `matplotlib`.

**What happens.** On a clean install per the pinned env, `visualize_graph.py`
raises `ModuleNotFoundError: matplotlib`.

**Why it matters.** PRD §5 acceptance: both requirements files should install
clean and the scripts should run. A script that needs an unpinned dep breaks
reproducibility.

**Fix.** Add the pinned `matplotlib` (the version actually used) to
`requirements-rec.txt`, or re-implement the plot with the already-pinned
`plotly`. Re-freeze with `uv pip freeze` inside the env to catch any other
unpinned transitive deps.

---

## 🟡 8. `run_baselines.py` defines `MODEL_CLASSES` twice

> **STATUS: ✅ RESOLVED (P0, commit `1f23efc`)** — dead 2-model dict and `# add this`
> comment removed; model imports regrouped. G0 gate verified: single `MODEL_CLASSES`
> with all four models, ast-parses, no functional change.

**Where:** `src/eval/run_baselines.py` lines ~30–40.

**What happens.** The dict is assigned twice — the first (`Pop`, `BPR` only) is
immediately shadowed by the second (adds `LightGCN`, `NGCF`). Dead code, plus a
stray `# add this` comment on the `LightGCN` import.

**Why it matters.** No functional bug, but this is the reproducibility-critical
runner; clutter here invites confusion. PRD leans on `configs/` + this script as
the audit trail.

**Fix.** Delete the first `MODEL_CLASSES` block and the `# add this` comment;
keep the single four-model dict.

---

## 🟡 9. Figure saved outside `results/figures/`

> **STATUS: ✅ RESOLVED (P0, commit `154bfe8`)** — `OUTPUT_PATH` now writes to
> `results/figures/`; the hardcoded "The 4 …" title replaced with a dynamic edge count.
> Existing PNG moved on disk to the new location (untracked, regenerable).

**Where:** `src/graph/visualize_graph.py` writes to `results/graph_sentiment_edges.png`;
PRD §9 designates `results/figures/` for figures (currently empty).

**What happens.** The PNG lands at the `results/` root instead of the intended
`figures/` subdir.

**Fix.** Point `OUTPUT_PATH` at `results/figures/graph_sentiment_edges.png`.
Combine with the #5 decision on whether figures are committed at all.

---

## 🟡 10. Docs and commit hygiene

> **STATUS: ✅ RESOLVED (P5/P6 close-out)** — README rewritten to match the real
> pipeline (three-split atomic files, normalization step, ST-HGRec, tests) with a
> repo map of the runbooks/design docs; KICKSTART step 3 corrected at G1. Commit
> style has been conventional-format since P0 (enforced at every gate).

- **README (this branch)** is still the scaffold version — it doesn't reflect the
  now-pinned envs, the actual run commands, or the real dataset stats. `KICKSTART.md`
  is the accurate guide; either fold the run instructions into README or point
  README at KICKSTART.
- **Commit messages** on `devanshi` (`add`, `graph`, `lightgcn + ngcf`) diverge
  from the conventional-commit style used on `main`. Not blocking; worth aligning
  before any merge to `main` so history stays consistent.

---

## 🟡 11. `requirements-absa.txt` missing runtime deps (addendum, found in P0)

> **STATUS: ✅ RESOLVED (P1, commit `1d2168f`)** — `pyarrow==25.0.0` (not in pyabsa's
> tree at all; required by `read_parquet`) and `update-checker==1.0.0` (transitive but
> previously unpinned) pinned; `uv pip compile` resolves 95 packages, zero conflicts.
> Full-install verification deferred to P2 on the ABSA machine.

**Where:** `environments/requirements-absa.txt`; `src/sentiment/run_absa.py`.

**What happens.** The absa env pins only 4 top-level packages, but `run_absa.py`
(a) imports `update_checker` at module top to monkeypatch it — not pinned, so a
fresh install crashes on import; (b) calls `pd.read_parquet(...)`, which needs
`pyarrow` at runtime — also not pinned.

**Why it matters.** PRD §5 acceptance: both requirements files install clean and the
scripts run. The absa env currently only works on machines where these arrived
transitively.

**Fix.** Pin `update_checker` and `pyarrow` (and ideally full-freeze the absa env with
`uv pip freeze` on the machine where it works) into `requirements-absa.txt`.
Scheduled for Phase P1/P2 alongside sentiment work.

---

## Suggested order of work

1. **#3** wire `build_graph` → `absa_full.csv` (quick, unblocks the real graph).
2. **#5** untrack the 90 MB pickle + tighten `.gitignore` (quick, stops the bleed).
3. **#1** preserve the official split + re-run baselines (correctness of numbers).
4. **#2** decide RecBole-GNN vs vanilla; install PyG before step 8.
5. **#4 / #6** trust representation + aspect-list decision (design choices).
6. **#7–#10** env pin, dead code, figure path, docs (cleanup).

Items #7–#9 are ~15 minutes total; #1 and #2 are the ones that change whether the
model-phase numbers are trustworthy and reproducible.
