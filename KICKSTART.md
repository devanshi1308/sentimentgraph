# KICKSTART — setup guide (Devanshi)

Everything you need to go from a fresh clone to running baselines. This mirrors
the build order in the PRD (§4). You own the hands-on pipeline: envs, data,
atomic files, baselines, ablations. Muaaz owns the expensive-to-reverse calls
(category, env split, harness), the explanation generator, and the ST-HGRec core.

If a task doesn't move the offline pipeline forward or support the
honest-vs-fluent comparison, it's out of scope for the sprint.

---

## 0. Prerequisites

- **Python 3.10+** (check: `python --version`)
- **git** + a GitHub account with access to `MuaazSM/st-hgrec`
- **~20 GB free disk** — the raw Video Games parquet is large
- A GPU is nice for LightGCN/NGCF but not required for steps 1–4
- Recommended: a HuggingFace account (`pip install huggingface_hub`, then
  `huggingface-cli login`) to pull the dataset

```bash
git clone https://github.com/MuaazSM/st-hgrec.git
cd st-hgrec
```

---

## 1. Two environments (locked — never share a process)

PyABSA and RecBole **will fight** if they share an environment. That's the whole
reason for the split. Pin versions in both from day one — this also feeds the
threats-to-validity section.

### Recommender env (RecBole + RecBole-GNN)

```bash
python -m venv env-rec
source env-rec/bin/activate          # Windows: env-rec\Scripts\activate
pip install -r environments/requirements-rec.txt
deactivate
```

### Sentiment env (PyABSA) — separate

```bash
python -m venv env-absa
source env-absa/bin/activate          # Windows: env-absa\Scripts\activate
pip install -r environments/requirements-absa.txt
deactivate
```

> The requirements files currently list packages with version pins left blank.
> **On your first clean install, freeze the exact versions** that worked
> (`pip freeze > environments/requirements-rec.txt` inside each env) and commit
> them. That is acceptance step 1: both files install clean with pinned versions.

`src/sentiment/*` never imports from the recommender side. The two halves meet
**only** through files in `data/`.

---

## 2. Get the data (Amazon Reviews 2023 — Video Games)

Source: `huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023`
Configs: `raw_review_Video_Games`, `raw_meta_Video_Games`.

```bash
source env-rec/bin/activate
python -m src.data.download        # pulls HF parquet configs -> data/raw/
```

**Known gotcha:** the dataset moved to Parquet and the old load script is
deprecated, so `trust_remote_code` can error on newer `datasets` versions. Load
the Parquet configs directly, or use the pre-made 5-core benchmark splits.
Budget time for this — it's the #1 schedule risk.

- Interactions: use the **official 5-core benchmark split** (timestamp-based
  train/valid/test) — don't roll your own k-core.
- Review text: pull `raw_review_Video_Games` separately; it feeds PyABSA and the
  trust signals.

**Acceptance:** a printout of users, items, interactions, and density for Video
Games (≈2.55M reviews over ≈116K items). See `notebooks/01_explore_video_games.ipynb`.

---

## 3. Build the RecBole atomic files

```bash
source env-rec/bin/activate
python -m src.data.build_atomic    # 5-core split -> data/processed/video_games/*.{inter,item,user}
```

Produces three interaction files — `video_games.train.inter`, `video_games.valid.inter`,
`video_games.test.inter` — preserving the official 5-core timestamp split verbatim, plus
`video_games.item` and `video_games.user`. RecBole loads them via `benchmark_filename`
(not a single combined `.inter`). The combined `video_games.inter` is intentionally absent.

**Acceptance:** RecBole loads all three split files without error and row counts match the
official 5-core split.

---

## 4. Run the first baselines

```bash
source env-rec/bin/activate
python -m src.eval.run_baselines --config configs/popularity.yaml
python -m src.eval.run_baselines --config configs/bpr.yaml
```

Each config carries its **seed + metric settings** — that's the reproducibility
spine. Every number in the paper traces back to a config file and a seed.

**Acceptance:** a small table with Recall@K and NDCG@K for Popularity and BPR,
written to `results/baselines/`.

> Steps 1–4 are the "done for today" spine. Protect them; later steps can slip a
> day without hurting July 12.

---

## 5. Sentiment sanity check (ABSA env)

```bash
source env-absa/bin/activate
python -m src.sentiment.run_absa   # PyABSA on ~300 reviews -> aspect/sentiment table
```

Fix the aspect list in `src/sentiment/aspects.py`, then eyeball ~20 outputs.

**Acceptance:** ~20 example aspect→sentiment outputs you've eyeballed for sanity.

---

## 6. Add the GNN baselines

```bash
source env-rec/bin/activate
python -m src.eval.run_baselines --config configs/lightgcn.yaml
python -m src.eval.run_baselines --config configs/ngcf.yaml
```

Add LightGCN + NGCF to the baseline table. All baselines run through the **one**
RecBole / RecBole-GNN harness.

---

## Later (after the spine is green)

- **Step 7** — build the graph + attach sentiment/trust edges (`src/graph/`,
  with Muaaz)
- **Step 8** — ST-HGRec, the heterogeneous LightGCN (`src/models/`, Muaaz-led)
- **Step 9** — explanation generator + faithfulness check (`src/explain/`, Muaaz)
- **Step 10** — switch-off (ablation) tests (`src/eval/ablations.py`, you)

---

## Non-goals (don't build these in the sprint)

- No language model in the **live** recommender
- No custom PyG code until ST-HGRec (step 8)
- No LLM graph augmentation unless offline numbers actually need help
- No running the human study in the sprint
- No extra datasets (Beauty comes later) until the pipeline works end to end

---

## When you're stuck

- Dataset loading errors → try the direct Parquet path / pre-made splits first.
- Dependency conflicts → confirm you're in the right venv (`which python`); never
  mix the two envs.
- Anything about category / env split / harness / first-number sanity → ping
  Muaaz before changing it; those are expensive to reverse.
