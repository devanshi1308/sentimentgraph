# st-hgrec

Trust-aware sentiment-graph recommendation — offline pipeline.

The recommender, the sentiment pipeline, and the heterogeneous graph that together
produce faithful, aspect-level explanations. The paper's contribution is the
calibration study; the offline system is built to be solid and reproducible, not
best-in-class.

## Repo map

| Doc | What it's for |
| --- | --- |
| `KICKSTART.md` | Setup guide: envs, data, atomic files, baselines |
| `P2_RUNBOOK.md` | Step-by-step artifact regeneration on the corrected data spine |
| `STHGREC_DESIGN.md` | ST-HGRec model design (signed sentiment, trust-4A, ablations, explanation hooks) |
| `HARNESS_DECISION.md` | Why vanilla RecBole + pure-Python PyG (RecBole-GNN dropped) |
| `fixes.md` | PRD-conformance issue tracker with resolution evidence |
| `IMPLEMENTATION_PROMPT.md` | Multi-agent orchestration brief that drove the build |

## Two environments (locked)

Two isolated Python envs that never share a process; the halves meet only through
files in `data/`. Both requirements files are fully pinned and resolve clean.

| Env | File | Covers |
| --- | --- | --- |
| Recommender | `environments/requirements-rec.txt` | RecBole 1.2.1, torch 2.5.1, PyG 2.6.1 (used only inside `sthgrec.py`) — `src/data`, `src/graph`, `src/models`, `src/eval`, `src/explain` |
| Sentiment | `environments/requirements-absa.txt` | PyABSA — `src/sentiment` only |

```bash
uv venv env-rec --python 3.11 && source env-rec/bin/activate
uv pip install -r environments/requirements-rec.txt
```

## Pipeline (data → numbers)

```
download.py ──► raw parquet (data/raw/, gitignored)
build_atomic.py ──► video_games.{train,valid,test}.inter + .item/.user
                    (official 5-core timestamp split preserved verbatim;
                     RecBole loads it via benchmark_filename)
run_absa.py (absa env) ──► absa_full.csv (open-vocab, checkpointed)
normalize_aspects.py ──► absa_normalized.csv (10 canonical aspects)
trust_signals.py ──► trust_signals.csv (per-user trust score)
build_graph.py ──► heterogeneous graph (train interactions only — no test leakage)
run_baselines.py ──► results/baselines/summary.csv  (Pop, BPR, LightGCN, NGCF)
run_sthgrec.py ──► ST-HGRec row in the same table
ablations.py ──► results/ablations/summary.csv (full / −sentiment / −trust / −both)
```

Every number traces to a `configs/*.yaml` + seed 2020. Heavy artifacts
(`data/processed/`, `data/interim/*.csv`, `*.gpickle`) are gitignored and
regenerate from code + config + seed.

## ST-HGRec

Heterogeneous LightGCN over user/item/aspect nodes (`src/models/`): signed
sentiment edge weights (±1, degrees on |w|), trust as a propagation weight on the
user's interaction edges (Option 4A), `torch.sparse.mm` propagation inside the
vanilla RecBole harness. Explanations (`src/explain/`): `honest.py` renders only
graph-traceable facts; `ungrounded.py` is a quarantined study stimulus (an AST
test proves no recommender module can import it); `faithfulness.py` implements the
remove-a-reason check.

## Tests

```bash
python -m pytest tests/ -q     # 24 tests, ~3s, synthetic data — no downloads needed
```

Covers: LightGCN parity, signed-weight semantics, trust scaling, ablation
exactness (bitwise), determinism, end-to-end harness smoke, explanation
grounding, ungrounded quarantine, faithfulness sensitivity.
