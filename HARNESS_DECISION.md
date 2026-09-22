# Harness decision (fix #2) — options analysis for sign-off

Companion to `STHGREC_DESIGN.md`. Produced by the T4.1 investigation (web + uv
resolution evidence, 2026-07-18). Decision is escalation-gated to Muaaz
(`IMPLEMENTATION_PROMPT.md` §6): the PRD §2 locked "RecBole + RecBole-GNN", but the
investigation shows that lock points at a dead project.

## Key findings

1. **RecBole-GNN is effectively unmaintained.** Last commit **2023-10-30**; single
   release (v0.1.0, 2022); 25 open issues; **not on PyPI** (source-install only, would
   need a git-SHA pin). Its README pins **`recbole==1.1.1`** — our locked env is
   `recbole==1.2.1`, and RecBole-GNN subclasses 1.1.1-era data-layer internals.
2. **Adopting it drags in the compiled-wheel matrix.** `recbole_gnn/model/layers.py`
   does `from torch_sparse import matmul` → hard dependency on compiled
   torch-scatter/torch-sparse wheels matched to torch+platform (extra index url,
   build-isolation traps confirmed in testing).
3. **Modern PyG needs none of that.** `torch_geometric==2.6.1` is a **pure-Python,
   platform-agnostic wheel** (`py3-none-any`); PyG ≥2.3 officially needs no external
   libs beyond torch. `uv pip compile` resolves `torch==2.5.1 + recbole==1.2.1 +
   torch_geometric==2.6.1` cleanly on **both** macOS arm64 and linux x86_64
   (only transitive add: aiohttp). One caveat: PyG 2.6's official compat table tops
   out at torch 2.4; low practical risk for a pure-Python wheel, runtime import to be
   confirmed in P2 (de-risk step 1).
4. **The no-PyG path is viable.** Heterogeneous LightGCN is per-edge-type normalized
   sparse matmuls — implementable with `torch.sparse` already in torch 2.5.1
   (~150–250 LOC + a toy-graph parity test), registered as a plain RecBole model in
   the existing harness.

## Options

| | A: adopt RecBole-GNN | B: vanilla RecBole + pure PyG in ST-HGRec only | C: no PyG, custom torch.sparse |
|---|---|---|---|
| New deps | git-SHA install + PyG + **compiled torch-sparse** | `torch-geometric==2.6.1` (+aiohttp) only | **none** |
| Baselines | must re-run through GNN path; numbers shift | unchanged, comparable | unchanged, comparable |
| "One harness" (PRD §2) | literally satisfied | harness single (RecBole Config/Trainer); model backend differs — needs a one-line PRD amendment | literally satisfied |
| Reproducibility | worst (source pin + wheel matrix + 1.1.1↔1.2.1 mismatch) | good (platform-agnostic wheel, uv-verified) | best |
| Effort / risk | high / **HIGH** | low deps, medium model work / LOW | medium-high model work / LOW-MED |

## Recommendation

**Option B, with Option C as the sanctioned fallback.** A is the worst trade —
unmaintained, version-mismatched, and it forces the compiled-wheel matrix we can
otherwise avoid, all to reuse math we can express in a few sparse matmuls. B buys
`HeteroData`/`MessagePassing` ergonomics for one pure-Python wheel; if PyG's
unofficial torch-2.5 status ever bites, C drops in with zero deps (the design in
`STHGREC_DESIGN.md` §5 is written to survive either).

> **What actually shipped (P4): B for the harness, C for the model internals.**
> See "Implementation outcome" below — the `torch-geometric` pin stayed, but the
> ST-HGRec propagation is hand-written `torch.sparse.mm`, not PyG `MessagePassing`.

Accepted PRD §2 amendment text (applies on sign-off):
> Harness = vanilla RecBole 1.2.1 (Config/Trainer/eval). PyG (`torch_geometric==2.6.1`,
> pure-Python) is used only inside `src/models/sthgrec.py`. RecBole-GNN dropped:
> unmaintained since 2023-10, no PyPI release, pinned to recbole 1.1.1.

## De-risk steps (P2/P4) — all cleared

1. **[x] PyG runtime import on torch 2.5.1.** `torch-geometric==2.6.1` installs and
   imports against `torch==2.5.1` in both `env-rec` and `rec-env`:
   `import torch_geometric` → `2.6.1`; `from torch_geometric.nn import MessagePassing`
   succeeds. The "PyG 2.6 compat table tops out at torch 2.4" caveat did not bite.
2. **[x] Toy-graph LightGCN parity.** Landed as
   `tests/test_sthgrec.py::test_parity_lightgcn`: a 5-node bipartite graph with
   sentiment+trust off, checked against a **closed-form** `D^-1/2 A D^-1/2`
   LightGCN reference (stronger than the vanilla-RecBole-vs-ours diff originally
   sketched). Part of the 24-test suite that passes green.
3. **[x] Same split as baselines (config level).** `configs/sthgrec.yaml` and every
   baseline config carry `benchmark_filename: [train, valid, test]`;
   `run_sthgrec.py` reuses the identical `create_dataset` / `data_preparation`
   flow; P2/P3 logs show identical `eval_args`. A runtime train/valid/test
   row-count assertion is still tracked as **fixes.md #1** (P2 acceptance, not
   harness-specific).

## Implementation outcome (P4)

**Harness = Option B; model internals = Option C.** The environment decision held
verbatim — vanilla RecBole 1.2.1 `Config`/`Trainer`/`evaluate` drive everything
(`src/eval/run_sthgrec.py`), `torch-geometric==2.6.1` is pinned, RecBole-GNN is
gone. But the ST-HGRec propagation itself (`src/models/sthgrec.py`,
`src/models/graph_tensors.py`) is hand-written `torch.sparse.mm` over per-relation
`D^-1/2 A D^-1/2` COO adjacencies — **`torch_geometric` is imported nowhere in
`src/` or `tests/`.** Commits `43bcde7` (pin) and `91477ed` (model, "sanctioned
fallback path under harness Option B").

Why the sparse path won in practice: heterogeneous LightGCN is one `spmm` per
relation per layer with no learned message function, so `HeteroData`/
`MessagePassing` bought no ergonomics over three `torch.sparse.mm` calls, and the
sparse path is deterministic on CPU and dependency-free.

**Status of the `torch-geometric` pin:** currently unused by any code path. Keep
it — it is a pure-Python `py3-none-any` wheel (zero install cost, uv-clean), and
`STHGREC_DESIGN.md` §2.3 explicitly reserves PyG for a possible future
per-aspect-relation extension. Revisit only if the env is being trimmed for
submission.

Device (updated 2026-09-10, see `GPU.md`): ST-HGRec now runs on **CUDA or CPU**.
`configs/sthgrec.yaml` sets `use_gpu: true` / `gpu_id: 0`; `run_sthgrec.py` moves
the per-relation sparse adjacencies onto the model device and forces CPU on Apple
MPS (no sparse kernels). `torch.sparse.mm` is deterministic on CPU and expected
deterministic on CUDA — the adjacencies are frozen buffers, so the only spmm
backward path is another structured spmm. Verify per-GPU with
`python -m src.eval.run_sthgrec --check-determinism` before reporting a number.

## Results (seed 2020, official 5-core split configs)

From `results/baselines/summary.csv` (P2/P3):

| model | recall@10 | recall@20 | ndcg@10 | ndcg@20 |
|-------|-----------|-----------|---------|---------|
| Pop | 0.0050 | 0.0071 | 0.0032 | 0.0039 |
| BPR | 0.0073 | 0.0111 | 0.0049 | 0.0060 |
| NGCF | 0.0075 | 0.0135 | 0.0049 | 0.0067 |
| LightGCN | 0.0079 | 0.0131 | 0.0047 | 0.0063 |
| **STHGRec** (no-sentiment sanity) | **0.0094** | **0.0142** | **0.0056** | **0.0071** |

- Ordering is sane (LightGCN ≥ NGCF ≥ BPR ≥ Pop). ST-HGRec lands modestly above
  LightGCN, consistent with the PRD bar ("solid and reproducible… at-or-modestly-
  above LightGCN is success").
- The STHGRec row is a **no-sentiment sanity run**: `absa_normalized.csv` is not
  on disk yet (**fixes.md #3**, P2), so the sentiment + has_aspect relations are
  empty regardless of the config flag; trust 4A is active. It is **not** the
  headline number. Trained 80 epochs, best at epoch 69, CPU.
- Whole benchmark runs ~10× lower than the stale `0.0865` reference in
  `STHGREC_DESIGN.md` §5 — that number predates the official-split correction
  (fix #1); update or footnote it so nobody chases it.

**Still open (not harness blockers):** headline ST-HGRec number with sentiment on
(blocked on ABSA full run, fixes.md #3); the −sentiment / −trust / −both ablation
rows (`src/eval/ablations.py` exists, commit `5d45e52`; P6); seed-stable rerun
assertion (design §4 / fixes.md #1).

## Status

- [x] **Muaaz signed off (2026-07-18): Option B** — vanilla RecBole harness, PyG
  pinned for ST-HGRec only; Option C stays the sanctioned fallback. PRD §2
  amendment in force.
- [x] **Muaaz signed off (2026-07-18): trust Option 4A** — trust as a propagation
  weight on the user's interaction edges (see `STHGREC_DESIGN.md` §3).
- [x] Pins applied to `requirements-rec.txt`: `torch-geometric==2.6.1` + transitives
  (aiohttp, aiohappyeyeballs, multidict, propcache, yarl); full file re-resolves clean.
- [x] **P4 T4.2/T4.3 delivered** (`91477ed`, `a1ae636`): `graph_tensors` +
  `STHGRec` + `run_sthgrec.py`, 24 tests green, real-data run in `summary.csv`.
  In the end the model uses the Option-C `torch.sparse` path under the Option-B
  harness (see "Implementation outcome").
- [x] All three de-risk steps cleared (see above).
