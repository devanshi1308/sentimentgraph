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

Accepted PRD §2 amendment text (applies on sign-off):
> Harness = vanilla RecBole 1.2.1 (Config/Trainer/eval). PyG (`torch_geometric==2.6.1`,
> pure-Python) is used only inside `src/models/sthgrec.py`. RecBole-GNN dropped:
> unmaintained since 2023-10, no PyPI release, pinned to recbole 1.1.1.

## De-risk steps (P2/P4)

1. `uv pip install torch_geometric==2.6.1` into the rec env on both machines;
   confirm `import torch_geometric; from torch_geometric.nn import MessagePassing`
   against torch 2.5.1.
2. Toy-graph parity test: one LightGCN propagation layer, RecBole vanilla vs our
   path, assert embeddings match.
3. Confirm `configs/sthgrec.yaml` drives the identical `benchmark_filename` split as
   baselines (apples-to-apples `summary.csv` row).

## Status

- [ ] Muaaz sign-off: harness option (B recommended / C fallback / A)
- [ ] Muaaz sign-off: trust representation (4A recommended — see `STHGREC_DESIGN.md` §3)
- On sign-off: add the chosen pins to `requirements-rec.txt`, apply the PRD amendment
  note here and in `fixes.md` #2, and unblock P4 T4.2/T4.3.
