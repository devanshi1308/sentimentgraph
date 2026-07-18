# STHGREC_DESIGN.md

Design document for **ST-HGRec** — a heterogeneous LightGCN over the
`user / item / aspect` graph (PRD step 8, "Option D"). This is the P4 design
that the lead signs off before `model-core` implements it. It is written to
survive whichever harness the parallel `env-harness` analysis picks
(RecBole-GNN / standalone PyG / custom `torch.sparse` in vanilla RecBole); the
harness-dependent bits are quarantined in §5 and §10.

Ground truth: `IMPLEMENTATION_PROMPT.md` (§1, §5 P4–P6, §8), `fixes.md` (#2, #4),
`src/graph/build_graph.py`, `configs/sthgrec.yaml`, `configs/lightgcn.yaml`,
`src/eval/run_baselines.py`, `src/data/trust_signals.py`,
`src/sentiment/aspects.py`, `results/baselines/summary.csv`.

Legend for status of a decision: **[DECIDED]** here · **[SIGN-OFF]** needs the
lead (§6 escalation) · **[HARNESS]** depends on the parallel harness analysis.

---

## 0. Scope in one paragraph

ST-HGRec is LightGCN with three edge types instead of one. The interaction graph
is `~800K–900K` train interactions over tens of thousands of users and items; on
top of it sit **10 aspect nodes** and a sentiment layer on the order of the
`absa_normalized.csv` row count. The model keeps LightGCN's discipline: **no
feature transforms, no nonlinearities, no self-loops-with-weights** — just
normalized neighbourhood aggregation, layer-mean combination, and a dot-product
score. Everything ST-HGRec adds (sentiment, trust) enters as **structure or edge
weights on the propagation**, never as an MLP. That is the whole point of the
PRD's Option D: a minimal, ablatable, reproducible heterogeneous extension, not a
SOTA architecture.

---

## 1. Graph → tensors

### 1.1 Deterministic ID mapping (seed-independent)

Node string keys in the graph are `user:{raw}`, `item:{raw}`, `aspect:{raw}`
(confirmed in `build_graph.py`). We map each type to a **contiguous, sorted,
seed-independent** index space:

```
user_id2idx  = {raw: k for k, raw in enumerate(sorted(user_raw_ids))}
item_id2idx  = {raw: k for k, raw in enumerate(sorted(item_raw_ids))}
aspect2idx   = {raw: k for k, raw in enumerate(sorted(aspect_raw_ids))}
```

- **Sort, don't rely on insertion / graph-iteration order.** `MultiDiGraph`
  node order is insertion-order and stable, but sorting the *raw* IDs removes any
  dependency on how `build_graph` happened to add nodes, so the mapping is
  reproducible from the source CSVs alone.
- Aspects: sort the 10 canonical names lexicographically. There are exactly 10
  (`aspects.CANONICAL_ASPECTS`: gameplay, graphics, price, difficulty, story,
  controls, sound, multiplayer, performance, value); the mapping is tiny and fixed.
- **Persisted mapping file:** `data/processed/video_games/sthgrec_id_maps.json`
  (JSON, human-diffable, git-trackable — it is small). Schema:
  ```json
  {"n_users": 41234, "n_items": 17389, "n_aspects": 10,
   "user_id2idx": {...}, "item_id2idx": {...}, "aspect2idx": {...},
   "source": "video_games.train.inter", "built_from": "csv"}
  ```
  This file is the contract between the model, the evaluator, and the P5
  explanation hooks: any component that needs to translate a raw ID ↔ tensor
  index reads it. It is **not** regenerated per run; it is regenerated only when
  the underlying split changes, and is asserted-equal on load.

### 1.2 Build from CSVs, not the gpickle — **[DECIDED]**

Recommendation: **build adjacency directly from the source CSVs**
(`video_games.train.inter` for interactions, `absa_normalized.csv` for
sentiment, `trust_signals.csv` for trust), *not* from `graph.gpickle`.

Rationale:
- The gpickle is a `~90 MB` regenerable artifact that fix #5 explicitly
  untracked; the model should not depend on an untracked binary (PRD §4.4,
  reproducibility spine). The three CSVs are the real source of truth.
- The gpickle is a `MultiDiGraph` with **parallel edges** (one `sentiment` edge
  per `(u,i,aspect)` row). Reading it back and de-duplicating is strictly more
  work than reading the CSV and aggregating.
- Building from CSV means the exact same ID mapping and dedup policy the
  evaluator uses, with no networkx round-trip.

A thin loader `graph_to_tensors(inter_csv, sentiment_csv, trust_csv, id_maps)`
lives in `src/models/` (or a `src/graph/to_tensors.py`) and returns the typed
adjacency described below. It reads `train.inter` **only** (leakage guard,
matching `build_graph.py` and `trust_signals.py`).

### 1.3 Per-edge-type adjacency + dedup/aggregation policy

Three logical relations, each built as a sparse `edge_index` (+ optional weight):

| Relation | Endpoints | Source | Parallel-edge policy |
|----------|-----------|--------|----------------------|
| `interacts` | user ↔ item | `train.inter` | 5-core is already de-duplicated per `(u,i)`; if a repeat appears, **collapse to one edge** (implicit weight 1). Ratings/timestamps are *not* used as propagation weights (LightGCN uses binary adjacency). |
| `sentiment` | user ↔ item | `absa_normalized.csv` | Multiple rows per `(u,i)` with different aspects. **Aggregate to one signed weight per `(u,i)`** (see §2.3) — do **not** keep parallel edges. |
| `has_aspect` | item ↔ aspect | `absa_normalized.csv` | Multiple rows per `(i,a)`. **Aggregate to a signed/counted weight per `(i,a)`** (mean polarity, or count). |

All relations are made **symmetric (undirected)** for propagation — LightGCN
message-passes both directions. We store one directed `edge_index` per relation
and symmetrize at normalization time (add the transpose), exactly as LightGCN
builds its symmetric norm adjacency.

Normalization: symmetric `D^{-1/2} A D^{-1/2}` per relation, degree computed on
the symmetrized relation. Weighted relations (sentiment, trust) fold their weight
into `A` **before** computing `D` so the normalization sees the weighted degree.

---

## 2. Model

### 2.1 Embedding table layout — **[DECIDED]**

**One embedding table** `E ∈ R^{(n_users + n_items + n_aspects) × d}`, `d = 64`
(matches `configs/lightgcn.yaml` `embedding_size`). Row blocks, in order:

```
[ 0                         .. n_users )                      → users
[ n_users                   .. n_users+n_items )              → items
[ n_users+n_items           .. n_users+n_items+n_aspects )    → aspects   (10 rows)
```

Global index = local index + block offset. A single table (a) lets one sparse
propagation op touch all node types, (b) keeps LightGCN's "just an embedding
table" simplicity, (c) makes the aspect embeddings first-class so their signal
flows into items and thence users. Xavier/normal init at seed 2020.

### 2.2 Layer-wise heterogeneous propagation

`n_layers = 3` (from `configs/lightgcn.yaml`). At each layer we propagate over the
union of the active relations and sum the per-relation messages (LightGCN-style,
no weights-per-layer, no transform):

```
E^{(0)} = E
for l in 1..L:
    E^{(l)} = Â_interacts · E^{(l-1)}
            + λ_s · Â_sentiment  · E^{(l-1)}      # only if use_sentiment_edges
            + λ_a · Â_has_aspect · E^{(l-1)}      # only if use_sentiment_edges
            + λ_t · Â_trust      · E^{(l-1)}      # only if use_trust_edges (see §3)
E_final = mean(E^{(0)}, E^{(1)}, ..., E^{(L)})     # LightGCN layer combination
```

- Each `Â_*` is the symmetric-normalized (weighted) adjacency of that relation
  over the **single global node index space** (§2.1), so one `spmm` per relation.
- `λ_s, λ_a, λ_t` are **fixed scalars = 1.0 by default** (a relation contributes
  at its normalized magnitude). They are *not* learned — keeping them constant
  preserves LightGCN's parameter-freedom-beyond-embeddings and keeps ablations
  clean (a flag flips the term on/off; the scalar just lets the lead tune
  relative strength without retraining code). Expose them in `configs/sthgrec.yaml`
  as optional keys (`sentiment_weight`, `aspect_weight`, `trust_weight`, all
  default 1.0) — **[SIGN-OFF]** on whether to expose the knobs at all.
- **No self-loops beyond LightGCN's convention, no nonlinearity, no dropout on
  messages.** Aspect nodes have no direct user edges; aspect signal reaches users
  only through item → aspect → item paths across layers, which is why `n_layers ≥ 2`
  matters for the aspect layer to do anything.

### 2.3 Sentiment contribution (concrete) — **[DECIDED]**

PyABSA emits `Positive / Negative / Neutral`. Map polarity to a **signed edge
weight**:

```
Positive → +1
Negative → -1
Neutral  →  0     # contributes no message; effectively drops the edge
```

Per `(u,i)` sentiment edge weight = **mean of the signed polarities across that
pair's aspects** (so a pair that is positive on gameplay but negative on price
nets out), clipped to `[-1, +1]`. Per `(i,aspect)` `has_aspect` weight = mean
signed polarity of that aspect for that item.

Signed weights survive `D^{-1/2} A D^{-1/2}` fine (degree uses `|weight|` so
negatives don't cancel the normalizer — degree is computed on absolute weights,
`Â` keeps the sign). Neutral → 0 means Neutral edges are structurally present in
`absa_normalized.csv` but contribute nothing; this is deliberate and documented,
not a silent drop.

Rationale for signed-scalar over relation-per-aspect: 10 aspect-specific relations
would 10× the number of sparse ops and the ablation surface for marginal benefit;
the aspect *identity* is still preserved via the `has_aspect` relation and the
aspect embedding block, which is what P5 needs to cite per-aspect facts (§7). If
the lead wants per-aspect propagation later, the `has_aspect` relation already
carries the aspect index, so it is a non-breaking extension.

### 2.4 Scoring

Standard LightGCN dot product on the final embeddings:

```
score(u, i) = E_final[user_offset + u] · E_final[item_offset + i]
```

Full-ranking eval scores a user against all items. No aspect/trust term enters
the score directly — their effect is already baked into `E_final` through
propagation (§3 discusses why the trust-at-scoring option is rejected for
ablation cleanliness).

---

## 3. Trust integration — fix #4 decision **[SIGN-OFF]**

`trust_signals.py` produces **one scalar `trust_score` per user** (mean of
verified-purchase share, normalized helpfulness, rating consistency; range
roughly `[0,1]`). We have **no social graph** — there are no user→user relations
anywhere in the data. Options:

### Option 4A — trust as a propagation weight (adjacency reweighting)
Scale each user's `interacts` edges by their `trust_score`: a high-trust user's
interactions push harder on the items they touched. Concretely, multiply the
user→item entries of `A_interacts` by `trust_score[u]` before normalization (or
add a parallel trust-weighted copy gated by `use_trust_edges`).
- **Leakage safety:** trust is a per-user *prior* computed over the user's full
  review history (`trust_signals.py` documents mild temporal leakage but no
  per-`(u,i)` label leakage). As an edge *weight* it never touches held-out
  labels. Safe.
- **Ablation cleanliness:** `use_trust_edges: false` → weight reverts to 1.0 →
  identical to no-trust. **Cleanly zeroable in one config line.** Best of the three.
- **Cost:** low — one element-wise scale on an existing tensor.
- **Interpretability for P5:** the "trust fact" is exactly `trust_score[u]` and
  the fact "this user's interactions were up/down-weighted by X" — directly
  citable.

### Option 4B — trust as typed edges
There is **no user→user social graph**, so an honest typed-trust relation could
only be a redundant `user --trust--> item` parallel copy of `interacts` weighted
by trust — which is *literally Option 4A expressed as a second relation*, or a
fabricated user→user graph (thresholded trust similarity) that the data does not
support and that risks inventing structure. Be honest: 4B's only non-fabricated
form collapses into 4A with extra bookkeeping.
- Leakage: same as 4A if item-linked. Ablation: cleanly gated. Cost: higher (a
  whole extra relation + normalization). Interpretability: no gain over 4A.

### Option 4C — trust as a scalar node feature at scoring time
Add/concat `trust_score` at scoring: e.g. `score += γ · trust_score[u]`.
- **Leakage:** safe (prior).
- **Ablation cleanliness:** zeroable (`γ=0`), but it shifts *every* candidate for
  a user by a constant, so it cannot change *ranking within a user* — it is inert
  for per-user top-K. Effectively does nothing for the metric. Bad.
- **Interpretability:** a bolt-on scalar, disconnected from message passing —
  weak "trust fact" story.
- Violates the "no feature transform" spirit if it grows beyond a scalar add.

### Recommendation: **4A** (trust as interaction-edge propagation weight).
It is the only option that (i) is faithful to the data we actually have (no
social graph), (ii) zeroes cleanly under `use_trust_edges` for P6, (iii) affects
ranking, and (iv) yields a directly citable trust fact for P5 at near-zero
implementation cost. Flagged **[SIGN-OFF]** per PRD §6 (trust representation is
expensive-to-reverse).

---

## 4. Training

- **Loss:** BPR (Bayesian Personalized Ranking), same objective as the RecBole
  baselines, over `train.inter` positives. `L = -Σ log σ(score(u,i⁺) - score(u,i⁻)) + reg_weight·‖E‖²`.
- **Negative sampling:** uniform 1 negative per positive per step (RecBole
  default for BPR/LightGCN), sampled from items the user has **not** interacted
  with in train. Sampler seeded at 2020.
- **Hyperparameters — start from `configs/lightgcn.yaml` for comparability:**
  `embedding_size 64`, `n_layers 3`, `reg_weight 1e-4`, `train_batch_size 2048`,
  `learning_rate 0.001`, `epochs 300`, Adam optimizer, `valid_metric NDCG@10`,
  early stopping on validation NDCG@10 (RecBole default patience). These are
  already mirrored in `configs/sthgrec.yaml`. **Any deltas → [SIGN-OFF]** (§10).
- **Determinism (seed 2020):** set `init_seed(2020, reproducibility=True)` (the
  same call `run_baselines.py` uses). Threats and mitigations:
  - *Sparse `spmm` / scatter-add non-determinism on GPU* — use deterministic
    algorithms (`torch.use_deterministic_algorithms(True)` where the harness
    allows; RecBole's `reproducibility: true` sets cuDNN deterministic). If a
    custom `torch.sparse` path is chosen [HARNESS], verify `spmm` is deterministic
    or fall back to CPU for the sanity gate.
  - *Negative-sampler RNG* — seed the sampler explicitly, not just the global RNG.
  - *Python set / dict iteration for ID maps* — eliminated by the **sorted**
    mapping (§1.1).
  - *DataLoader worker shuffling* — seed workers, or `num_workers=0` for the
    reproducibility gate.
  - **Acceptance:** two runs at seed 2020 produce bit-stable (or within-tolerance
    on GPU) metric rows — same bar as the baselines gate.

---

## 5. Evaluation

- **Identical split + metrics as baselines.** ST-HGRec evaluates on the **same
  official 5-core benchmark split** via `benchmark_filename: [train, valid, test]`
  (RecBole `Dataset.build()` early-returns and partitions by file size, bypassing
  `eval_args` splitting — documented in `configs/popularity.yaml`). Metrics:
  `Recall@{10,20}`, `NDCG@{10,20}`, `mode: full` (full-ranking over the whole item
  set). Same `topk`, same `valid_metric NDCG@10`.
- **Output row into `results/baselines/summary.csv`.** Confirmed schema:
  ```
  model,seed,recall@10,recall@20,ndcg@10,ndcg@20
  ```
  ST-HGRec emits exactly this row shape (`model=STHGRec, seed=2020, ...`) so it
  slots in alongside `Pop / BPR / LightGCN / NGCF`. Current baseline reference
  (for sanity ordering): LightGCN `recall@10=0.0865, ndcg@10=0.047`; ST-HGRec
  should land **≥ LightGCN** to justify the added structure, but the PRD bar is
  "solid and reproducible," not "beats LightGCN."
- **Runner integration — [DECIDED, harness-aware]:** add a **dedicated
  `src/eval/run_sthgrec.py`** rather than extending `run_baselines.py`'s
  `MODEL_CLASSES`. Reasons: (i) `MODEL_CLASSES` maps to RecBole
  `general_recommender` classes constructed as `cls(config, train_data.dataset)`;
  ST-HGRec builds its own typed adjacency from CSVs and may not be a RecBole model
  class at all under a standalone-PyG or custom-`torch.sparse` harness [HARNESS];
  (ii) a separate entrypoint keeps the baseline runner (a reproducibility-critical
  audit file, fix #8) untouched; (iii) it lets ST-HGRec own its graph-tensor
  construction and still call the **same evaluation + summary-append helpers**.
  Refactor the summary-writing + metric-flattening logic in `run_baselines.py`
  into a shared `src/eval/_report.py` (or import it) so both runners write the
  identical CSV schema — the only shared surface. **[HARNESS]** If `env-harness`
  picks RecBole-GNN and ST-HGRec ends up expressible as a RecBole model class,
  `run_sthgrec.py` can still reuse RecBole's `Trainer`/`evaluate` verbatim; if it
  picks standalone PyG or custom sparse, `run_sthgrec.py` owns its own train/eval
  loop but reuses the **RecBole dataset + split loader** for metric parity.

---

## 6. Ablations (P6 preview)

The two flags already exist in `configs/sthgrec.yaml`:
`use_sentiment_edges: true`, `use_trust_edges: true`.

Flow: `run_sthgrec.py` reads the config → passes the two booleans into
`graph_to_tensors(...)` / the propagation step → each `false` **omits that
relation's term from §2.2's sum** (does not just zero a learned weight; the
sparse op is skipped entirely, so switch-off is exact and cheap).

```
use_sentiment_edges=false  → drop Â_sentiment AND Â_has_aspect terms
use_trust_edges=false      → drop the trust reweighting (interacts weight → 1.0)
```

Each is a **one-line config change**; no code edits per ablation.

**P6 run matrix:**

| Run | use_sentiment_edges | use_trust_edges | Purpose |
|-----|---------------------|-----------------|---------|
| full        | true  | true  | the headline ST-HGRec number |
| −sentiment  | false | true  | isolates the sentiment contribution |
| −trust      | true  | false | isolates the trust contribution |
| −both       | false | false | ≈ plain heterogeneous LightGCN (interacts only) → should ≈ the LightGCN baseline, a sanity anchor |

Results to `results/ablations/` (per PRD §5 P6), same metric schema.

---

## 7. Explanation hooks (P5 dependency)

`honest.py` must be able to extract, for a `(user, item)` recommendation, the
facts already in the graph — no new claims. The trained model exposes:

```python
def explain_pair(self, user_raw_id, item_raw_id) -> dict:
    """Per-recommendation facts for the honest generator. No LLM here.
    Returns only facts derivable from the graph + trained embeddings."""
    return {
        "score": float,                 # score(u,i)
        "aspect_contributions": {       # per canonical aspect, this item's signed sentiment
            "gameplay": +1.0, "price": -1.0, ...   # from has_aspect weights for this item
        },
        "supporting_sentiment_edges": [ # the actual absa_normalized rows behind the item
            {"aspect": "gameplay", "sentiment": "Positive"}, ...
        ],
        "user_trust": {                 # the trust fact (Option 4A)
            "trust_score": 0.72,
            "role": "interaction edges up-weighted by trust_score",
        },
    }
```

```python
def aspect_item_polarity(self, item_raw_id) -> dict[str, float]:
    """The signed has_aspect weights for an item (drives 'this game is praised
    for X, criticized for Y'). Reads from the same aggregation §2.3 uses."""

def user_neighborhood(self, user_raw_id, top_k=10) -> list[tuple[str, float]]:
    """Items most responsible for the user's embedding (for 'because you liked …').
    Optional; supports the honest rationale."""
```

Key requirements the model must satisfy for P5:
- **Per-aspect contribution scores** are read straight from the aggregated
  `has_aspect` signed weights (§2.3) + the aspect embedding block — no separate
  computation, so the explanation cites the *same* structure the model trained on
  (faithful by construction).
- **Supporting sentiment edges** come from `absa_normalized.csv` filtered to the
  item, giving `honest.py` verbatim `(aspect, sentiment)` facts to rephrase.
- **Trust context** is the scalar `trust_score` + its role (4A) — a citable fact.
- All lookups go through `sthgrec_id_maps.json` (§1.1) for raw↔index translation.
- The **faithfulness check** (P5, remove-a-reason) drops a cited `has_aspect`
  edge, rebuilds the adjacency, re-scores, and checks the score moves — 4A/§2.3
  make this a clean edge deletion.

---

## 8. Risks & de-risk order

**Memory estimate (fp32).** Embedding table:
`(≈41K users + ≈17K items + 10 aspects) × 64 × 4 bytes ≈ 15 MB` — trivial.
Sparse adjacencies: `interacts ≈ 900K` non-zeros symmetrized `≈ 1.8M`, sentiment
on the order of the ABSA rows, `has_aspect ≤ n_items × 10`. Each COO `edge_index`
is `~2 × nnz × 8 bytes`; total sparse footprint is **well under 1 GB**. Activations
are `n_nodes × d` per layer `≈ 15 MB × (L+1)`. **This fits comfortably on a single
modest GPU or even CPU** — scale is not the risk here.

**Real risks (de-risk in this order):**
1. **Aspect-node degree imbalance — `price`/`gameplay` will dominate.** A handful
   of high-frequency aspects will have enormous degree; symmetric `D^{-1/2}`
   normalization already damps this, but smoke-test the aspect degree distribution
   and confirm no aspect swamps propagation. **Smoke-test first.**
2. **Sentiment edge sanity.** After fix #3, confirm the sentiment/`has_aspect`
   edge counts are realistic (thousands+), not the old 4-edge placeholder, before
   trusting any ablation.
3. **Cold-start users/items in valid/test.** The graph is train-only (leakage
   guard). Valid/test users/items absent from train get the plain (untrained-in-
   context) embedding; full-ranking eval handles this the same way LightGCN
   baselines do, so it is *comparable*, but verify no crash on unseen IDs (map
   returns nothing → they simply aren't scorable, matching baseline behavior).
4. **Determinism of sparse ops** (§4) — verify the seeded rerun is stable before
   reporting any headline number.
5. **`−both` ≈ LightGCN baseline** — the strongest single sanity check: with both
   flags off, ST-HGRec is heterogeneous-LightGCN-with-only-interacts, which should
   reproduce the LightGCN baseline row within tolerance. If it doesn't, the graph→
   tensor path is wrong. **Build this equivalence check first.**

---

## 9. Non-goals guard

This design deliberately excludes (PRD §6 / `IMPLEMENTATION_PROMPT.md` §8):
**no LLM anywhere in the recommender** — the model is pure sparse propagation +
dot product, and the P5 explanation hooks return *facts only* (rephrasing happens
in `honest.py`, never here); **no learned feature transforms, no nonlinearities,
no attention** — keeping LightGCN's Option-D simplicity is the point, so we do not
drift toward a general HGNN; **no LLM/RLMRec-style graph augmentation**; **no
per-aspect learned relation weights** (fixed scalars only) unless the lead asks;
**no SOTA-chasing** — ST-HGRec must be solid, reproducible, and ablatable, and
landing at-or-modestly-above LightGCN is success, not a failure to beat published
numbers. Datasets stay Video-Games-only; no Beauty until the pipeline is
end-to-end.

---

## 10. Sign-off needed (lead checklist)

Before `model-core` implements P4, the lead must confirm:

- [ ] **Trust representation (fix #4):** approve **Option 4A** (trust as
      `interacts`-edge propagation weight). Alternatives 4B/4C analyzed in §3 and
      rejected. — *expensive-to-reverse, PRD §6.*
- [ ] **Harness (fix #2 / T4.1):** RecBole-GNN vs standalone PyG vs custom
      `torch.sparse`-in-vanilla-RecBole. **Deferred to the parallel `env-harness`
      analysis** — see that agent's report. This design is harness-agnostic; only
      §5 (runner form: RecBole `Trainer` reuse vs own train/eval loop) and §4
      (which `spmm` determinism knob) change with the pick. Sign both off together.
- [ ] **Hyperparameter deltas:** default is *identical to `configs/lightgcn.yaml`*
      (d=64, L=3, reg=1e-4, bs=2048, lr=1e-3, epochs=300, seed=2020) for
      comparability. Approve as-is, or specify deltas.
- [ ] **Optional relation weights** `sentiment_weight / aspect_weight /
      trust_weight` (default 1.0): expose as config knobs, or hard-code to 1.0 and
      keep the config surface minimal?
- [ ] **ID-map file tracked in git?** `sthgrec_id_maps.json` is small and is the
      raw↔index contract; recommend tracking it (unlike the gpickle). Confirm.

---

*Once signed off, implementation fills `src/models/sthgrec.py` (currently a stub),
adds `src/eval/run_sthgrec.py`, and wires `graph_to_tensors`. No code is written
before sign-off.*
