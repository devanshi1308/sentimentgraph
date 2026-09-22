# Testing Guide — run this before every push (Devanshi)

The repo has a **24-test suite that runs in ~4 seconds on synthetic data** — no
dataset downloads, no GPU, no long jobs. It is the gate that catches broken wiring
before it reaches the shared branch. **Green suite → push. Red suite → fix or ping
Muaaz. Never push red.**

This guide covers code testing. Artifact regeneration (the long data jobs) lives in
`P2_RUNBOOK.md` — different document, different purpose.

---

## 0. One-time setup (~5 min)

The tests run in the **rec env** plus `pytest`:

```bash
cd st-hgrec
source env-rec/bin/activate        # built from environments/requirements-rec.txt
uv pip install pytest              # test runner only; not a pinned runtime dep
python -c "import torch, recbole, torch_geometric, pandas; print('test deps OK')"
```

Windows (PowerShell): `env-rec\Scripts\Activate.ps1` instead of `source ... activate`;
everything else on this page (`python -m pytest ...`) is identical.

If `torch_geometric` fails to import: your env predates the P4 pins — rebuild it
(`P2_RUNBOOK.md` step 1). Never run tests from the absa env — they import the
recommender side.

---

## 1. The one command

From the **repo root** (imports are `src.`-relative — running from elsewhere breaks
them):

```bash
python -m pytest tests/ -q
```

Expected: `24 passed` in a few seconds. That's the whole gate.

Two useful variants:

```bash
python -m pytest tests/ -q -p no:warnings   # hide RecBole's deprecation noise
python -m pytest tests/ -x -q               # stop at first failure (debug loop)
```

---

## 2. What the suite actually checks (so failures mean something to you)

### `tests/test_sthgrec.py` — the model core (11 tests)

| Test | What breaks if it fails |
|---|---|
| `test_parity_lightgcn` | Propagation math no longer matches vanilla LightGCN — someone changed normalization/layer-combine |
| `test_signed_sentiment_*` / `test_degree_uses_abs_weight` | Negative sentiment no longer flows with correct sign, or degrees stopped using \|w\| |
| `test_trust_scales_interactions` / `test_trust_off_restores_untrusted` | Trust-4A weighting broken, or the trust ablation switch leaks |
| `test_sentiment_off_equals_no_file` | Sentiment ablation is no longer *exactly* equivalent to having no sentiment data |
| `test_determinism` | Seed-2020 reproducibility lost — a nondeterministic op crept in |
| `test_end_to_end_smoke` | The RecBole harness path (config → dataset → train → eval → summary row) broke |
| `test_explanation_hook` / `test_remove_reason_changes_contribution` | The P5 explanation API contract changed |
| `test_yaml_ablation_flags_win` | **The config bug regression test.** YAML flags stopped reaching the model — this silently corrupts every ablation run, which is exactly why it exists |

### `tests/test_ablations.py` — the switch-off driver (4 tests)

Schema/finiteness of `results/ablations/summary.csv` rows, structural proof that
`-sentiment` really drops the adjacency, append-without-duplicate-headers, and
run-twice determinism.

### `tests/test_explain.py` — explanations + faithfulness (9 tests)

Honest facts exactly match the model hooks; rendered text cites **only** real
aspects (anti-hallucination); determinism; the LLM seam stays injected-only;
**ungrounded quarantine** (an AST scan proving no recommender module imports the
study stimulus — if you ever wire `ungrounded` into anything under `src/models`,
`src/eval`, `src/graph`, `src/data`, `src/sentiment`, this fails, on purpose);
faithfulness sensitivity (removing a cited reason changes the score) and grounding
scores.

---

## 3. Debug loop for a failure

```bash
python -m pytest tests/ -x -q                      # find the first failure
python -m pytest tests/test_sthgrec.py -q          # narrow to one file
python -m pytest tests/ -k "trust" -q              # narrow by keyword
python -m pytest tests/test_sthgrec.py::test_determinism -q -l   # one test, show locals
```

Rules when red:

1. **Read the assertion message first** — the tests assert with context.
2. `git stash && python -m pytest tests/ -q` — if it's green stashed, *your diff*
   broke it; `git stash pop` and bisect your change.
3. **Never** mark a failing test `skip`/`xfail`, loosen a tolerance, or delete an
   assertion to get green. Those thresholds encode research-correctness decisions
   (parity, bitwise ablation equality, determinism). If a test seems *wrong*, ping
   Muaaz with the full output — changing test semantics is his call.
4. Env-shaped failures (`ModuleNotFoundError`, RecBole import errors) → rebuild the
   env from the pinned file before debugging code.

---

## 4. If you add or change code, add/extend a test

Patterns to copy (look at the existing files):

- **Synthetic fixtures in temp dirs, never repo `data/`.** Every test builds its
  tiny dataset in `tmp_path`/`TemporaryDirectory`. A test that writes into the repo
  tree will be rejected — after `pytest`, `git status` must be unchanged.
- **Deterministic seeds.** Anything stochastic gets seed 2020 and, where it
  matters, an identical-across-two-runs assertion.
- **Shared toy builder:** `tests/fixtures.py::build_toy_recbole` (used by
  `test_explain.py`). `test_sthgrec.py`/`test_ablations.py` keep their own inline
  helpers on purpose — don't unify them without asking.
- New pipeline behavior (a flag, a path, a schema) needs a test that would **fail
  on the old behavior** — that's what made `test_yaml_ablation_flags_win` valuable.

---

## 5. Pre-push checklist (every time, ~1 minute)

```bash
# 1. Suite green, from repo root, rec env
python -m pytest tests/ -q                     # -> 24 passed (or more, never fewer)

# 2. Nothing forbidden staged: data artifacts are gitignored — NEVER git add -f them
git status --short                             # only code/docs/results-tables changes

# 3. No accidental big file (largest tracked file should be ~KB-scale)
git ls-files -z | xargs -0 du -k 2>/dev/null | sort -rn | head -3
# Windows (PowerShell), same check:
#   git ls-files | ForEach-Object { Get-Item $_ } | Sort-Object Length -Descending | Select-Object -First 3 FullName, Length

# 4. Up to date with the remote before you push
git pull --rebase origin devanshi

# 5. Conventional commit subject: feat|fix|chore|docs|refactor|results: ...
git log -1 --format=%s
```

Then push. What's allowed in a commit, by directory:

| Path | Commit? |
|---|---|
| `src/`, `tests/`, `configs/`, `*.md` | ✅ yes |
| `results/baselines/summary.csv`, `results/ablations/summary.csv` | ✅ yes (small tables are the point) |
| `results/figures/*.png` | ✅ only if <1 MB (reduce dpi otherwise) |
| `data/raw/`, `data/interim/`, `data/processed/`, `*.gpickle` | ❌ never — gitignored, regenerable |
| `env-rec/`, `env-absa/`, checkpoints, logs | ❌ never |

If a config under `configs/` needs to change — ping Muaaz first. Every number in
the paper traces to those files; they don't change casually.

---

## 6. What the suite does NOT cover (know the gap)

- **Real-data behavior** — row counts vs the official split, ABSA coverage, real
  metric values. That's `P2_RUNBOOK.md`'s acceptance checks; the suite proves the
  machinery, the runbook proves the artifacts.
- **The `−both ≈ LightGCN` sanity anchor** — checked numerically only on the real
  ablation run, against the LightGCN row in `results/baselines/summary.csv`.
- **The absa env** — `run_absa.py` runs there and has no pytest coverage (it's a
  batch job with checkpoints). Its sanity check is the runbook's coverage report.

There is no CI. This suite, run by you, **is** the CI. Green → push.
