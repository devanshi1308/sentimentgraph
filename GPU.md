# Running the pipeline on GPU

The whole stack is **PyTorch** (RecBole for the recommender, PyABSA +
HuggingFace `transformers` for sentiment). No TensorFlow anywhere. "GPU" here
means **NVIDIA CUDA** — Apple MPS is not usable for ST-HGRec (no sparse-tensor
kernels) and the code falls back to CPU on a Mac automatically.

| Stage | Script | GPU path |
| --- | --- | --- |
| Baselines (Pop/BPR/LightGCN/NGCF) | `src/eval/run_baselines.py` | RecBole native — uses CUDA whenever `torch.cuda.is_available()` |
| ABSA extraction | `src/sentiment/run_absa.py` | PyABSA — `--device auto` picks CUDA; `--batch-size 256` for throughput |
| ST-HGRec + ablations | `src/eval/run_sthgrec.py`, `src/eval/ablations.py` | `torch.sparse.mm` on CUDA; adjacencies moved to the model device |

## 1. Install the CUDA build of PyTorch

The pinned `requirements-*.txt` install the **CPU** wheel of torch. On a CUDA
box, install the matching CUDA wheel *after* the requirements file, e.g. for
CUDA 12.1:

```bash
# recommender env
uv venv env-rec --python 3.11 && source env-rec/bin/activate
uv pip install -r environments/requirements-rec.txt
uv pip install --force-reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# sentiment env
uv venv env-absa --python 3.11 && source env-absa/bin/activate
uv pip install -r environments/requirements-absa.txt
uv pip install --force-reinstall torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -m spacy download en_core_web_sm
```

**Windows (PowerShell)** — same commands, only activation differs (`bin/activate`
-> `Scripts\Activate.ps1`; if the script is blocked, run PowerShell as admin once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`):

```powershell
# recommender env
uv venv env-rec --python 3.11 ; env-rec\Scripts\Activate.ps1
uv pip install -r environments/requirements-rec.txt
uv pip install --force-reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# sentiment env
uv venv env-absa --python 3.11 ; env-absa\Scripts\Activate.ps1
uv pip install -r environments/requirements-absa.txt
uv pip install --force-reinstall torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -m spacy download en_core_web_sm
```

CUDA itself needs no Windows-specific handling: install the matching NVIDIA
driver (WDDM, ships with the normal GeForce/Studio/Data Center driver — no WSL
required), and the `cu121` wheels above work identically on native Windows.

`torch-geometric==2.6.1` is a pure-Python wheel — it needs no CUDA-specific
build. Check the install:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Baselines

No change needed — the configs now carry `use_gpu: true` / `gpu_id: 0`, and
RecBole downgrades to CPU on its own if no CUDA device is visible.

```bash
for c in popularity bpr ngcf lightgcn; do
  python -m src.eval.run_baselines --config configs/$c.yaml
done
```

Pick a specific GPU with `gpu_id: 2` in the YAML, or
`CUDA_VISIBLE_DEVICES=2 python -m ...` (bash) /
`$env:CUDA_VISIBLE_DEVICES=2; python -m ...` (PowerShell) /
`set CUDA_VISIBLE_DEVICES=2 && python -m ...` (cmd.exe).

## 3. ABSA extraction (the long pole)

`FAST_LCF_ATEPC` runs ~9 reviews/s on CPU (~16-20 h for the full ~748K set);
a single modern CUDA GPU is roughly 10-50x faster (~1-2 h). The job is
checkpointed and resumable, so it is safe to interrupt.

```bash
source env-absa/bin/activate   # Windows: env-absa\Scripts\Activate.ps1
python -m src.sentiment.run_absa --device auto --batch-size 256
```

- `--device` : `auto` (default, picks CUDA if present), `cpu`, `cuda`, `cuda:N`,
  or `allcuda` (shard across every visible GPU).
- `--batch-size` : reviews per checkpoint chunk. 32 is a safe CPU default; use
  ~256 on GPU so checkpoint I/O does not dominate. PyABSA batches internally on
  top of this.
- Resume: delete nothing — rerun the same command; it reads
  `data/interim/absa_full_progress.txt` and continues.

Then normalize (CPU, fast):

```bash
python -m src.sentiment.normalize_aspects
```

## 4. ST-HGRec and the ablation matrix

`configs/sthgrec.yaml` now sets `use_gpu: true` / `gpu_id: 0` (previously pinned
to `device: cpu`). The three normalized adjacencies are built on CPU by
`graph_tensors` and moved onto the model device in `STHGRec.__init__`.

**Before trusting a headline number on a new GPU, run the determinism check:**

```bash
source env-rec/bin/activate   # Windows: env-rec\Scripts\Activate.ps1
python -m src.eval.run_sthgrec --config configs/sthgrec.yaml --check-determinism
```

It builds the model and runs two seeded forward+backward passes on the resolved
device, asserting bit-identical embeddings *and* gradients. This is expected to
pass on CUDA: the adjacencies are frozen buffers (`requires_grad=False`), so the
only backward path through `torch.sparse.mm` is a second structured spmm, which
cuSPARSE computes deterministically for a fixed sparsity pattern. If it ever
fails, fall back to `use_gpu: false` for the reported run and file it.

Full run + ablations:

```bash
python -m src.eval.run_sthgrec --config configs/sthgrec.yaml
python -m src.eval.ablations --config configs/sthgrec.yaml
```

Memory is not a constraint — the embedding table is ~15 MB and all sparse
adjacencies are well under 1 GB, so any CUDA card with a few GB of VRAM is
enough. The bottleneck is kernel availability, not capacity.

## 5. Apple Silicon (MPS) note

`run_sthgrec.py` / `ablations.py` detect a requested `mps` device, warn, and
force CPU — ST-HGRec's sparse propagation has no MPS kernel. Baselines and ABSA
can use MPS where their libraries support it, but there is no MPS path for the
heterogeneous graph model. Use a CUDA host for GPU acceleration of ST-HGRec.

## 6. Windows note

There is no Windows-specific fallback because none is needed: MPS is an
Apple-only backend, so on Windows `resolve_device` only ever sees `cuda` or
`cpu`, and the CUDA path is native (no WSL required — install the normal
Windows NVIDIA driver + `cu121` wheels per §1). Every script here is invoked
as `python -m ...` and uses `pathlib` for all paths, so nothing else in the
pipeline changes on Windows. The only per-OS difference anywhere in the repo
is how you *activate* a venv — see the PowerShell commands in §1 and
`README.md`.
