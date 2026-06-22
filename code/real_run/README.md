# FAITH-SAE — `code/real_run/` (the publication run)

### Author: Rajia Rani · ``

This folder is the **real, GPU-scale** counterpart of the FAITH-SAE teaching
scaffold. It runs the **same study, same code shapes, same scoring math** as the
toy path — but on a **real frozen CLIP ViT-L/14 backbone** and **real ImageNet +
OOD data**, to produce the numbers that go in the paper.

> **One question (see `../../DESIGN_BRIEF.md`):** are the concept directions a
> Sparse Autoencoder finds inside a frozen vision model **causally faithful** —
> monotone, specific, sufficient — and **do they stay faithful as the input
> images shift out of distribution**? We answer it with a single **Causal
> Faithfulness Score (CFS)** measured across an OOD ladder, comparing
> **on-manifold steering (ours)** against naive / random / clamp / supervised.

---

## The toy → real story (same code, real config + real data)

FAITH-SAE ships in two layers that share **one set of interfaces and one scoring
model**, so nothing has to be re-derived when you scale up:

| | Toy scaffold (`../../src/`, `../milestone_*/`) | **This folder (`real_run/`)** |
|---|---|---|
| Backbone | shape-matched **synthetic** activations | **frozen CLIP ViT-L/14** (`open_clip`, `laion2b_s32b_b82k`), also ViT-B/16, ViT-H/14 |
| Data | a planted-concept Gaussian bank | **ImageNet-1k** train + **ImageNet-R / Sketch / C / ObjectNet** OOD ladder |
| SAE | tiny TopK over `d=64` | **TopK**, `n_features = expansion·d` (e.g. 32768), trained on **~300M–1B patch-token activations** |
| Steering | `naive/random/clamp/onmanifold` (toy) | the **same four steerers**, same semantics, at scale |
| Score | `src.utils.cfs_score` (harmonic mean) | **the exact same `src.utils.cfs_score`** (imported, not re-implemented) |
| Compute | CPU, seconds | **1× A100/H100 80GB, ~15–120 GPU-hours** |

The crucial reuse: every module here adds `../../` to `sys.path` and imports
`src.utils.cfs_score` and the steering **semantics** from the toy `src/`, then
re-implements only the **heavy** parts (streaming extraction, large-SAE training,
SVD manifold basis) for scale. **The faithfulness math is identical**, so a
result that holds in the smoke path holds, in shape, in the real run.

---

## File map

```
code/real_run/
  README.md                 <- you are here (what this is + quickstart)
  RUN_AT_SCALE.md           <- hardware, GPU-hour/$ budget, dataset downloads, full command sequence
  requirements_real.txt     <- real deps (open_clip, datasets, ...) + smoke-vs-real split
  cost_estimate.py          <- runnable closed-form GPU-hour + $ estimator (CLI)
  run_all_real.sh           <- ordered pipeline driver: extract->...->analysis, with --smoke
  __init__.py               <- marks the package (sibling imports)
  .gitignore                <- ignore the heavy cache/ + outputs/ artifacts (keep only .gitkeep)
  configs/
    vit_l14.yaml            <- full-scale config (ViT-L/14, full ladder)    [THE real config]
    vit_b16.yaml            <- cheaper backbone variant (ViT-B/16, d=768)
    vit_h14.yaml            <- largest backbone variant (ViT-H/14, d=1280)
    smoke.yaml              <- tiny CPU config (synthetic-shaped tensors)
  cache/                    <- activation shards (acts_*.npy, labels_*.npy, manifest_*.json); git-ignored
  outputs/                  <- sae.safetensors, U_r.npy, *.csv, fig*.png; git-ignored

  # pipeline modules (authored by sibling agents; this folder drives them):
  data_real.py          build_backbone / iter_image_batches / extract_activations / iter_activation_shards
  extract_activations.py  CLI entry: run the backbone over one dataset -> cache shards
  sae_real.py           TopKSAE + sae_loss + normalize_activations + save/load
  train_sae.py          streaming AdamW+warmup+AMP trainer, FVU/L0/dead% metrics
  manifold.py           estimate_manifold_basis (SVD) -> U_r ; project_onmanifold
  steering_real.py      naive_/random_/clamp_/onmanifold_steer registry + offmanifold_residual
  concept_select.py     max_activating_images / reliability_score / select_concepts
  probes.py             train_linear_probe / probe_readout
  cfs_eval.py           compute_cfs / evaluate_all_methods (library; imported by the sweeps)
  ood_sweep.py          run_ood_sweep over the OOD ladder -> ood_cfs_sweep.csv (RQ1 clean rung + RQ3)
  ablations_real.py     A1-A5 -> ablations.csv
  analysis_real.py      bootstrap_ci + findings (bootstrap_ci.csv, FINDINGS.md)
  figures_real.py       make_real_figures (fig1_cfs_ood_sweep.png, fig7_by_method_bar.png, ...)
```

> **Note on entry points.** `data_real.py` is a *library* (no standalone CLI) —
> extraction is driven per-dataset by **`extract_activations.py`**. `cfs_eval.py`
> is likewise a *library* (`compute_cfs` / `evaluate_all_methods`) that
> `ood_sweep.py` and `ablations_real.py` import; the **clean-image RQ1 CFS is the
> `clean` rung of `ood_sweep`**, so there is no separate "cfs on clean" script.
> `analysis_real.py` / `figures_real.py` consume the result **CSVs** (not the
> config), so they take `--results-dir` / `--out-dir`. The driver
> `run_all_real.sh` wires all of this in the correct order for you.

The **shared interface** (config schema, the `cache/acts_{dataset}_{shard}.npy`
activation-cache format, and every module signature) is the contract that lets
these modules import each other cleanly. See `RUN_AT_SCALE.md` and the
`DESIGN_BRIEF.md` for the full contract.

---

## Quickstart

### Now, on this CPU box — the smoke path (no GPU, no `open_clip`, no downloads)

The smoke path runs every stage on **synthetic-but-real-SHAPED** tensors so you
can prove the modules wire together end-to-end before renting a GPU. It needs
only the core scientific-Python stack (already present as `/usr/bin/python3`);
`open_clip` is import-guarded so **every module imports on CPU without it**.

```bash
# whole pipeline, tiny synthetic path (seconds, ~$0):
# (smoke writes ONLY to cache_smoke/ + outputs_smoke/, never the real dirs)
PYTHON=/usr/bin/python3 bash run_all_real.sh --smoke

# or a single stage at a time (each module honours --smoke):
/usr/bin/python3 extract_activations.py --smoke --config configs/smoke.yaml \
    --dataset clean --cache_dir ./cache_smoke
/usr/bin/python3 train_sae.py   --smoke --config configs/smoke.yaml --cache_dir ./cache_smoke

# plan the real run's cost without touching a GPU:
/usr/bin/python3 cost_estimate.py                 # default ViT-L/14 plan
/usr/bin/python3 cost_estimate.py --smoke         # the tiny plan
```

### Later, on a GPU box — the real run (the publishable numbers)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements_real.txt              # pulls open_clip + datasets
# stage ImageNet-1k + the OOD ladder, point configs/vit_l14.yaml at them, then:
bash run_all_real.sh --real --config configs/vit_l14.yaml
```

The **default** mode of every entry script is the **real** path; `--smoke` is the
opt-in tiny path. Full hardware, time/$ budget, dataset sources+sizes, the SAE
size recipe, and the exact end-to-end command order are in **`RUN_AT_SCALE.md`**.

---

_For research and educational purposes only._
