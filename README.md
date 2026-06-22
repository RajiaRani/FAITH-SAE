# FAITH-SAE: Are Sparse-Autoencoder Concept Directions in Vision Models Causally Faithful Under Distribution Shift?
### Starter code — Rajia Rani · Vision Interpretability

Auto-generated scaffold matching **your** 8-week roadmap (see `ROADMAP.pdf`).
Runs on **day one** — no GPU or model download needed for the smoke path.

## Project snapshot
| | |
|---|---|
| **Building blocks** | Steering methods over a frozen CLIP ViT-B/16 + TopK SAE: `naive_steer` (off-manifold activation-addition), `random_steer`, `clamp_steer`, `onmanifold_steer` (project the edit onto the top-`r` real-image subspace, ours) |
| **Baselines** | Supervised TCAV-style concept-direction steering (quality reference), naive off-manifold activation-addition (main competitor), random-direction (null), raw-clamp — all at **matched steering strength** |
| **Benchmarks** | Clean ImageNet-val → ImageNet-R → ImageNet-Sketch → ImageNet-C (severity 1–5) → ObjectNet |
| **Metrics** | Monotonicity, specificity, sufficiency → **Causal Faithfulness Score (CFS) ∈ [0,1]**; off-manifold residual; ΔCFS-per-shift degradation slope with bootstrap CIs over concepts |
| **Big idea** | A steering *method* alone is unfalsifiable; a faithfulness *metric* on naive edits just measures artifacts. Combine them: measure **CFS of on-manifold steering across the OOD ladder** — each cures the other's blind spot. |
| **Target** | On-manifold steering achieves higher CFS than naive/random/clamp at matched strength on clean images, and degrades more gracefully under shift — locating the collapse knee. |

## Your research questions
- **RQ1.** Does on-manifold projected steering achieve a higher Causal Faithfulness Score (CFS) than naive off-manifold activation-addition steering — and than random-direction, raw-clamp, and supervised concept-direction (TCAV-style) steering — at **matched steering strength** on clean, in-distribution images?
- **RQ2.** How does CFS **decompose** into monotonicity / specificity / sufficiency, what fraction of discovered concepts steer reliably (the field's "only ~10–15%" claim), and how does the faithfulness optimum depend on the two design knobs — **steering strength** and **manifold-projection rank `r`**?
- **RQ3.** Does CFS **survive distribution shift** across clean ImageNet → ImageNet-R → ImageNet-Sketch → ImageNet-C (corruption-severity dial) → ObjectNet, and **where is the collapse knee** — i.e. what is the ΔCFS-per-shift-level degradation slope, and does on-manifold steering degrade more gracefully than naive?

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

python -m src.run_experiments --smoke    # offline sanity run -> results/metrics_all.csv
python -m pytest -q                       # day-one tests
```
> The smoke path uses a tiny **synthetic SAE over synthetic activations with a planted
> concept** so everything runs anywhere: it fits a toy TopK SAE, recovers the planted
> feature, steers it with each variant, and checks `onmanifold_steer` yields a higher
> synthetic-CFS than `naive`/`random`. Swap in the real CLIP ViT-B/16 + ImageNet shifts
> (uncomment `requirements.txt`) as you hit Milestone 2.

## 8-week plan (from your roadmap)
- M1 (weeks 1 to 2): Literature Review & Block Design
- M2 (weeks 3 to 4): Data Pipeline & Implementation
- M3 (weeks 5 to 6): Training & Ablation Experiments
- M4 (weeks 7 to 8): Analysis, Insights & Manuscript

## What's in here
```
.gitignore
README.md
ROADMAP.pdf
requirements.txt
configs/
  default.yaml
data/
  data_cards.md
notebooks/
  01_EDA.ipynb
results/
  .gitkeep
src/
  __init__.py
  data.py
  evaluate.py
  model.py
  run_experiments.py
  train.py
  utils.py
  blocks/
    __init__.py
tests/
  test_smoke.py
```

The research-specific modules under `src/` are the files named in your roadmap.
They ship documented stubs + a runnable offline demo and clear `TODO(M2/M3)`
markers for the parts you implement. Full details (12 papers, datasets, ablations,
deliverables, acceptance checks) are in `ROADMAP.pdf`.

_For research and educational purposes only._
