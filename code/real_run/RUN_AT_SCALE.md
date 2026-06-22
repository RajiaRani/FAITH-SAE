# RUN AT SCALE — FAITH-SAE real run

### Author: Rajia Rani · ``

The operational guide for executing the **real, publishable** FAITH-SAE pipeline
on a GPU box: exact **hardware**, per-stage **GPU-hour + dollar** budget,
**storage** planning, **dataset** notes, the **SAE size recipe**, and the **full
ordered command sequence** end to end.

> **What this run is.** The backbone is a **standard supervised `timm` ViT**
> (`backbone.framework: timm`, the default), run across **three sizes** — ViT-S /
> ViT-B / ViT-L (`vit_small/base/large_patch16_224`, ≈ 22M / 86M / 304M params).
> The domain-shift ladder is the student's own datasets, ordered by shift
> strength: **`in1k` → `in100` → `food101` → `cifar100`**. So the study has **two
> axes** — *faithfulness × model-size × domain-shift* (see §0). The original
> open_clip CLIP path still works (`backbone.framework: open_clip`).

All time/cost figures are reproducible with the bundled estimator —
`python cost_estimate.py` — whose constants are documented inline. Treat them as
**±2×** planning numbers, not guarantees: real wall-clock depends on your
dataloader, disk, and exact GPU.

---

## 0. The two-axis study (faithfulness × model-size × domain-shift)

This run measures CFS across **two** axes at once:

- **Model-size axis** — three supervised `timm` ViTs, each its own config:

  | Config | `backbone.name` | Params | `d_in` | SAE features | layer |
  |---|---|---|---|---|---|
  | `configs/vit_s_20m.yaml` | `vit_small_patch16_224` | ≈ 22M | 384 | 12,288 | 8 |
  | `configs/vit_b_84m.yaml` | `vit_base_patch16_224` | ≈ 86M | 768 | 24,576 | 9 |
  | `configs/vit_l_307m.yaml` | `vit_large_patch16_224` | ≈ 304M | 1024 | 32,768 | 18 |

  > Swap `backbone.name` to **your exact model string** if these don't match how
  > you load your ViT (e.g. `vit_base_patch16_224.augreg_in21k_ft_in1k`).

- **Domain-shift axis** — the student's ladder, ordered by shift strength:
  `in1k` (in-distribution, **also the SAE-training source**) → `in100` (mild;
  a 100-class IN-1k subset) → `food101` (domain shift) → `cifar100` (strong
  domain + **resolution** shift: 32×32 upsampled to 224).

**Run the whole 2-axis sweep with one command:**

```bash
bash run_model_size_sweep.sh --real          # all 3 sizes x the ladder
# offline wiring check (CPU, no timm/downloads):
bash run_model_size_sweep.sh --smoke
```

It runs the full pipeline for each size and stacks every per-rung CFS into one
combined CSV indexed by **(model_size × dataset × steering method)**:

```
outputs_model_sweep/cfs_model_size_sweep.csv
   columns: model_size, backbone, d_in, rung/dataset, method, cfs (+ components)
```

That single table is the model-size × domain-shift result the paper plots. On a
SLURM cluster the equivalent is `hpc/submit_model_sweep.sh` (see `hpc/CLUSTER_GUIDE.md`).

---

## 1. Hardware

| Component | Recommended | Minimum | Notes |
|---|---|---|---|
| GPU | **1× NVIDIA A100 80GB** or **1× H100 80GB** | 1× A100/A6000 48GB | 80GB lets the 32768-wide SAE + a large activation chunk sit in VRAM; H100 ≈ 1.5–1.7× faster |
| CPU | 16+ cores | 8 cores | JPEG decode + the activation dataloader are CPU-bound during extraction |
| RAM | 128 GB | 64 GB | streaming keeps the working set small, but the manifold/eval bank (≈2M×1024 fp32 ≈ 8 GB) lives in RAM |
| Disk | **2–4 TB NVMe** | 1 TB | activation shards are large; see §3 (stream or subset) |
| Network | fast (multi-GB datasets) | — | one-time dataset pulls (≈250 GB raw images) |

A single 80GB GPU is sufficient — **this study does not need multi-GPU**. One
A100/H100 is the design target.

---

## 2. Per-stage GPU-hour + dollar budget

Run `python cost_estimate.py` for your exact knobs. The default plan
(**CLIP ViT-L/14, 300M-token budget, SAE 4 passes, full OOD ladder, A1–A5**):

```
stage                               GPU-hours  cost @ $1.80/h
-------------------------------------------------------------
1_extract_activations                   3.4 h           $6.14
2_train_sae                            57 min           $1.71
3_manifold_svd                          0 min           $0.00
4_concept_select                        0 min           $0.01
5_cfs_clean                            15 min           $0.45
6_ood_sweep                             2.2 h           $4.02
7_ablations                             8.5 h          $15.29
-------------------------------------------------------------
TOTAL (compute model)                  15.3 h          $27.61
TOTAL (honest band x1.0-1.8)    15.3 h-27.6 h          $28-50
```

Pushing the SAE to a **1B-token** budget (full ImageNet-grade dictionary):

```
TOTAL (compute model)                  23.8 h          $42.77
TOTAL (honest band x1.0-1.8)    23.8 h-42.8 h          $43-77
```

**Headline envelope (what to budget for the paper):** with the full ImageNet
extraction, the 1B-token SAE, the complete A1–A5 grid, **and a few random seeds /
the wider OOD-C severity counts**, the end-to-end run sits in the documented
band of **~40–120 GPU-hours ≈ $80–250** on mid-market A100/H100 cloud. The
estimator reproduces the **lower** portion of that envelope with the default
knobs; the upper portion is the multi-seed / full-severity / large-SAE corner.
Cheapest viable run (ViT-B/16, 300M tokens, single seed) is ≈ $15–30.

The `×1.0–1.8` band is the **honest overhead multiplier**: the pure-compute model
assumes a saturated GPU, but extraction is I/O-bound (JPEG decode + fp16 writes)
so real wall-clock runs 1–1.8× longer.

---

## 3. Storage planning (the real gotcha)

Activations are the dominant storage cost.

> **One ImageNet-1k train epoch of ViT-L patch activations ≈ 0.5 TB.**
> 1.28M images × 196 patch tokens (patch16@224) × 1024 dims × **2 bytes (fp16)**
> ≈ 1.28e6 × 196 × 1024 × 2 ≈ **0.51 TB** (ViT-S/B are ⅜/¾ of this at d=384/768).

Three honest strategies (the pipeline supports all three; pick via the config):

1. **Stream, never fully materialize (recommended).** Use `webdataset` /
   `iter_image_batches` to push images through the frozen backbone and feed the
   SAE trainer **on the fly**, writing only a **rolling shard window** to
   `cache/`. Peak disk ≈ a few tens of GB. This is the default for the 1B-token
   budget — you re-stream rather than re-read 600 GB.
2. **Subset to the token budget.** The default 300M-token budget needs only
   ≈ 1.17M image-forwards' worth of tokens; cached fp16 that is ≈ **150 GB** on
   disk (fits a 1 TB NVMe with room for the OOD ladder).
3. **The shift ladder is cheap to cache fully.** in100 (≈ 130k imgs) + food101
   (≈ 25k test) + cifar100 (10k test) ≈ a few tens of GB at d=1024 — cache these
   once and reuse across CFS, OOD sweep, and ablations. (Multiply by 3 across the
   model sizes; ViT-S/B are smaller per token.)

`cache*/` and `outputs*/` are **git-ignored** (only `.gitkeep` is tracked) —
**never commit activation shards.**

---

## 4. Datasets — the student's domain-shift ladder

The ladder is the **student's own datasets**, ordered by shift strength. Only
ImageNet-1k must be staged manually; Food-101 and CIFAR-100 are downloaded by
torchvision itself, and IN-100 is derived from IN-1k.

| Rung | Role | Size (approx) | Source |
|---|---|---|---|
| **`in1k`** ImageNet-1k train | in-distribution **+ SAE training source** | **1.28M imgs, ≈ 150 GB** | already on the cluster (image-net.org / HF `imagenet-1k`) |
| **`in100`** 100-class IN-1k subset | mild shift | filtered from `in1k` (no extra disk) | `in100_classes.txt` (100 wnids) |
| **`food101`** | domain shift | **101k imgs, ≈ 5 GB** | `torchvision.datasets.Food101(download=True)` |
| **`cifar100`** | **strong** domain + resolution shift (32×32 → 224) | **60k imgs, ≈ 170 MB** | `torchvision.datasets.CIFAR100(download=True)` |

Config keys (one per model-size config, e.g. `configs/vit_b_84m.yaml`):

```yaml
data:
  imagenet_train_dir: /path/already/on/cluster/imagenet/train   # backs in1k AND in100
  in100_classes_file: ./in100_classes.txt   # the 100 wnids that define IN-100
  data_dir: ./data                           # torchvision downloads Food101/CIFAR100 here
  batch_size: 384
  num_workers: 8
  max_images: null          # set an int to subset for a faster partial run

ood:
  levels: [in1k, in100, food101, cifar100]   # ordered by shift strength
```

> **IN-100:** fill `in100_classes.txt` (in `real_run/`) with the 100 ImageNet
> wnids of your chosen split (see that file's header). If left unfilled, the
> `in100` rung falls back to the full IN-1k class set so the pipeline still runs.

---

## 5. SAE size recipe

The SAE is the scientific core; size it to balance dictionary richness against
compute. Knobs live under `sae:` in each size config (e.g. `configs/vit_l_307m.yaml`).

| Knob | Default (ViT-L) | Range | What it controls |
|---|---|---|---|
| `d_in` | **1024** | 384 / 768 / 1024 | = backbone width (ViT-S/B/L) |
| `expansion` | **32** → `n_features = 32768` | 8–64 | dictionary overcompleteness; bigger = more monosemantic features, more compute |
| `k` (TopK) | **32** | 32–64 | active features/token (sparsity); A2 ablation |
| `token_budget` | **300M** | 300M–1B | total patch-token updates the SAE sees |
| `batch_tokens` | **8192** | 4k–16k | tokens per optimizer step |
| `lr` / `warmup` | **4e-4 / 1000** | — | AdamW + linear warmup |
| `aux_k` | **256** | 128–512 | AuxK auxiliary loss width (revives dead features) |
| `dead_window` | **10M** | — | a feature unused for this many tokens = "dead" |
| `normalize` | `unit_meansquare` | — | activation normalization before the SAE |

**Recipe choices.** Start at **expansion 32 (n=32768), k=32, 300M tokens** — the
sweet spot that fits 80GB and trains in ≈ 1 GPU-hour at 4 passes. Go to
**expansion 64 / 1B tokens** only for the final headline dictionary. Keep
`d_in`, `expansion`, and `k` **identical across all steering variants** so RQ1 is
a fair matched-strength comparison; vary them only inside the A1/A2 ablations.

---

## 6. Full ordered command sequence (end to end)

The **model-size sweep** runs the whole thing for all three sizes (§0):

```bash
# 0) environment (GPU box)
python -m venv .venv && . .venv/bin/activate
pip install -r requirements_real.txt

# 1) fill in in100_classes.txt + point each config's data.* at your datasets, then:
bash run_model_size_sweep.sh --real    # -> outputs_model_sweep/cfs_model_size_sweep.csv
```

To run **one size** end to end, the one-shot driver runs all eight stages in
dependency order for a single config:

```bash
bash run_all_real.sh --real --config configs/vit_b_84m.yaml
```

Or run the stages by hand (this is exactly what the driver does, in order). The
heavy stages take `--config` and `--cache_dir`; the **analysis/figure** stages
take `--results-dir`/`--out-dir` because they consume the result CSVs, not the
config. The real path is the default; `--smoke` swaps in the tiny CPU path.

```bash
CFG=configs/vit_b_84m.yaml ; CACHE=./cache_vit_b ; OUT=./outputs_vit_b

# 1. EXTRACT activations -- run ONCE PER DATASET (in1k + each shift rung).
#    extract_activations.py forwards the frozen timm ViT over one dataset and
#    writes cache/acts_{ds}_*.npy + labels + manifest. (data_real does the work.)
for DS in in1k in100 food101 cifar100 ; do
  python extract_activations.py --config $CFG --dataset $DS --cache_dir $CACHE
done

# 2. TRAIN the TopK SAE by streaming the cached in1k (in-distribution) shards.
python train_sae.py      --config $CFG --cache_dir $CACHE          # -> outputs/sae.safetensors

# 3. SELECT the reliable ~10-15% testable concepts.
python concept_select.py --config $CFG --cache_dir $CACHE

# 4. ESTIMATE the on-manifold projection basis U_r via SVD.
python manifold.py       --config $CFG --cache_dir $CACHE          # -> outputs/U_r.npy

# 5. (optional) CFS scorer self-test. cfs_eval.py is a LIBRARY imported by the
#    sweeps; the real in-distribution CFS (RQ1) is the 'in1k' rung of step 6.
python cfs_eval.py --smoke                                          # sanity only

# 6. OOD SWEEP across the ladder (RQ1 in1k rung + RQ3 in1k->in100->food101->cifar100).
python ood_sweep.py      --config $CFG --cache_dir $CACHE          # -> outputs/ood_cfs_sweep.csv

# 7. ABLATIONS A1-A5 (SAE type, k, proj-rank r, selection thresh, layer/token).
python ablations_real.py --config $CFG --cache_dir $CACHE          # -> outputs/ablations.csv

# 8. ANALYSIS (bootstrap CIs + findings) then FIGURES (fig1, fig7, ...).
#    These read the CSVs from $OUT, so they take --results-dir / --out-dir.
python analysis_real.py  --results-dir $OUT --out-dir $OUT         # -> bootstrap_ci.csv, FINDINGS.md
python figures_real.py   --results-dir $OUT --out-dir $OUT         # -> outputs/fig1_*.png, fig7_*.png
```

> **Per-concept CSV caveat (honest gap).** `analysis_real.py` / `figures_real.py`
> read a `per_concept_cfs.csv` for the per-concept bootstrap (`fig4` reliability
> tail). The OOD sweep currently embeds the per-concept CFS inside
> `ood_cfs_sweep.csv` (column `cfs_per_concept`); exploding that one rung into a
> standalone `per_concept_cfs.csv` is the one small glue step between step 6 and
> step 8 on the real path (the `--smoke` analysis path fabricates it directly).

**Smoke first.** Before any of the above, prove the wiring on CPU (no GPU, no
`open_clip`, no downloads):

```bash
PYTHON=/usr/bin/python3 bash run_all_real.sh --smoke
```

**Resumability.** Stages 2–8 read the artifacts of earlier stages from
`cache/`/`outputs/`, so a crash mid-run can be resumed by re-invoking from the
failed stage — the activation cache (the expensive part) is already on disk.

---

## 7. Sanity checklist before you spend money

- [ ] `python cost_estimate.py --backbone <yours>` prints a plan you accept.
- [ ] `bash run_model_size_sweep.sh --smoke` completes clean (the 2-axis wiring,
      all 3 widths, on CPU).
- [ ] `backbone.name` in each config matches **your exact timm model string**.
- [ ] `in100_classes.txt` filled with the 100 IN-100 wnids (or accept the IN-1k
      fallback for the `in100` rung).
- [ ] Datasets resolve: `imagenet_train_dir` exists; Food-101 + CIFAR-100 download
      to `data_dir`.
- [ ] Disk has room for your storage strategy (§3), **×3 across the model sizes**.
- [ ] `backbone.device: cuda`, AMP on, backbone frozen + `eval()` (no grad).

---

_For research and educational purposes only._
