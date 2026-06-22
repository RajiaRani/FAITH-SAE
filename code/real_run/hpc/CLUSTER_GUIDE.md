# CLUSTER GUIDE — running FAITH-SAE on a university supercomputer

### Author: Rajia Rani · ``

This is the **plain-English, no-prior-HPC-experience** walkthrough for running the
real FAITH-SAE pipeline on a SLURM cluster. If you can edit one text file and type
three commands, you can run this. The expensive parts (a GPU, ImageNet) are
provided **for free** by your university's HPC center.

**This run uses standard supervised `timm` ViTs (ViT-S / ViT-B / ViT-L, ≈
22M / 86M / 304M params) as the frozen backbone, and your datasets — ImageNet-1k,
IN-100, Food-101, CIFAR-100 — as the domain-shift ladder.** It runs across all
three model sizes (the *model-size axis*), so the study has two axes:
**faithfulness × model-size × domain-shift.**

---

## (a) Words you'll see, one line each

- **Cluster / supercomputer** — a big pile of shared computers ("nodes") you log
  into and borrow. You don't own a GPU; you *request* one for a few hours.
- **Login node** — the small machine you land on when you `ssh` in. You edit
  files and *submit* jobs here. **Never run the heavy training here** — it's
  shared; you send work to the compute nodes instead.
- **SLURM** — the "job scheduler." It owns the GPUs and hands them out fairly.
  You ask SLURM for resources; it runs your job when a GPU is free.
- **`sbatch <file>`** — "submit this batch job." Returns a **job ID** and runs
  your script later, on a compute node, when resources free up.
- **`squeue`** — "show my jobs in the queue" (waiting or running).
- **`sacct`** — "show my jobs' history" (finished, failed, how long they took).
- **Scratch** — fast, huge, *temporary* disk for working data. Put activations
  and outputs here. (It can be auto-deleted after weeks — copy keepers to your
  home/project space.)
- **Job array** — one `sbatch` that fans out into N near-identical tasks (we use
  it to extract activations for the 4 datasets in parallel).
- **Dependency (`afterok`)** — "don't start job B until job A finished OK." This
  is how the whole pipeline runs hands-off, in order.

---

## (b) One-time setup (do these three things once)

All commands run **on the login node**, from inside this `hpc/` folder.

1. **Edit `cluster_env.sh`** — the only file you touch. Fill in every line
   marked `<<< EDIT`:
   - `ACCOUNT`, `PARTITION` — your allocation + the GPU queue name (ask your
     admin or run `sinfo`).
   - `IMAGENET_DIR` — the path where ImageNet-1k **already lives** on the
     cluster (we do **not** download it). It backs **both** `in1k` (the
     in-distribution rung) **and** `in100` (a 100-class subset). Ask your admin;
     it's usually under `/datasets/...` or `/scratch/shared/...`.
   - `DATA_DIR`, `CACHE_DIR`, `OUT_DIR` — point these at **scratch**. Food-101
     and CIFAR-100 download (small) into `DATA_DIR`.
   - `BACKBONE_FRAMEWORK` — leave `timm` (the default supervised ViT path).
   - `BACKBONE` — for a one-off single-size run: `vit_s_20m` / `vit_b_84m` /
     `vit_l_307m` (see the sizing rule in (f)). The model-size sweep runs all
     three regardless.
   - `MODEL_SIZES` — the three sizes the sweep runs (leave the default).
   - Optional: `GPU_CONSTRAINT` to pin a GPU type; leave empty to take any.

   > **timm model string:** each config's `backbone.name` is a standard timm
   > string (`vit_small/base/large_patch16_224`). **If your exact model loads
   > differently** (e.g. a specific pretraining tag like
   > `vit_base_patch16_224.augreg_in21k_ft_in1k`), edit `backbone.name` in the
   > config — nothing else changes.

2. **Create the software environment:**
   ```bash
   bash 00_setup_env.sh
   ```
   This loads cluster modules (you may need to fix the `module load` names — see
   troubleshooting), makes a conda/venv environment, installs everything from
   `requirements_real.txt`, and ends by printing **"env ready"** after importing
   `torch`, `timm`, and `torchvision`.

3. **Stage the data:**
   ```bash
   bash 01_stage_data.sh
   ```
   This checks ImageNet is present (it backs both `in1k` and `in100`),
   **pre-downloads the two small torchvision sets** (Food-101 ≈ 5 GB, CIFAR-100
   ≈ 170 MB) into `DATA_DIR`, notes that **IN-100 is derived from IN-1k** via
   `in100_classes.txt`, and patches **all three** model-size configs so every
   path points at your cluster. Nothing here needs a form or a license.

   > **Fill in `in100_classes.txt`** (in `real_run/`) with the 100 ImageNet
   > wnids that define IN-100 — see that file's header for where to get the
   > standard split. If left unfilled, the `in100` rung falls back to the full
   > IN-1k classes so the pipeline still runs.

---

## (c) Launch everything (one command)

**The model-size study (all three ViT sizes — this is the headline run):**

```bash
bash submit_model_sweep.sh
```

That loops the whole pipeline over `vit_s_20m`, `vit_b_84m`, `vit_l_307m` (each
to its own per-size cache/out dir), then a final CPU job stitches every size's
`ood_cfs_sweep.csv` into one combined **CFS-vs-(model_size × dataset)** table at
`${OUT_DIR}_model_sweep/cfs_model_size_sweep.csv`.

**A single size only** (cheaper / a quick wiring check):

```bash
bash submit_all.sh           # runs just the $BACKBONE size
```

Either submits the four stages with the right dependencies and prints the job IDs:

```
02_extract  (array 0-3)  →  03_train_sae  →  04_experiments  →  05_analysis
```

You can then **log out** — SLURM runs it for you. To run a single stage by hand
instead (e.g. to re-run just analysis), submit one file, e.g.:

```bash
source cluster_env.sh
sbatch --account=$ACCOUNT --partition=$PARTITION --export=ALL --chdir="$PWD" 05_analysis.sbatch
```

(`submit_all.sh` is just these four `sbatch` calls wired together with
`--dependency=afterok:`.)

---

## (d) Watch your jobs / know when it's done

```bash
squeue -u $USER          # what's waiting (PD = pending) or running (R)
sacct  --format=JobID,JobName,State,Elapsed,ExitCode   # history + exit codes
```

Each job also writes a live log next to the scripts, named like
`fsae_extract_<jobid>_0.out`, `fsae_train_<jobid>.out`, etc. Tail it:

```bash
tail -f fsae_train_<jobid>.out
```

**"Done" looks like:** every stage shows `State=COMPLETED` (ExitCode `0:0`) in
`sacct`, and the final `05_analysis` log prints **"DONE. Paper-ready artifacts"**.
A `FAILED` state means read that stage's `.out` log — the error is at the bottom.

---

## (e) Where the results land (and what maps to the paper)

Each model size writes to its own `${OUT_DIR}_<size>` (e.g. `..._vit_s_20m`). The
files that feed the paper, per size:

| File in `OUT_DIR_<size>` | What it is | Paper role |
|---|---|---|
| `sae.safetensors` | the trained TopK SAE | the model artifact |
| `ood_cfs_sweep.csv` | CFS across the domain-shift ladder | the RQ3 numbers |
| `ablations.csv` | A1–A5 ablation grid | the ablation table |
| `per_concept_cfs.csv` | per-concept CFS (built in stage 05) | bootstrap input |
| `bootstrap_ci.csv` | confidence intervals | the CIs in the results |
| **`FINDINGS.md`** | auto-written findings + `\pending{}` numbers | **paste into the paper** |
| **`fig1_cfs_ood_sweep.png`** | CFS-vs-shift curve | **Figure 1** |
| **`fig7_by_method_bar.png`** | CFS by steering method | **Figure 7** |

And the **two-axis** product of the sweep:

| File | What it is | Paper role |
|---|---|---|
| **`${OUT_DIR}_model_sweep/cfs_model_size_sweep.csv`** | CFS vs (model_size × dataset × method), all 3 sizes stacked | **the model-size × shift table** |

Copy these off scratch to your project/home space so they aren't purged.

---

## (f) The model-size axis & GPU-sizing rule

The **study runs all three sizes** via `submit_model_sweep.sh` — that *is* the
model-size axis. The three configs (supervised `timm` ViTs):

| Config | `backbone.name` | Params | Width `d_in` | SAE features | GPU |
|---|---|---|---|---|---|
| `configs/vit_s_20m.yaml` | `vit_small_patch16_224` | ≈ 22M | 384 | 12,288 | small (V100 / A40 / RTX) |
| `configs/vit_b_84m.yaml` | `vit_base_patch16_224` | ≈ 86M | 768 | 24,576 | mid |
| `configs/vit_l_307m.yaml` | `vit_large_patch16_224` | ≈ 304M | 1024 | 32,768 | big (A100 / H100, 40–80 GB) |

> **timm model string:** if your exact ViT loads under a different name (e.g. a
> pretraining tag like `vit_base_patch16_224.augreg_in21k_ft_in1k`), edit
> `backbone.name` in the config — nothing else changes.

For a **single one-off run**, set `BACKBONE` to one of the three config stems in
`cluster_env.sh` and use `submit_all.sh`. When unsure, start with `vit_s_20m`: it
finishes fastest and proves the wiring.

If you run out of GPU memory, lower `data.batch_size` (e.g. 512 → 128) and/or
`sae.batch_tokens` (8192 → 4096) in that size's config YAML.

---

## (g) Troubleshooting (the five things that actually go wrong)

| Symptom | Fix |
|---|---|
| **CUDA out of memory** | Run a smaller size: `BACKBONE=vit_s_20m` (or drop `vit_l_307m` from `MODEL_SIZES`); if still OOM, lower `data.batch_size` / `sae.batch_tokens` in that size's config YAML. |
| **`timm` model not found / wrong weights** | Your exact ViT loads under a different timm string. Edit `backbone.name` in the config (e.g. `vit_base_patch16_224.augreg_in21k_ft_in1k`). List candidates with `python -c "import timm; print(timm.list_models('vit_*patch16_224*'))"`. |
| **`module: command not found` or a module name fails** | Module names differ per cluster. Run `module avail` / `module spider cuda` and edit the `module load` lines in `00_setup_env.sh` to match. If your site has no modules, delete those lines. |
| **Job stuck `PD` (pending) forever** | Wrong `ACCOUNT`/`PARTITION`, or you asked for a GPU type that's busy. Check `squeue --start -j <jobid>` for the reason; fix `ACCOUNT`/`PARTITION` or clear `GPU_CONSTRAINT` in `cluster_env.sh`. |
| **`No activation shards` / `path does not exist`** | A data path is wrong (or `in100_classes.txt` is empty). Re-check `IMAGENET_DIR` / `DATA_DIR` / `CACHE_DIR` in `cluster_env.sh`, then re-run `01_stage_data.sh` (it re-patches the configs). |
| **`timm` / `torchvision` import error inside a job** | The job didn't activate the env. Confirm `CONDA_ENV` (or `USE_VENV`/`VENV_DIR`) in `cluster_env.sh` matches what `00_setup_env.sh` created. |

---

## How long / how much

- **Compute:** the bundled estimator (`python cost_estimate.py`) puts a single
  ViT-L plan at **~15 GPU-hours**; the **model-size sweep runs three sizes**, so
  budget roughly **2–3× that** (ViT-S/B are much cheaper than ViT-L). With extra
  seeds and a bigger SAE the honest envelope is **~40–120 GPU-hours** of
  wall-clock across the three sizes' stages.
- **Money:** on a university HPC this is **free** (it's your allocation, not a
  cloud bill). The dollar figures the estimator prints are only for comparison to
  renting a cloud A100/H100.
- **Wall-clock you'll actually wait:** mostly queue time + the longest stage
  (`04_experiments`, the ablations). Submit it, check back the next day.

---

_For research and educational purposes only._
