#!/usr/bin/env bash
# =============================================================================
# cluster_env.sh  --  THE ONE FILE YOU EDIT.
# FAITH-SAE real-scale pipeline  ·  SLURM HPC launch kit
# Author: Rajia Rani  ()
#
# Every other script in this folder `source`s THIS file, so all cluster-specific
# values live in one place. Edit the placeholders below ONCE, then never touch
# the .sbatch files. Anything marked  <<< EDIT  must be set for your cluster.
#
# How to find the values: ask your cluster's docs / `sacctmgr show assoc` (for
# your account+partition), `sinfo` (for partition + GPU constraints), and your
# admin for the scratch path. Nothing here is secret.
#
# For research and educational purposes only.
# =============================================================================

# ---------------------------------------------------------------------------
# 1. SLURM ACCOUNTING  --  who pays / which queue.  (sacctmgr / sinfo)
# ---------------------------------------------------------------------------
# ACCOUNT     : your allocation/project code charged for the GPU-hours.
# PARTITION   : the GPU queue name on your cluster (e.g. gpu, gpuq, a100, dgx).
export ACCOUNT="my_account"          # <<< EDIT  (e.g. cs_dept_gpu)
export PARTITION="gpu"               # <<< EDIT  (e.g. gpu, a100, gpuq)

# ---------------------------------------------------------------------------
# 2. GPU REQUEST  --  do NOT hardcode a GPU type; this is your knob.
# ---------------------------------------------------------------------------
# GPU_GRES       : the generic-resource request. "gpu:1" = one GPU of any type.
#                  To pin a type by gres name use e.g. "gpu:a100:1".
# GPU_CONSTRAINT : OPTIONAL feature constraint (-C/--constraint) to force a node
#                  with a specific GPU. Leave EMPTY to take whatever is free
#                  (recommended on a mixed cluster). Examples: "a100", "v100",
#                  "h100", "a40". Check `sinfo -o "%P %f %G"` for valid names.
export GPU_GRES="gpu:1"              # 1 GPU of any type (safe default)
export GPU_CONSTRAINT=""             # <<< OPTIONAL  (e.g. a100); empty = any GPU

# ---------------------------------------------------------------------------
# 3. BACKBONE  --  the GPU-sizing knob (see the sizing rule at the bottom).
# ---------------------------------------------------------------------------
# BACKBONE_FRAMEWORK : which backbone family the jobs load.
#   timm      (DEFAULT) -- standard SUPERVISED timm ViTs (ViT-S/B/L). This is the
#                          student's setup. Each config has backbone.framework=timm.
#   open_clip           -- the original CLIP path (set a vit_l14/vit_b16 config).
export BACKBONE_FRAMEWORK="timm"     # timm (default) | open_clip
#
# BACKBONE picks which SINGLE config a one-off run uses (the model-size SWEEP
# below ignores this and runs all three). Maps to a config in real_run/configs/:
#   vit_l_307m -> configs/vit_l_307m.yaml  (ViT-L ~304M, d=1024). BIG GPU (A100/H100).
#   vit_b_84m  -> configs/vit_b_84m.yaml   (ViT-B ~86M,  d=768).  mid GPU.
#   vit_s_20m  -> configs/vit_s_20m.yaml   (ViT-S ~22M,  d=384).  small GPU.
# (respect a pre-set BACKBONE so submit_model_sweep.sh can override it per size.)
export BACKBONE="${BACKBONE:-vit_b_84m}"   # vit_s_20m | vit_b_84m | vit_l_307m
#
# MODEL_SIZES : the model-size AXIS the sweep runs (the second axis of the study,
#               faithfulness x model-size x domain-shift). submit_model_sweep.sh
#               loops the whole pipeline over each of these config stems.
export MODEL_SIZES="vit_s_20m vit_b_84m vit_l_307m"   # the 3 sizes (S/B/L)

# ---------------------------------------------------------------------------
# 4. DATA + STORAGE PATHS  --  point these at SCRATCH (fast, large, purgeable).
# ---------------------------------------------------------------------------
# IMAGENET_DIR : ImageNet-1k train, ALREADY on the cluster's shared storage.
#                We do NOT download it. Set this to the existing path. It must
#                be ImageFolder layout: <dir>/n01440764/*.JPEG (one subdir/class).
#                Used for BOTH in1k (in-distribution) AND in100 (a 100-class
#                subset filtered via real_run/in100_classes.txt).
#                Ask your admin; common spots: /datasets/imagenet/train,
#                /scratch/shared/imagenet/train, /data/ImageNet/train.
export IMAGENET_DIR="/path/already/on/cluster/imagenet/train"   # <<< EDIT

# DATA_DIR  : where torchvision DOWNLOADS the small datasets Food-101 (~5 GB) and
#             CIFAR-100 (~170 MB). These are tiny; torchvision fetches them itself
#             (no manual download). Put on scratch.
# CACHE_DIR : where extracted activation shards go (the heavy artifacts; can be
#             100s of GB, x3 across the model sizes). MUST be on fast scratch.
# OUT_DIR   : where SAE checkpoint, CSVs, FINDINGS.md and figures land.
# (CACHE_DIR / OUT_DIR respect a pre-set value so submit_model_sweep.sh can give
#  each model size its own per-size cache/out dir; default to the base path.)
export DATA_DIR="${DATA_DIR:-/scratch/$USER/faithsae/data}"      # <<< EDIT to scratch
export CACHE_DIR="${CACHE_DIR:-/scratch/$USER/faithsae/cache}"   # <<< EDIT to scratch
export OUT_DIR="${OUT_DIR:-/scratch/$USER/faithsae/outputs}"     # <<< EDIT to scratch

# ---------------------------------------------------------------------------
# 5. PYTHON ENVIRONMENT  --  conda env name (00_setup_env.sh creates it).
# ---------------------------------------------------------------------------
# CONDA_ENV : name of the conda/venv environment that has requirements_real.txt
#             installed (torch, open_clip, ...). If your site uses a plain venv
#             instead of conda, set USE_VENV=1 and VENV_DIR; 00_setup_env.sh and
#             the .sbatch files honor both.
export CONDA_ENV="faithsae"
export USE_VENV=0                    # 0 = conda env named $CONDA_ENV ; 1 = venv
export VENV_DIR="/scratch/$USER/faithsae/.venv"     # used only if USE_VENV=1

# ---------------------------------------------------------------------------
# 6. WALLTIMES + RESOURCES  --  per-stage SLURM limits. Bump if a job times out.
# ---------------------------------------------------------------------------
# TIME_* are SLURM walltimes (D-HH:MM:SS or HH:MM:SS). Defaults are generous for
# ViT-L on one A100; shrink them for ViT-B or a faster card to queue sooner.
export TIME_EXTRACT="08:00:00"       # 02_extract.sbatch  (per array task)
export TIME_TRAIN="12:00:00"         # 03_train_sae.sbatch
export TIME_EXPERIMENTS="16:00:00"   # 04_experiments.sbatch (ood+ablations heavy)
export TIME_ANALYSIS="02:00:00"      # 05_analysis.sbatch (CPU only)

# Per-job CPU + RAM. Extraction is JPEG-decode / dataloader bound -> give CPUs.
export CPUS="8"                      # --cpus-per-task (matches data.num_workers)
export MEM="64G"                     # --mem  (RAM; manifold/eval bank is RAM-hungry)

# ---------------------------------------------------------------------------
# (do not edit below)  resolve THIS repo's real_run/ dir so jobs find the code.
# REAL_RUN_DIR = the parent of this hpc/ folder (i.e. .../code/real_run).
# ---------------------------------------------------------------------------
_CLUSTER_ENV_SELF="${BASH_SOURCE[0]:-$0}"
export HPC_DIR="$(cd "$(dirname "${_CLUSTER_ENV_SELF}")" && pwd)"
export REAL_RUN_DIR="$(cd "${HPC_DIR}/.." && pwd)"

# Map BACKBONE -> the matching real config shipped in real_run/configs/.
# (Used by the SINGLE-config jobs; the model-size sweep loops $MODEL_SIZES.)
case "${BACKBONE}" in
  vit_s_20m)  export CONFIG="${REAL_RUN_DIR}/configs/vit_s_20m.yaml" ;;
  vit_b_84m)  export CONFIG="${REAL_RUN_DIR}/configs/vit_b_84m.yaml" ;;
  vit_l_307m) export CONFIG="${REAL_RUN_DIR}/configs/vit_l_307m.yaml" ;;
  *) echo "[cluster_env] ERROR: unknown BACKBONE='${BACKBONE}' (use vit_s_20m | vit_b_84m | vit_l_307m)" >&2 ;;
esac

# Helper: map a model-size stem -> its config path (used by submit_model_sweep.sh).
config_for_size () {  # config_for_size <stem>  ->  echoes the config path
  echo "${REAL_RUN_DIR}/configs/$1.yaml"
}

# Build the optional -C/--constraint flag only when GPU_CONSTRAINT is set.
export SBATCH_CONSTRAINT_FLAG=""
[[ -n "${GPU_CONSTRAINT}" ]] && export SBATCH_CONSTRAINT_FLAG="--constraint=${GPU_CONSTRAINT}"

# Helper every .sbatch sources to activate the python env (conda OR venv).
activate_env () {
  if [[ "${USE_VENV}" -eq 1 ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
  else
    # `conda activate` needs the shell hook; fall back to `source activate`.
    if command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)" 2>/dev/null || true
      conda activate "${CONDA_ENV}"
    else
      echo "[cluster_env] WARNING: conda not on PATH; did you `module load anaconda`?" >&2
    fi
  fi
}

# =============================================================================
# GPU-SIZING RULE (read once):
#   The STUDY itself runs all three sizes (the model-size axis) via
#   submit_model_sweep.sh. For a SINGLE one-off run pick BACKBONE by your GPU:
#   BIG GPU  (A100 / H100, 40-80 GB) -> BACKBONE=vit_l_307m (ViT-L ~304M, d=1024,
#       batch 256, SAE 32768 features). The headline, biggest backbone.
#   MID GPU  -> BACKBONE=vit_b_84m (ViT-B ~86M, d=768, SAE 24576 features).
#   SMALL GPU (V100 / A40 / RTX, 16-48 GB) -> BACKBONE=vit_s_20m (ViT-S ~22M,
#       d=384, SAE 12288 features). Lightest; finishes fastest.
#       If you STILL hit CUDA out-of-memory, lower data.batch_size in the YAML
#       (e.g. 512 -> 128) and/or sae.batch_tokens (8192 -> 4096).
#   When in doubt, start with vit_s_20m: it proves the wiring fastest; then run
#   the full vit_s/vit_b/vit_l sweep for the model-size axis.
#
# TIMM MODEL STRING (read once):
#   Each config's backbone.name is a standard timm string
#   (vit_small/base/large_patch16_224). If YOUR exact model loads differently
#   (e.g. a specific pretraining tag like *.augreg_in21k_ft_in1k), edit
#   backbone.name in the config -- nothing else changes.
# =============================================================================
