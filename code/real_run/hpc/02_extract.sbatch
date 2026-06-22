#!/usr/bin/env bash
# =============================================================================
# 02_extract.sbatch  --  STAGE 1: extract frozen-backbone activations (GPU).
# FAITH-SAE real-scale pipeline  ·  SLURM HPC launch kit
# Author: Rajia Rani  ()
#
# SLURM JOB ARRAY over the 4 datasets of the student's ladder (one task each):
#     0=in1k (ImageNet-1k train; in-dist + SAE source)   1=in100 (100-class subset)
#     2=food101 (torchvision)                            3=cifar100 (torchvision)
# Each task forwards the frozen timm ViT over ONE dataset and writes the cached
# activation shards to $CACHE_DIR (data_real does the work). Tasks are
# independent, so SLURM runs them in parallel as GPUs free up.
#
# The #SBATCH lines below are DEFAULTS. submit_all.sh overrides account /
# partition / gres / time / mem / cpus from cluster_env.sh via sbatch CLI flags
# (CLI beats the directives), so you normally launch via submit_all.sh and never
# edit these. To run THIS stage alone:
#     sbatch --account=$ACCOUNT --partition=$PARTITION ... 02_extract.sbatch
#   (or simpler, after sourcing cluster_env.sh, just: bash submit_all.sh)
#
# For research and educational purposes only.
# =============================================================================
#SBATCH --job-name=fsae_extract
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
set -euo pipefail

# Locate cluster_env.sh relative to this submitted script. SLURM copies the
# script, but exports the original path in SLURM_SUBMIT_DIR; we also fall back to
# the env var HPC_DIR that submit_all.sh exports.
HPC_DIR="${HPC_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# shellcheck source=cluster_env.sh
source "${HPC_DIR}/cluster_env.sh"
activate_env

cd "${REAL_RUN_DIR}"

# Map the array index -> dataset name (the exact tokens extract_activations.py
# accepts: in1k | in100 | food101 | cifar100).
DATASETS=(in1k in100 food101 cifar100)
DS="${DATASETS[${SLURM_ARRAY_TASK_ID:-0}]}"

echo "=============================================================="
echo " [02_extract] dataset=${DS}  (array task ${SLURM_ARRAY_TASK_ID:-0})"
echo "   config    : ${CONFIG}"
echo "   cache_dir : ${CACHE_DIR}"
echo "   node      : $(hostname)   gpu(s): ${CUDA_VISIBLE_DEVICES:-?}"
echo "=============================================================="

# REAL flags (verified via --help): --config --dataset --cache_dir
python extract_activations.py \
    --config   "${CONFIG}" \
    --dataset  "${DS}" \
    --cache_dir "${CACHE_DIR}"

echo "[02_extract] done: ${DS} -> shards in ${CACHE_DIR}"
