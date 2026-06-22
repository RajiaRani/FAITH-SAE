#!/usr/bin/env bash
# =============================================================================
# 03_train_sae.sbatch  --  STAGE 2: train the TopK SAE on cached acts (GPU).
# FAITH-SAE real-scale pipeline  ·  SLURM HPC launch kit
# Author: Rajia Rani  ()
#
# Streams the cached IN-DISTRIBUTION (in1k = ImageNet-1k train) activation shards
# through the TopK sparse autoencoder trainer (AdamW + warmup + AMP, dead-feature
# revival) and writes the SAE checkpoint to <out_dir>/sae.safetensors.
# Depends on 02_extract (needs the 'in1k' shards); submit_all.sh chains it with
# --dependency=afterok.
#
# #SBATCH lines are DEFAULTS; submit_all.sh overrides account/partition/gres/
# time/mem/cpus from cluster_env.sh. Launch via submit_all.sh.
#
# For research and educational purposes only.
# =============================================================================
#SBATCH --job-name=fsae_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
set -euo pipefail

HPC_DIR="${HPC_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# shellcheck source=cluster_env.sh
source "${HPC_DIR}/cluster_env.sh"
activate_env

cd "${REAL_RUN_DIR}"

echo "=============================================================="
echo " [03_train_sae]"
echo "   config    : ${CONFIG}"
echo "   cache_dir : ${CACHE_DIR}"
echo "   out (cfg) : ${OUT_DIR}/sae.safetensors"
echo "   node      : $(hostname)   gpu(s): ${CUDA_VISIBLE_DEVICES:-?}"
echo "=============================================================="

# REAL flags (verified via --help): --config --cache_dir [--dataset] [--smoke]
# (the SAE trains on the cached 'in1k' shards by default; out path comes from
#  paths.sae_ckpt in the config, which 01_stage_data.sh pointed at $OUT_DIR.)
python train_sae.py \
    --config   "${CONFIG}" \
    --cache_dir "${CACHE_DIR}"

echo "[03_train_sae] done -> ${OUT_DIR}/sae.safetensors"
