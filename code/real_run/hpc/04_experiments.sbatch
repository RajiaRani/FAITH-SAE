#!/usr/bin/env bash
# =============================================================================
# 04_experiments.sbatch  --  STAGES 3-7: concept select + manifold + CFS + OOD
#                            sweep + ablations  (GPU).
# FAITH-SAE real-scale pipeline  ·  SLURM HPC launch kit
# Author: Rajia Rani  ()
#
# Runs the science stages IN THE SAME ORDER as run_all_real.sh (stages 3-7),
# all reusing the cached activations + the trained SAE:
#   3) concept_select.py  -- pick the reliable ~10-15% testable concepts
#   4) manifold.py        -- SVD the real-image bank -> on-manifold basis U_r
#   5) cfs_eval.py --smoke-- CFS scorer self-test (it is a LIBRARY; the real
#                            clean-vs-OOD CFS is produced by the OOD sweep)
#   6) ood_sweep.py       -- CFS across in1k->in100->food101->cifar100
#   7) ablations_real.py  -- A1..A5 ablation grid
# Depends on 03_train_sae (needs sae.safetensors); chained via afterok.
#
# #SBATCH lines are DEFAULTS; submit_all.sh overrides them from cluster_env.sh.
#
# For research and educational purposes only.
# =============================================================================
#SBATCH --job-name=fsae_experiments
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
set -euo pipefail

HPC_DIR="${HPC_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# shellcheck source=cluster_env.sh
source "${HPC_DIR}/cluster_env.sh"
activate_env

cd "${REAL_RUN_DIR}"

echo "=============================================================="
echo " [04_experiments]  concept_select -> manifold -> cfs -> ood -> ablations"
echo "   config    : ${CONFIG}"
echo "   cache_dir : ${CACHE_DIR}"
echo "   node      : $(hostname)   gpu(s): ${CUDA_VISIBLE_DEVICES:-?}"
echo "=============================================================="

COMMON=(--config "${CONFIG}" --cache_dir "${CACHE_DIR}")

echo ""; echo "----- [3/7] concept_select ----------------------------------"
# REAL flags: --config --cache_dir
python concept_select.py "${COMMON[@]}"

echo ""; echo "----- [4/7] manifold (U_r basis) ----------------------------"
# REAL flags: --config --cache_dir   -> writes outputs/U_r.npy
python manifold.py "${COMMON[@]}"

echo ""; echo "----- [5/7] cfs_eval self-test ------------------------------"
# cfs_eval.py is a LIBRARY imported by ood_sweep/ablations; its ONLY CLI is the
# offline --smoke self-test (asserts on-manifold beats naive/random). The REAL
# in-distribution CFS is the 'in1k' rung of the OOD sweep below.
python cfs_eval.py --smoke

echo ""; echo "----- [6/7] ood_sweep (RQ1 clean + RQ3 ladder) --------------"
# REAL flags: --config --cache_dir   -> writes outputs/ood_cfs_sweep.csv
python ood_sweep.py "${COMMON[@]}"

echo ""; echo "----- [7/7] ablations_real (A1..A5) -------------------------"
# REAL flags: --config --cache_dir   -> writes outputs/ablations.csv
python ablations_real.py "${COMMON[@]}"

echo ""
echo "[04_experiments] done -> ${OUT_DIR}/{ood_cfs_sweep.csv, ablations.csv, U_r.npy}"
