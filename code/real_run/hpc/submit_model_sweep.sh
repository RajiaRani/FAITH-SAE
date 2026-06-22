#!/usr/bin/env bash
# =============================================================================
# submit_model_sweep.sh  --  the MODEL-SIZE AXIS on SLURM (ViT-S / ViT-B / ViT-L).
# FAITH-SAE real-scale pipeline  ·  SLURM HPC launch kit
# Author: Rajia Rani  ()
#
# Loops the WHOLE pipeline over every model size in $MODEL_SIZES so all three
# backbones run -- this is the second axis of the study (faithfulness x
# MODEL-SIZE x domain-shift). For EACH size it submits the same four dependency-
# chained stages submit_all.sh uses:
#     02_extract (array, 4 datasets) -> 03_train_sae -> 04_experiments -> 05_analysis
# Each size runs to its OWN per-size cache/out dir (CACHE_DIR_<size> /
# OUT_DIR_<size>), passed to the jobs via --export so the three runs never
# collide. After ALL sizes finish, a final CPU job stitches each size's
# ood_cfs_sweep.csv into ONE combined CFS-vs-(model_size x dataset) CSV.
#
# Every cluster-specific value comes from cluster_env.sh. The per-size CONFIG /
# CACHE_DIR / OUT_DIR are exported per loop iteration; cluster_env.sh respects a
# pre-set BACKBONE / CACHE_DIR / OUT_DIR, so each job re-derives the right config.
#
# Run on the LOGIN node, after 00_setup_env.sh + 01_stage_data.sh:
#     bash submit_model_sweep.sh
#
# For research and educational purposes only.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"

command -v sbatch >/dev/null 2>&1 || {
  echo "ERROR: 'sbatch' not found -- are you on a SLURM login node?" >&2
  exit 1
}

echo "=============================================================="
echo " FAITH-SAE  --  submit_model_sweep  (the model-size axis)"
echo "   account    : ${ACCOUNT}      partition : ${PARTITION}"
echo "   framework  : ${BACKBONE_FRAMEWORK}"
echo "   sizes      : ${MODEL_SIZES}"
echo "   base cache : ${CACHE_DIR}    base out  : ${OUT_DIR}"
echo "=============================================================="

export HPC_DIR="${HERE}"

# Common GPU sbatch flags (account/partition/gres/cpus/mem/constraint).
GPU_COMMON=(
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --gres="${GPU_GRES}"
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
  --chdir="${HERE}"
)
[[ -n "${GPU_CONSTRAINT}" ]] && GPU_COMMON+=(--constraint="${GPU_CONSTRAINT}")

# Collect the per-size final-analysis job ids so the combine job waits on them all.
ALL_ANALYSIS_JIDS=()

for SIZE in ${MODEL_SIZES}; do
  CFG="$(config_for_size "${SIZE}")"
  SIZE_CACHE="${CACHE_DIR}_${SIZE}"
  SIZE_OUT="${OUT_DIR}_${SIZE}"
  mkdir -p "${SIZE_CACHE}" "${SIZE_OUT}"

  echo ""
  echo "----- size ${SIZE}  (config ${CFG}) -----"
  echo "   cache : ${SIZE_CACHE}"
  echo "   out   : ${SIZE_OUT}"

  # Per-size environment carried into every stage job (cluster_env.sh respects
  # these pre-set values; BACKBONE picks the matching config).
  SIZE_EXPORT="ALL,BACKBONE=${SIZE},CACHE_DIR=${SIZE_CACHE},OUT_DIR=${SIZE_OUT},HPC_DIR=${HERE}"

  # Stage 02: EXTRACT (array over the 4 datasets in1k/in100/food101/cifar100).
  JID_EXTRACT=$(sbatch --parsable \
    "${GPU_COMMON[@]}" --export="${SIZE_EXPORT}" \
    --time="${TIME_EXTRACT}" --array=0-3 \
    "${HERE}/02_extract.sbatch")
  echo "   02_extract     : ${JID_EXTRACT}  (array 0-3)"

  # Stage 03: TRAIN SAE (after all extract array tasks succeed).
  JID_TRAIN=$(sbatch --parsable \
    "${GPU_COMMON[@]}" --export="${SIZE_EXPORT}" \
    --time="${TIME_TRAIN}" \
    --dependency="afterok:${JID_EXTRACT}" \
    "${HERE}/03_train_sae.sbatch")
  echo "   03_train_sae   : ${JID_TRAIN}  (afterok:${JID_EXTRACT})"

  # Stage 04: EXPERIMENTS (ood sweep + ablations).
  JID_EXPT=$(sbatch --parsable \
    "${GPU_COMMON[@]}" --export="${SIZE_EXPORT}" \
    --time="${TIME_EXPERIMENTS}" \
    --dependency="afterok:${JID_TRAIN}" \
    "${HERE}/04_experiments.sbatch")
  echo "   04_experiments : ${JID_EXPT}  (afterok:${JID_TRAIN})"

  # Stage 05: ANALYSIS (CPU; no --gres).
  JID_ANALYSIS=$(sbatch --parsable \
    --account="${ACCOUNT}" --partition="${PARTITION}" \
    --cpus-per-task=4 --mem=32G --time="${TIME_ANALYSIS}" \
    --export="${SIZE_EXPORT}" --chdir="${HERE}" \
    --dependency="afterok:${JID_EXPT}" \
    "${HERE}/05_analysis.sbatch")
  echo "   05_analysis    : ${JID_ANALYSIS}  (afterok:${JID_EXPT})"
  ALL_ANALYSIS_JIDS+=("${JID_ANALYSIS}")
done

# --- Combine: after EVERY size finishes, stitch the per-size ood_cfs_sweep.csv
#     into one CFS-vs-(model_size x dataset) CSV. Pure CPU; tiny.
DEP="$(IFS=:; echo "${ALL_ANALYSIS_JIDS[*]}")"
JID_COMBINE=$(sbatch --parsable \
  --account="${ACCOUNT}" --partition="${PARTITION}" \
  --job-name=fsae_combine_sizes \
  --cpus-per-task=2 --mem=8G --time="00:20:00" \
  --output="%x_%j.out" --error="%x_%j.out" \
  --export="ALL,HPC_DIR=${HERE}" --chdir="${HERE}" \
  --dependency="afterok:${DEP}" \
  --wrap="bash '${HERE}/_combine_model_sweep.sh'")
echo ""
echo "submitted combine  : ${JID_COMBINE}  (afterok:${DEP})"

echo "=============================================================="
echo " ALL SIZES SUBMITTED. Watch:  squeue -u \$USER"
echo " Combined table lands in: ${OUT_DIR}_model_sweep/cfs_model_size_sweep.csv"
echo "=============================================================="
