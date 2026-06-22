#!/usr/bin/env bash
# =============================================================================
# submit_all.sh  --  submit the WHOLE pipeline with SLURM dependencies.
# FAITH-SAE real-scale pipeline  ·  SLURM HPC launch kit
# Author: Rajia Rani  ()
#
# Submits the four batch stages in dependency order so the run is hands-off:
#     02_extract (job array, 4 datasets)  --afterok-->
#     03_train_sae                         --afterok-->
#     04_experiments                       --afterok-->
#     05_analysis (CPU)
#
# This runs ONE backbone size ($BACKBONE). To run the MODEL-SIZE SWEEP (all three
# sizes, the second axis of the study), use submit_model_sweep.sh instead.
# Each stage starts ONLY if the previous finished OK (--dependency=afterok). For
# the array, 'afterok:<JOBID>' waits for ALL array tasks to succeed.
#
# Every cluster-specific value comes from cluster_env.sh and is passed on the
# sbatch command line (CLI flags OVERRIDE the #SBATCH defaults inside each file),
# so you only ever edit cluster_env.sh.
#
# Run on the LOGIN node, after 00_setup_env.sh + 01_stage_data.sh:
#     bash submit_all.sh
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
echo " FAITH-SAE  --  submit_all  (SLURM)"
echo "   account   : ${ACCOUNT}      partition : ${PARTITION}"
echo "   gpu gres  : ${GPU_GRES}     constraint: ${GPU_CONSTRAINT:-<any>}"
echo "   backbone  : ${BACKBONE}  ->  ${CONFIG}"
echo "   cache_dir : ${CACHE_DIR}"
echo "   out_dir   : ${OUT_DIR}"
echo "=============================================================="

# Common sbatch flags for the GPU stages (export HPC_DIR so the jobs can re-find
# cluster_env.sh; --export=ALL carries the sourced env into the job).
export HPC_DIR="${HERE}"
GPU_COMMON=(
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --gres="${GPU_GRES}"
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
  --export=ALL
  --chdir="${HERE}"
)
# Add the optional GPU-type constraint only if the user set one.
[[ -n "${GPU_CONSTRAINT}" ]] && GPU_COMMON+=(--constraint="${GPU_CONSTRAINT}")

# --- Stage 02: EXTRACT (job array over the 4 datasets) ------------------------
JID_EXTRACT=$(sbatch --parsable \
  "${GPU_COMMON[@]}" \
  --time="${TIME_EXTRACT}" \
  --array=0-3 \
  "${HERE}/02_extract.sbatch")
echo "submitted 02_extract     : job ${JID_EXTRACT}  (array 0-3)"

# --- Stage 03: TRAIN SAE (after ALL extract array tasks succeed) --------------
JID_TRAIN=$(sbatch --parsable \
  "${GPU_COMMON[@]}" \
  --time="${TIME_TRAIN}" \
  --dependency="afterok:${JID_EXTRACT}" \
  "${HERE}/03_train_sae.sbatch")
echo "submitted 03_train_sae   : job ${JID_TRAIN}   (afterok:${JID_EXTRACT})"

# --- Stage 04: EXPERIMENTS (after SAE trains) --------------------------------
JID_EXPT=$(sbatch --parsable \
  "${GPU_COMMON[@]}" \
  --time="${TIME_EXPERIMENTS}" \
  --dependency="afterok:${JID_TRAIN}" \
  "${HERE}/04_experiments.sbatch")
echo "submitted 04_experiments : job ${JID_EXPT}   (afterok:${JID_TRAIN})"

# --- Stage 05: ANALYSIS (CPU; no --gres) -------------------------------------
JID_ANALYSIS=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --cpus-per-task=4 \
  --mem=32G \
  --time="${TIME_ANALYSIS}" \
  --export=ALL \
  --chdir="${HERE}" \
  --dependency="afterok:${JID_EXPT}" \
  "${HERE}/05_analysis.sbatch")
echo "submitted 05_analysis    : job ${JID_ANALYSIS}   (afterok:${JID_EXPT})"

echo "=============================================================="
echo " ALL SUBMITTED. Job IDs:"
echo "   extract=${JID_EXTRACT}  train=${JID_TRAIN}  experiments=${JID_EXPT}  analysis=${JID_ANALYSIS}"
echo ""
echo " Watch:   squeue -u \$USER"
echo "          sacct  -j ${JID_EXTRACT},${JID_TRAIN},${JID_EXPT},${JID_ANALYSIS} --format=JobID,JobName,State,Elapsed"
echo "          tail -f ${HERE}/fsae_extract_${JID_EXTRACT}_0.out"
echo " Results land in: ${OUT_DIR}  (FINDINGS.md, fig1_*.png, fig7_*.png)"
echo "=============================================================="
