#!/bin/bash
# =============================================================================
# run_all.sbatch  --  run the whole FAITH-SAE suite as a DETACHED batch job.
# Immune to SSH disconnects: the cluster runs it with no terminal attached.
#
#   cd into the quickstart folder, then:   sbatch run_all.sbatch
#   watch progress:                        tail -f faith_all_<jobid>.log
#   check it is running:                   squeue -u $USER
#
# If the conda env or lab paths differ, edit PY / HF_HOME below.
# =============================================================================
#SBATCH --job-name=faith_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=faith_all_%j.log

echo "=== FAITH-SAE full suite starting on $(hostname) at $(date) ==="
cd "$SLURM_SUBMIT_DIR" || exit 1
export HF_HOME="$HOME/rizk_lab/shared/rajia/hf_cache"
PY="$HOME/miniforge3/envs/sae_research/bin/python"
echo "submit dir : $SLURM_SUBMIT_DIR"
echo "python     : $PY"
echo "HF_HOME    : $HF_HOME"
"$PY" run_all.py
echo "=== suite finished at $(date) (exit $?) ==="
