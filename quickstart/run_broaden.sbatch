#!/bin/bash
# Broadened-concept re-run: 40 concepts instead of 8, on the two CLIP experiments
# that produce the headline result. Disconnect-proof (sbatch). ~15-20 min.
#   cd into the quickstart folder, then:  sbatch run_broaden.sbatch
#SBATCH --job-name=faith_broaden
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=faith_broaden_%j.log

echo "=== broadened-concept run (40 concepts) on $(hostname) at $(date) ==="
cd "$SLURM_SUBMIT_DIR" || exit 1
export HF_HOME="$HOME/rizk_lab/shared/rajia/hf_cache"
PY="$HOME/miniforge3/envs/sae_research/bin/python"
echo ">>> M7  CLIP semantic faithfulness, 40 concepts:"
"$PY" run_clip_faith.py --concepts 40
echo ">>> M8  positive control + TCAV, 40 concepts:"
"$PY" run_clip_control.py --concepts 40
echo "=== done at $(date) (exit $?) ==="
