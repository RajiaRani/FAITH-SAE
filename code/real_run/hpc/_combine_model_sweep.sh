#!/usr/bin/env bash
# =============================================================================
# _combine_model_sweep.sh  --  stitch the per-size OOD sweeps into ONE table.
# FAITH-SAE real-scale pipeline  ·  SLURM HPC launch kit
# Author: Rajia Rani  ()
#
# Called (via sbatch --wrap) by submit_model_sweep.sh AFTER every model size has
# finished its 05_analysis stage. Reads each size's OUT_DIR_<size>/ood_cfs_sweep.csv,
# tags every row with the model_size + backbone name + d_in, and concatenates them
# into the combined CFS-vs-(model_size x dataset) table the paper plots.
#
# Output: ${OUT_DIR}_model_sweep/cfs_model_size_sweep.csv
#
# For research and educational purposes only.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"
activate_env

COMBINE_OUT="${OUT_DIR}_model_sweep"
mkdir -p "${COMBINE_OUT}"

echo "=============================================================="
echo " [combine] stitching per-size OOD sweeps into one table"
echo "   sizes : ${MODEL_SIZES}"
echo "   out   : ${COMBINE_OUT}/cfs_model_size_sweep.csv"
echo "=============================================================="

MODEL_SIZES="${MODEL_SIZES}" OUT_DIR="${OUT_DIR}" REAL_RUN_DIR="${REAL_RUN_DIR}" \
COMBINE_OUT="${COMBINE_OUT}" python3 - <<'PYEOF'
import os, sys, pathlib
import pandas as pd

real_run = os.environ["REAL_RUN_DIR"]
sys.path.insert(0, real_run)
from data_real import load_real_config

base_out = os.environ["OUT_DIR"]
sizes = os.environ["MODEL_SIZES"].split()
frames = []
for size in sizes:
    sweep = pathlib.Path(f"{base_out}_{size}") / "ood_cfs_sweep.csv"
    cfg_path = os.path.join(real_run, "configs", f"{size}.yaml")
    if not sweep.exists():
        print(f"  WARNING: missing {sweep}; skipping size {size}")
        continue
    df = pd.read_csv(sweep)
    backbone = d_in = ""
    if os.path.exists(cfg_path):
        cfg = load_real_config(cfg_path)
        backbone = str(cfg.backbone.name)
        d_in = int(cfg.sae.d_in)
    df.insert(0, "model_size", size)
    df.insert(1, "backbone", backbone)
    df.insert(2, "d_in", d_in)
    frames.append(df)

if not frames:
    sys.exit("  ERROR: no per-size ood_cfs_sweep.csv found; did the sizes finish?")

combined = pd.concat(frames, ignore_index=True)
out = pathlib.Path(os.environ["COMBINE_OUT"]) / "cfs_model_size_sweep.csv"
combined.to_csv(out, index=False)
print(f"  wrote {out}  ({len(combined)} rows, "
      f"{combined['model_size'].nunique()} sizes)")
PYEOF

echo "[combine] done -> ${COMBINE_OUT}/cfs_model_size_sweep.csv"
