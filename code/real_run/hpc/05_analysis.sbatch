#!/usr/bin/env bash
# =============================================================================
# 05_analysis.sbatch  --  STAGE 8: bootstrap CIs + figures + FINDINGS  (CPU).
# FAITH-SAE real-scale pipeline  ·  SLURM HPC launch kit
# Author: Rajia Rani  ()
#
# CPU-ONLY job (NO GPU requested -- it just reads the result CSVs). It:
#   (glue) Explodes the per-concept CFS that the OOD sweep embeds inside
#          ood_cfs_sweep.csv (column 'cfs_per_concept', ';'-joined) into the
#          standalone per_concept_cfs.csv that analysis_real / figures_real read.
#          This is the one documented glue step between stage 6 and stage 8 (see
#          RUN_AT_SCALE.md "Per-concept CSV caveat"). It only RESHAPES measured
#          numbers -- it does not fabricate any.
#   8a) analysis_real.py  -> bootstrap_ci.csv, findings.json, FINDINGS.md
#   8b) figures_real.py   -> fig1_cfs_ood_sweep.png, fig7_by_method_bar.png
# Depends on 04_experiments (needs ood_cfs_sweep.csv); chained via afterok.
#
# Note: NO #SBATCH --gres line (CPU job). submit_all.sh does not pass --gres here.
#
# For research and educational purposes only.
# =============================================================================
#SBATCH --job-name=fsae_analysis
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
set -euo pipefail

HPC_DIR="${HPC_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# shellcheck source=cluster_env.sh
source "${HPC_DIR}/cluster_env.sh"
activate_env

cd "${REAL_RUN_DIR}"

echo "=============================================================="
echo " [05_analysis]  (CPU)  results -> CIs + figures + FINDINGS"
echo "   out_dir : ${OUT_DIR}"
echo "   node    : $(hostname)"
echo "=============================================================="

# --- GLUE: build per_concept_cfs.csv from the OOD sweep's embedded column -----
echo ""; echo "----- [glue] explode cfs_per_concept -> per_concept_cfs.csv --"
OUT_DIR="${OUT_DIR}" python3 - <<'PYEOF'
import os, pathlib, sys
out = pathlib.Path(os.environ["OUT_DIR"])
sweep = out / "ood_cfs_sweep.csv"
per   = out / "per_concept_cfs.csv"
if not sweep.exists():
    sys.exit(f"[glue] ERROR: {sweep} not found -- did stage 04 (ood_sweep) run?")
import pandas as pd
df = pd.read_csv(sweep)
if "cfs_per_concept" not in df.columns:
    sys.exit("[glue] ERROR: ood_cfs_sweep.csv lacks 'cfs_per_concept' column "
             "(schema changed upstream).")
rows = []
for _, r in df.iterrows():
    raw = str(r["cfs_per_concept"]).strip()
    vals = [v for v in raw.split(";") if v != ""]
    for ci, v in enumerate(vals):
        try:
            cfs = float(v)
        except ValueError:
            continue
        rows.append({
            # column names match the analysis/figures auto-detectors
            # (_method_col accepts 'method'; level detection accepts 'rung').
            "method": r["method"],
            "rung": r["rung"],
            "shift": r["rung"],            # alias the detectors also look for
            "shift_index": r.get("shift_index", ""),
            "concept_id": ci,
            "cfs": cfs,
            # component means are per-RUNG (not per-concept) upstream; carry them
            # so the schema is complete. Documented limitation, not fabrication.
            "monotonicity": r.get("monotonicity", ""),
            "specificity": r.get("specificity", ""),
            "sufficiency": r.get("sufficiency", ""),
        })
pd.DataFrame(rows).to_csv(per, index=False)
print(f"[glue] wrote {per}  ({len(rows)} per-concept rows from {len(df)} sweep rows)")
PYEOF

# --- 8a ANALYSIS: bootstrap CIs + significance + FINDINGS ----------------------
echo ""; echo "----- [8a] analysis_real --------------------------------------"
# REAL flags (verified via --help): --results-dir --out-dir [--n-boot --ci-pct --floor]
python analysis_real.py \
    --results-dir "${OUT_DIR}" \
    --out-dir     "${OUT_DIR}"

# --- 8b FIGURES: fig1 (OOD CFS sweep) + fig7 (CFS by method) -------------------
echo ""; echo "----- [8b] figures_real ---------------------------------------"
# REAL flags (verified via --help): --results-dir --out-dir [--floor --n-boot --ci-pct]
python figures_real.py \
    --results-dir "${OUT_DIR}" \
    --out-dir     "${OUT_DIR}"

echo ""
echo "=============================================================="
echo " [05_analysis] DONE. Paper-ready artifacts in ${OUT_DIR}:"
echo "   FINDINGS.md  findings.json  bootstrap_ci.csv"
echo "   fig1_cfs_ood_sweep.png  fig7_by_method_bar.png"
echo "=============================================================="
