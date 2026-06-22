#!/usr/bin/env bash
# =============================================================================
# run_model_size_sweep.sh  --  FAITH-SAE  ·  the MODEL-SIZE axis (S/B/L) runner
# Author: Rajia Rani  ()
#
# Runs the full FAITH-SAE pipeline for EACH of the three supervised-ViT sizes and
# collects CFS vs (model_size x dataset) into ONE combined CSV. This is the second
# axis of the study: faithfulness x MODEL-SIZE x domain-shift.
#
#   ViT-S (~22M, configs/vit_s_20m.yaml)
#   ViT-B (~86M, configs/vit_b_84m.yaml)
#   ViT-L (~304M, configs/vit_l_307m.yaml)
#
# Two modes (the ONLY switch you need):
#   --smoke : tiny CPU path across the 3 widths (384/768/1024) on synthetic-but-
#             real-SHAPED tensors. NO timm, NO downloads, NO GPU. Finishes in
#             seconds. Use this to PROVE the model-size axis wires together.
#   (default / --real) : the real GPU run. Runs run_all_real.sh per size config,
#             then stacks each run's ood_cfs_sweep.csv into the combined CSV.
#             Budget ~3x a single-size run (see RUN_AT_SCALE.md).
#
# Usage:
#   bash run_model_size_sweep.sh --smoke
#   bash run_model_size_sweep.sh --real
#   PYTHON=/usr/bin/python3 bash run_model_size_sweep.sh --smoke   # pin interpreter
#
# Output: outputs_model_sweep/cfs_model_size_sweep.csv
#         (one row per model_size x dataset x steering method; columns include
#          model_size, backbone, d_in, rung/dataset, method, cfs + components).
#
# Honesty: on a CPU-only box ONLY --smoke completes; the real path stops at the
# first stage that needs timm / CUDA / the datasets, which is correct and
# intended. For research and educational purposes only.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

PYTHON="${PYTHON:-python3}"

SMOKE_FLAG=""
for arg in "$@"; do
  case "${arg}" in
    --smoke) SMOKE_FLAG="--smoke" ;;
    --real)  SMOKE_FLAG="" ;;
    -h|--help)
      grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -40
      exit 0 ;;
    *) echo "[run_model_size_sweep] unknown arg: ${arg}" >&2; exit 2 ;;
  esac
done

echo "=============================================================="
echo " FAITH-SAE  --  model-size sweep (ViT-S / ViT-B / ViT-L)"
echo "   mode   : ${SMOKE_FLAG:-real}"
echo "   python : ${PYTHON}"
echo "=============================================================="

# The python driver runs each size's pipeline and writes the combined CSV. On the
# real path it shells back into run_all_real.sh per config (PYTHON is forwarded).
PYTHON="${PYTHON}" "${PYTHON}" model_size_sweep.py ${SMOKE_FLAG} --python "${PYTHON}"

echo ""
echo "=============================================================="
echo " DONE. Combined CFS-vs-(model_size x dataset) table:"
echo "   outputs_model_sweep/cfs_model_size_sweep.csv"
echo "=============================================================="
