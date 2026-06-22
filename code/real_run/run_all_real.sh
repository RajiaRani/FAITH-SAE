#!/usr/bin/env bash
# =============================================================================
# run_all_real.sh  --  FAITH-SAE  code/real_run/  end-to-end pipeline driver
# Author: Rajia Rani  ()
#
# Runs the WHOLE publication pipeline in the correct dependency order:
#
#   extract -> train_sae -> concept_select -> manifold -> cfs_eval (self-test)
#           -> ood_sweep -> ablations -> analysis (+ figures)
#
# Two modes (the ONLY switch you need):
#   --smoke : tiny CPU path on synthetic-but-real-SHAPED tensors. NO open_clip,
#             NO downloads, NO GPU. Every stage runs its own --smoke branch on the
#             tiny configs/smoke.yaml (d_in=64, a few thousand tokens) and the
#             whole thing finishes in well under a minute. Use this to PROVE the
#             modules wire together before you rent a GPU. Smoke writes only to
#             the smoke cache/outputs (cache_smoke/, outputs_smoke/), so it never
#             touches the real cache/ or outputs/.
#   (default / --real) : the real GPU run. Reads the real config (default
#             configs/vit_l14.yaml) + the real datasets you have staged. Expect
#             ~15-120 GPU-hours (see RUN_AT_SCALE.md and `python cost_estimate.py`).
#             DO NOT run this on the build machine -- it has no GPU and no
#             open_clip, so it will (correctly) stop at the first real stage.
#
# Usage:
#   bash run_all_real.sh --smoke
#   bash run_all_real.sh --real --config configs/vit_l14.yaml
#   bash run_all_real.sh --config configs/vit_l14.yaml   # --real is the default
#   PYTHON=/usr/bin/python3 bash run_all_real.sh --smoke  # pin the interpreter
#
# WHY the stage CLIs differ below (they are NOT all `--config --cache_dir`):
#   * EXTRACTION is per-DATASET via extract_activations.py (clean + each OOD rung)
#     -- there is no single "extract everything" call; the backbone forward is
#     run once per dataset and the shards are cached.
#   * cfs_eval.py is a LIBRARY (compute_cfs / evaluate_all_methods) imported by
#     ood_sweep + ablations; its only CLI is `--smoke` (an offline self-test).
#     The real clean-vs-OOD CFS table is produced by ood_sweep (the 'clean' rung
#     IS the RQ1 clean CFS).
#   * analysis_real.py / figures_real.py read the RESULT CSVs from an output dir,
#     so they take --results-dir / --out-dir, not --config / --cache_dir.
#
# Honesty note: on a CPU-only box ONLY --smoke completes; the real path stops at
# the first stage that needs open_clip / CUDA / the datasets, which is correct
# and intended. For research and educational purposes only.
# =============================================================================
set -euo pipefail

# --- resolve paths relative to THIS script (so it runs from anywhere) --------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

# --- interpreter: override with PYTHON=... ; default to python3 on PATH -------
PYTHON="${PYTHON:-python3}"

# --- arg parsing -------------------------------------------------------------
SMOKE=0
MODE="real"
CONFIG=""
CACHE_DIR=""
OUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; MODE="smoke"; shift ;;
    --real)  MODE="real"; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
    --cache_dir|--cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --out_dir|--out-dir) OUT_DIR="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -48
      exit 0 ;;
    *) echo "[run_all_real] unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --- pick the config + the cache/out dirs if the caller did not --------------
# The real run uses the full-scale config (default configs/vit_l14.yaml; swap in
# vit_b16.yaml / vit_h14.yaml to change backbone). Smoke uses the tiny twin
# configs/smoke.yaml, whose paths point at cache_smoke/ + outputs_smoke/ so the
# offline path never collides with the real, git-ignored cache/ + outputs/.
if [[ -z "${CONFIG}" ]]; then
  if [[ "${SMOKE}" -eq 1 ]]; then CONFIG="configs/smoke.yaml"; else CONFIG="configs/vit_l14.yaml"; fi
fi
if [[ -z "${CACHE_DIR}" ]]; then
  if [[ "${SMOKE}" -eq 1 ]]; then CACHE_DIR="./cache_smoke"; else CACHE_DIR="./cache"; fi
fi
if [[ -z "${OUT_DIR}" ]]; then
  if [[ "${SMOKE}" -eq 1 ]]; then OUT_DIR="./outputs_smoke"; else OUT_DIR="./outputs"; fi
fi

# Flag blocks shared across stages. Heavy stages take --config + --cache_dir;
# smoke adds --smoke. The analysis/figure stages instead take --results-dir/
# --out-dir (they consume the result CSVs, not the config), so they get their
# own flag list below.
SMOKE_FLAG=""
[[ "${SMOKE}" -eq 1 ]] && SMOKE_FLAG="--smoke"
COMMON=(--config "${CONFIG}" --cache_dir "${CACHE_DIR}")

echo "=============================================================="
echo " FAITH-SAE real_run pipeline"
echo "   mode      : ${MODE}"
echo "   python    : ${PYTHON}"
echo "   config    : ${CONFIG}"
echo "   cache_dir : ${CACHE_DIR}"
echo "   out_dir   : ${OUT_DIR}"
echo "=============================================================="
mkdir -p "${CACHE_DIR}" "${OUT_DIR}"

# Helper: announce + run a stage, fail loudly with the stage name.
stage () {
  local name="$1"; shift
  echo ""
  echo "----- [stage] ${name} ------------------------------------------"
  if ! "$@"; then
    echo "[run_all_real] FAILED at stage: ${name}" >&2
    echo "  (on a CPU-only box the real path stops here by design; try --smoke)" >&2
    exit 1
  fi
}

# =============================================================================
# THE PIPELINE -- strict dependency order. Each stage consumes the previous
# stage's artifacts (activation shards -> SAE ckpt -> manifold basis -> CFS).
# =============================================================================

# 1) EXTRACT activations for EVERY rung of the student's domain-shift ladder into
#    the cache (sharded [n_tokens, d_in] fp16 patch tokens + per-token labels +
#    manifest json). This is the heavy backbone-forward stage; it is run ONCE
#    PER DATASET via extract_activations.py (data_real.extract_activations).
#    in1k is the in-distribution rung AND the SAE-training source.
#    In smoke mode each call fabricates tiny synthetic-but-real-shaped shards.
for DS in in1k in100 food101 cifar100; do
  stage "1/8 extract_activations[${DS}]" \
    "${PYTHON}" extract_activations.py --config "${CONFIG}" \
        --dataset "${DS}" --cache_dir "${CACHE_DIR}" ${SMOKE_FLAG}
done

# 2) TRAIN the TopK SAE by streaming the cached clean-train shards (AdamW +
#    warmup + AMP, dead-feature tracking, FVU/L0/dead% metrics, checkpoints).
#    Writes <out_dir>/sae.safetensors. (train_sae.train_sae)
stage "2/8 train_sae" \
  "${PYTHON}" train_sae.py ${SMOKE_FLAG} "${COMMON[@]}"

# 3) SELECT the reliable, testable concepts (the field's ~10-15% tail) via
#    max-activating images + reliability score. Writes the concept id list.
#    (concept_select.select_concepts)
stage "3/8 concept_select" \
  "${PYTHON}" concept_select.py ${SMOKE_FLAG} "${COMMON[@]}"

# 4) MANIFOLD basis: SVD the real-image activation bank to get U_r[d, r], the
#    on-manifold projection subspace. Writes <out_dir>/U_r.npy.
#    (manifold.estimate_manifold_basis)
stage "4/8 manifold" \
  "${PYTHON}" manifold.py ${SMOKE_FLAG} "${COMMON[@]}"

# 5) CFS SELF-TEST: cfs_eval.py is a LIBRARY (compute_cfs / evaluate_all_methods)
#    that ood_sweep + ablations import. Its only CLI is the offline --smoke
#    self-test (it asserts onmanifold > naive/random on planted concepts). The
#    REAL clean-vs-OOD CFS table is produced by stage 6 (ood_sweep), whose
#    'clean' rung IS the RQ1 clean CFS -- so on the real path we just confirm
#    the scorer is healthy here and let ood_sweep do the measurement.
if [[ "${SMOKE}" -eq 1 ]]; then
  stage "5/8 cfs_eval (self-test)" \
    "${PYTHON}" cfs_eval.py --smoke
else
  echo ""
  echo "----- [stage] 5/8 cfs_eval (library; CFS computed in stage 6) --------"
  echo "  cfs_eval.py is imported by ood_sweep/ablations; no standalone real run."
fi

# 6) OOD SWEEP: re-run CFS across the OOD ladder (clean -> R -> Sketch -> C
#    sev1..5 -> ObjectNet) reusing each dataset's cached acts. The RQ1 (clean
#    rung) + RQ3 (full ladder) answer. Writes <out_dir>/ood_cfs_sweep.csv.
#    (ood_sweep.run_ood_sweep)
stage "6/8 ood_sweep" \
  "${PYTHON}" ood_sweep.py ${SMOKE_FLAG} "${COMMON[@]}"

# 7) ABLATIONS A1-A5: SAE type, k, proj-rank r, selection threshold, layer/token.
#    Writes <out_dir>/ablations.csv. (ablations_real.run_ablations)
stage "7/8 ablations" \
  "${PYTHON}" ablations_real.py ${SMOKE_FLAG} "${COMMON[@]}"

# 8) ANALYSIS + FIGURES: bootstrap CIs over concepts (-> bootstrap_ci.csv +
#    FINDINGS.md) and render the figure manifest (at least fig1_cfs_ood_sweep.png
#    + fig7_by_method_bar.png). These read the result CSVs from <out_dir>, so
#    they take --results-dir / --out-dir (not --config / --cache_dir).
#    (analysis_real.main ; figures_real.make_real_figures)
stage "8/8 analysis" \
  "${PYTHON}" analysis_real.py ${SMOKE_FLAG} \
      --results-dir "${OUT_DIR}" --out-dir "${OUT_DIR}"
stage "8/8 figures" \
  "${PYTHON}" figures_real.py ${SMOKE_FLAG} \
      --results-dir "${OUT_DIR}" --out-dir "${OUT_DIR}"

echo ""
echo "=============================================================="
echo " DONE (${MODE}). Artifacts in ${OUT_DIR} :"
echo "   sae.safetensors  U_r.npy  ood_cfs_sweep.csv  ablations.csv"
echo "   bootstrap_ci.csv  FINDINGS.md"
echo "   fig1_cfs_ood_sweep.png  fig7_by_method_bar.png  (+ manifest figs)"
echo "=============================================================="
