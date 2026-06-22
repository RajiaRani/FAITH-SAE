#!/usr/bin/env bash
# =============================================================================
# 01_stage_data.sh  --  ONE-TIME data staging.  RUN ON THE LOGIN NODE.
# FAITH-SAE real-scale pipeline  ·  SLURM HPC launch kit
# Author: Rajia Rani  ()
#
# What it does (in order):
#   1. VERIFY $IMAGENET_DIR exists -- ImageNet-1k is ALREADY on the cluster's
#      shared storage; we do NOT download it. It serves BOTH the in1k rung AND
#      the in100 rung (a 100-class subset filtered by in100_classes.txt).
#   2. TRIGGER the two SMALL torchvision downloads (Food-101 ~5 GB, CIFAR-100
#      ~170 MB) into $DATA_DIR -- torchvision fetches them itself (no forms, no
#      accounts). If torchvision isn't importable yet it prints the one-liner.
#   3. NOTE that IN-100 is derived from IN-1k via real_run/in100_classes.txt
#      (nothing to download).
#   4. PATCH the chosen model-size config(s) so data.*, cache_dir and out_dir
#      point at YOUR cluster paths -- because the config loader reads the YAML
#      literally (it does NOT expand env vars), the paths must be baked in.
#
# Run on the login node (it downloads the small sets; it does not need a GPU):
#     bash 01_stage_data.sh
#
# For research and educational purposes only.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"

echo "=============================================================="
echo " FAITH-SAE  --  01_stage_data  (login node, one time)"
echo "   IMAGENET_DIR : ${IMAGENET_DIR}"
echo "   DATA_DIR     : ${DATA_DIR}  (Food-101 + CIFAR-100 download here)"
echo "   framework    : ${BACKBONE_FRAMEWORK}"
echo "   model sizes  : ${MODEL_SIZES}"
echo "=============================================================="

# ---------------------------------------------------------------------------
# 1. VERIFY ImageNet-1k is already present (we do NOT download it).
#    It backs BOTH in1k (in-distribution) AND in100 (a 100-class subset).
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] checking ImageNet-1k at \$IMAGENET_DIR ..."
if [[ ! -d "${IMAGENET_DIR}" ]]; then
  echo "ERROR: IMAGENET_DIR does not exist:" >&2
  echo "       ${IMAGENET_DIR}" >&2
  echo "  ImageNet-1k is supposed to be ALREADY on your cluster's shared storage." >&2
  echo "  Ask your admin for the path and set IMAGENET_DIR in cluster_env.sh." >&2
  echo "  Expected ImageFolder layout: \$IMAGENET_DIR/n01440764/*.JPEG" >&2
  exit 1
fi
_n_classes="$(find "${IMAGENET_DIR}" -maxdepth 1 -type d -name 'n*' 2>/dev/null | wc -l | tr -d ' ')"
echo "  found ImageNet dir; ~${_n_classes} class subdirs (expect ~1000)."
[[ "${_n_classes}" -lt 100 ]] && echo "  WARNING: <100 class dirs -- is this really the ImageNet TRAIN split?"

mkdir -p "${DATA_DIR}"

# ---------------------------------------------------------------------------
# 2. TRIGGER the small torchvision downloads (Food-101 + CIFAR-100).
#    torchvision downloads these automatically the FIRST time the dataset is
#    constructed with download=True. We pre-trigger here so the GPU extraction
#    jobs don't waste GPU time downloading. Both are small + license-free.
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] pre-downloading Food-101 (~5 GB) + CIFAR-100 (~170 MB) into ${DATA_DIR} ..."
# Run the torchvision download in a heredoc; if it fails (e.g. torchvision not
# importable yet) print the manual one-liner instead of aborting the script. The
# heredoc is kept SEPARATE from the fallback so `bash -n` parses cleanly.
if DATA_DIR="${DATA_DIR}" python3 - <<'PYEOF'
import os
import torchvision
d = os.environ["DATA_DIR"]
# Food-101: 101 food classes; we use the official 'test' split for the eval rung.
torchvision.datasets.Food101(os.path.join(d, "food101"), split="test", download=True)
# CIFAR-100: 100 classes, 32x32 (upsampled to 224 by the ViT preprocess -> a
# STRONG domain + resolution shift). We use the test split for the eval rung.
torchvision.datasets.CIFAR100(os.path.join(d, "cifar100"), train=False, download=True)
print("  Food-101 + CIFAR-100 ready under", d)
PYEOF
then
  :
else
  echo "  >> NOTE: torchvision not importable yet (activate the env first:"
  echo "     bash 00_setup_env.sh). The GPU extraction job will download them"
  echo "     on first use anyway. Or run this one-liner after activating the env:"
  echo "       python3 -c \"import torchvision,os; d=os.environ.get('DATA_DIR','./data');"
  echo "       torchvision.datasets.Food101(os.path.join(d,'food101'),split='test',download=True);"
  echo "       torchvision.datasets.CIFAR100(os.path.join(d,'cifar100'),train=False,download=True)\""
fi

# ---------------------------------------------------------------------------
# 3. IN-100 is DERIVED from IN-1k (no download).
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] IN-100 ..."
_IN100_FILE="${REAL_RUN_DIR}/in100_classes.txt"
_IN100_N="$(grep -vcE '^\s*(#|$)' "${_IN100_FILE}" 2>/dev/null || echo 0)"
echo "  IN-100 is a 100-class SUBSET of IN-1k -- nothing to download."
echo "  It is filtered from \$IMAGENET_DIR using the wnid list in:"
echo "    ${_IN100_FILE}  (currently ${_IN100_N} uncommented wnids; expect ~100)."
[[ "${_IN100_N}" -lt 100 ]] && echo "  >> FILL IN in100_classes.txt with the 100 IN-100 wnids (see the file header)."

# ---------------------------------------------------------------------------
# 4. PATCH every model-size config so its paths point at THIS cluster.
#    The config loader (data_real.load_real_config) reads the YAML LITERALLY --
#    it does not expand $VARS -- so we bake the resolved paths into each file.
#    We patch ALL of $MODEL_SIZES so the model-size sweep is ready to run.
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] patching the model-size configs with your cluster paths ..."
for _size in ${MODEL_SIZES}; do
  _cfg="$(config_for_size "${_size}")"
  if [[ ! -f "${_cfg}" ]]; then
    echo "  WARNING: config not found, skipping: ${_cfg}" >&2
    continue
  fi
  echo "  patching ${_cfg} ..."
  IMAGENET_DIR="${IMAGENET_DIR}" DATA_DIR="${DATA_DIR}" \
  CACHE_DIR="${CACHE_DIR}_${_size}" OUT_DIR="${OUT_DIR}_${_size}" \
  IN100_FILE="${_IN100_FILE}" FRAMEWORK="${BACKBONE_FRAMEWORK}" \
  CONFIG="${_cfg}" python3 - <<'PYEOF'
import os, sys
cfg_path = os.environ["CONFIG"]
try:
    import yaml
except Exception:
    sys.exit("PyYAML not importable; activate the env first (00_setup_env.sh) or "
             "edit the config's data.* / paths.* by hand.")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

bb = cfg.setdefault("backbone", {})
bb["framework"] = os.environ["FRAMEWORK"]            # timm (default) | open_clip

data = cfg.setdefault("data", {})
data["imagenet_train_dir"] = os.environ["IMAGENET_DIR"]   # backs in1k AND in100
data["in100_classes_file"] = os.environ["IN100_FILE"]     # the 100-class wnid list
data["data_dir"]           = os.environ["DATA_DIR"]       # Food-101 / CIFAR-100 root

paths = cfg.setdefault("paths", {})
paths["cache_dir"] = os.environ["CACHE_DIR"]              # per-size cache dir
paths["out_dir"]   = os.environ["OUT_DIR"]                # per-size output dir
paths["sae_ckpt"]       = os.path.join(os.environ["OUT_DIR"], "sae.safetensors")
paths["manifold_basis"] = os.path.join(os.environ["OUT_DIR"], "U_r.npy")

with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
print("    backbone.framework ->", bb["framework"], "| name ->", bb.get("name"))
print("    data.imagenet_train_dir ->", data["imagenet_train_dir"])
print("    data.data_dir (food101/cifar100) ->", data["data_dir"])
print("    paths.cache_dir ->", paths["cache_dir"])
print("    paths.out_dir   ->", paths["out_dir"])
PYEOF
  mkdir -p "${CACHE_DIR}_${_size}" "${OUT_DIR}_${_size}"
done

echo ""
echo "=============================================================="
echo " data staging done."
echo "   ImageNet-1k : reused in place (in1k + in100); not downloaded."
echo "   Food-101    : torchvision download under ${DATA_DIR}/food101."
echo "   CIFAR-100   : torchvision download under ${DATA_DIR}/cifar100."
echo "   IN-100      : derived from IN-1k via in100_classes.txt."
echo "   configs     : ${MODEL_SIZES} now point at your cluster paths."
echo "   Next:  bash submit_model_sweep.sh   (all 3 sizes)"
echo "      or  bash submit_all.sh           (single \$BACKBONE only)"
echo "=============================================================="
