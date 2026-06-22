#!/usr/bin/env bash
# =============================================================================
# 00_setup_env.sh  --  ONE-TIME python environment setup.  RUN ON THE LOGIN NODE.
# FAITH-SAE real-scale pipeline  ·  SLURM HPC launch kit
# Author: Rajia Rani  ()
#
# This is NOT a SLURM batch job -- run it directly on the login/head node:
#     bash 00_setup_env.sh
# It loads the cluster modules, creates the conda (or venv) environment named in
# cluster_env.sh, and pip-installs ../requirements_real.txt (torch, timm,
# torchvision, ...). Do this ONCE; afterwards the batch jobs just `activate_env`.
#
# NOTE: `module load` names vary on EVERY cluster -- the lines below are
# PLACEHOLDERS. Run `module avail` / `module spider cuda` on your cluster and
# replace them with the real module names. If your site has no module system,
# just delete the module lines.
#
# For research and educational purposes only.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster_env.sh
source "${HERE}/cluster_env.sh"

echo "=============================================================="
echo " FAITH-SAE  --  00_setup_env  (login node, one time)"
echo "   backbone   : ${BACKBONE}  ->  ${CONFIG}"
echo "   env        : $([[ "${USE_VENV}" -eq 1 ]] && echo "venv @ ${VENV_DIR}" || echo "conda env '${CONDA_ENV}'")"
echo "   real_run   : ${REAL_RUN_DIR}"
echo "=============================================================="

# ---------------------------------------------------------------------------
# 1. MODULE LOADS  --  PLACEHOLDERS. Replace with YOUR cluster's module names.
#    Find them with:  module avail   |   module spider cuda   |   module spider anaconda
# ---------------------------------------------------------------------------
echo "[1/4] loading modules (edit these to match your cluster) ..."
if command -v module >/dev/null 2>&1; then
  # --- BEGIN PLACEHOLDER MODULE LINES (edit me) ---------------------------
  module load anaconda3   2>/dev/null || echo "  (skip) anaconda3 module not found -- edit 00_setup_env.sh"
  module load cuda/12.1   2>/dev/null || echo "  (skip) cuda/12.1 module not found -- edit 00_setup_env.sh"
  # module load gcc/11.2.0            # some sites need a compiler for builds
  # --- END PLACEHOLDER MODULE LINES ---------------------------------------
else
  echo "  (no 'module' command on this node -- skipping; that's fine if your site has none)"
fi

# ---------------------------------------------------------------------------
# 2. CREATE the python environment (conda OR venv, per cluster_env.sh).
# ---------------------------------------------------------------------------
if [[ "${USE_VENV}" -eq 1 ]]; then
  echo "[2/4] creating venv at ${VENV_DIR} ..."
  if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
  else
    echo "  venv already exists -- reusing."
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip
else
  echo "[2/4] creating conda env '${CONDA_ENV}' ..."
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not on PATH. Either 'module load anaconda3' (fix the module" >&2
    echo "       name above) or set USE_VENV=1 in cluster_env.sh to use a venv." >&2
    exit 1
  fi
  eval "$(conda shell.bash hook)"
  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -y -n "${CONDA_ENV}" python=3.10
  else
    echo "  conda env '${CONDA_ENV}' already exists -- reusing."
  fi
  conda activate "${CONDA_ENV}"
fi

# ---------------------------------------------------------------------------
# 3. INSTALL the real-run requirements (torch, open_clip, datasets, ...).
#    NOTE: on many clusters you want a CUDA-matched torch wheel. If the default
#    install gives a CPU-only torch, install torch FIRST from the right index,
#    e.g.:  pip install torch --index-url https://download.pytorch.org/whl/cu121
#    then re-run this script (it will pick up the rest).
# ---------------------------------------------------------------------------
echo "[3/4] pip installing ${REAL_RUN_DIR}/requirements_real.txt ..."
python -m pip install -r "${REAL_RUN_DIR}/requirements_real.txt"

# ---------------------------------------------------------------------------
# 4. FINAL "env ready" CHECK  --  prove torch + timm + torchvision import.
#    (timm is the DEFAULT backbone; open_clip is optional, only for the legacy
#    framework, so we don't hard-fail on it.)
# ---------------------------------------------------------------------------
echo "[4/4] verifying the environment ..."
python -c "import torch, timm, torchvision; print('  torch', torch.__version__, '| cuda available:', torch.cuda.is_available()); print('  timm', timm.__version__, '| torchvision', torchvision.__version__)"
python -c "import open_clip; print('  open_clip', open_clip.__version__)" 2>/dev/null || echo "  open_clip not installed (OK -- only needed for backbone.framework: open_clip)"

echo ""
echo "=============================================================="
echo " env ready."
echo "   Next: edit data paths are in cluster_env.sh, then run:"
echo "     bash 01_stage_data.sh        # download Food-101 + CIFAR-100, patch configs"
echo "     bash submit_model_sweep.sh   # launch ALL 3 sizes (the model-size axis)"
echo "     bash submit_all.sh           # or just the single \$BACKBONE size"
echo "=============================================================="
