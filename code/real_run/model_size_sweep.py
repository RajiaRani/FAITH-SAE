#!/usr/bin/env python3
# ===========================================================================
#  model_size_sweep.py  --  FAITH-SAE REAL RUN  ·  the MODEL-SIZE axis driver
#  --------------------------------------------------------------------------
#  The second axis of the study. The OOD sweep (ood_sweep.py) measures CFS vs
#  DOMAIN SHIFT for ONE backbone; this driver runs that whole pipeline for EACH
#  of the three supervised-ViT sizes (S/B/L) and stacks every per-rung CFS into
#  ONE combined CSV indexed by (model_size x dataset). The result is the 2-axis
#  table the paper plots: faithfulness x model-size x domain-shift.
#
#        model size  -->   ViT-S (~22M)   ViT-B (~86M)   ViT-L (~304M)
#        domain shift -->  in1k -> in100 -> food101 -> cifar100   (each size)
#
#  For each config it runs the full pipeline (extract -> train_sae -> manifold ->
#  concept_select -> probes -> ood_sweep), reads that run's ood_cfs_sweep.csv,
#  tags every row with the model_size + the backbone name + d_in, and appends to
#  outputs_model_sweep/cfs_model_size_sweep.csv.
#
#  TWO PATHS, ONE DRIVER (same pattern as every real_run module):
#    * REAL (default): needs timm + the datasets + a GPU. Each size runs to its
#      own cache_*/outputs_* dir (the configs already point there), so the three
#      runs never collide.
#    * --smoke: runs each size's pipeline on fabricated, real-SHAPED activations
#      on CPU (no timm, no downloads), proving the model-size axis wiring. It
#      shrinks the per-size config so the whole 3-size sweep finishes in seconds.
#
#  author: Rajia Rani
#  For research and educational purposes only.
# ===========================================================================
from __future__ import annotations

import argparse
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The three model-size configs (the model-size axis). Each is a self-contained
# real_run YAML with its own backbone.name / d_in / cache_dir / out_dir.
MODEL_SIZE_CONFIGS = [
    ("vit_s_20m", "configs/vit_s_20m.yaml"),    # ViT-S  ~22M  d=384
    ("vit_b_84m", "configs/vit_b_84m.yaml"),    # ViT-B  ~86M  d=768
    ("vit_l_307m", "configs/vit_l_307m.yaml"),  # ViT-L ~304M  d=1024
]


# --------------------------------------------------------------------------- #
#  Smoke config: a tiny per-size config (no YAML, no timm, no downloads). Each  #
#  size differs only in d_in (the width that distinguishes the backbones), so   #
#  the smoke sweep exercises the SAME code three times at three widths.         #
# --------------------------------------------------------------------------- #
def _smoke_size_cfg(size_name: str, d_in: int, out_dir, cache_dir) -> dict:
    return {
        "seed": 0,
        "backbone": {"framework": "timm", "name": f"smoke_{size_name}",
                     "pretrained": True, "layer": 2, "token_type": "patch",
                     "image_size": 224, "device": "cpu"},
        "data": {"imagenet_train_dir": "./_unused", "batch_size": 8,
                 "num_workers": 0, "max_images": 64,
                 "in100_classes_file": "./_unused", "data_dir": "./_unused"},
        "sae": {"d_in": d_in, "expansion": 8, "k": 8,
                "normalize": "unit_meansquare", "lr": 4.0e-4, "warmup": 5,
                "token_budget": 300_000, "batch_tokens": 512, "aux_k": 32,
                "dead_window": 50_000, "ckpt_every": 1_000_000},
        "steering": {"strength_grid": [0, 0.5, 1, 2, 4], "proj_rank_r": 16,
                     "bank_tokens": 4096},
        "cfs": {"n_probe_classes": 5, "bootstrap_n": 200, "select_thresh": 0.0,
                "max_act_top": 8},
        "ood": {"levels": ["in1k", "in100", "food101", "cifar100"],
                "usability_floor": 0.5},
        "paths": {"cache_dir": str(cache_dir), "out_dir": str(out_dir),
                  "sae_ckpt": str(pathlib.Path(out_dir) / "sae.safetensors"),
                  "manifold_basis": str(pathlib.Path(out_dir) / "U_r.npy")},
        "smoke": {"n_shards": 3, "tokens_per_shard": 8192, "n_patches": 64},
    }


# The three smoke widths mirror the real S/B/L widths (384/768/1024), so the
# combined CSV's model_size axis is meaningful even on the offline path.
_SMOKE_SIZES = [("vit_s_20m", 384), ("vit_b_84m", 768), ("vit_l_307m", 1024)]


# --------------------------------------------------------------------------- #
#  Run ONE size's full pipeline on the smoke path and return its sweep rows.    #
# --------------------------------------------------------------------------- #
def _run_one_smoke(size_name: str, d_in: int, root) -> "object":
    """Fabricate the cache, train the SAE, build U_r, and run the OOD sweep for a
    single model size on CPU, returning the sweep DataFrame tagged with the size.
    Mirrors smoke_real.py's stage order but only as far as the OOD sweep (the CFS
    vs (size x dataset) table is the model-size sweep's product)."""
    import extract_activations as extract
    import ood_sweep
    from data_real import _Cfg

    out_dir = pathlib.Path(root) / f"out_{size_name}"
    cache_dir = pathlib.Path(root) / f"cache_{size_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = _smoke_size_cfg(size_name, d_in, out_dir, cache_dir)
    dcfg = _Cfg(cfg)

    # Fabricate imagenet_train (SAE source) + every ladder rung.
    for ds in ["imagenet_train"] + list(cfg["ood"]["levels"]):
        extract._fabricate_smoke_shards(dcfg, ds, str(cache_dir))

    # The OOD sweep's smoke path fabricates its own real-shaped eval banks, trains
    # a tiny SAE internally, and produces the per-rung CFS table -- exactly the
    # product we want per model size.
    sweep_df = ood_sweep.run_ood_sweep(cfg, str(cache_dir), smoke=True)
    sweep_df = sweep_df.copy()
    sweep_df.insert(0, "model_size", size_name)
    sweep_df.insert(1, "backbone", cfg["backbone"]["name"])
    sweep_df.insert(2, "d_in", d_in)
    return sweep_df


# --------------------------------------------------------------------------- #
#  Run ONE size's full pipeline on the REAL path via run_all_real.sh.           #
# --------------------------------------------------------------------------- #
def _run_one_real(config_path: str, python_exe: str) -> "object":
    """REAL RUN: drive the full pipeline for one size via run_all_real.sh, then
    read that run's ood_cfs_sweep.csv and tag it with the size. Each config writes
    to its own cache_*/outputs_* dir, so the three runs are independent."""
    import subprocess

    import pandas as pd
    from data_real import load_real_config

    cfg = load_real_config(config_path)
    out_dir = pathlib.Path(cfg.paths.out_dir)
    d_in = int(cfg.sae.d_in)
    backbone = str(cfg.backbone.name)
    size_name = pathlib.Path(config_path).stem

    print(f"\n[model_size_sweep] REAL pipeline for {size_name} "
          f"(backbone={backbone}, d_in={d_in}) ...")
    # run_all_real.sh runs extract -> train -> manifold -> concept_select ->
    # ood_sweep -> ablations -> analysis for this config (the real GPU path).
    cmd = ["bash", str(_HERE / "run_all_real.sh"), "--real",
           "--config", config_path]
    env = {"PYTHON": python_exe}
    import os
    full_env = dict(os.environ); full_env.update(env)
    subprocess.run(cmd, check=True, cwd=str(_HERE), env=full_env)

    sweep_csv = out_dir / "ood_cfs_sweep.csv"
    if not sweep_csv.exists():
        raise FileNotFoundError(
            f"{sweep_csv} not found after running {size_name}; did run_all_real.sh "
            "complete the ood_sweep stage?")
    df = pd.read_csv(sweep_csv)
    df.insert(0, "model_size", size_name)
    df.insert(1, "backbone", backbone)
    df.insert(2, "d_in", d_in)
    return df


# --------------------------------------------------------------------------- #
#  The combined driver.                                                        #
# --------------------------------------------------------------------------- #
def run_model_size_sweep(smoke: bool = False, out_dir: str = None,
                         python_exe: str = None):
    """Run the pipeline for every model size and stack the per-rung CFS into one
    combined CSV (CFS vs model_size x dataset). Returns the combined DataFrame."""
    import pandas as pd

    out_dir = pathlib.Path(out_dir or (_HERE / "outputs_model_sweep"))
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = out_dir / "cfs_model_size_sweep.csv"

    print("=" * 74)
    print("FAITH-SAE — MODEL-SIZE SWEEP (axis 2: faithfulness x model size)")
    print("=" * 74)
    print(f"  sizes : {[s for s, _ in MODEL_SIZE_CONFIGS]}")
    print(f"  mode  : {'SMOKE (synthetic, CPU)' if smoke else 'REAL'}")

    frames = []
    if smoke:
        import tempfile
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="faith_modelsweep_"))
        try:
            for size_name, d_in in _SMOKE_SIZES:
                print(f"\n----- size {size_name} (d_in={d_in}) -----")
                frames.append(_run_one_smoke(size_name, d_in, tmp))
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        py = python_exe or sys.executable
        for _size_name, cfg_path in MODEL_SIZE_CONFIGS:
            frames.append(_run_one_real(str(_HERE / cfg_path), py))

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(combined_csv, index=False)
    print(f"\n  wrote {combined_csv}  ({len(combined)} rows: "
          f"{combined['model_size'].nunique()} sizes x "
          f"{combined['rung'].nunique()} rungs x "
          f"{combined['method'].nunique()} methods)")

    # A compact pivot (on-manifold CFS by size x rung) for a quick eyeball.
    try:
        onm = combined[combined["method"] == "onmanifold_steer"]
        pivot = onm.pivot_table(index="model_size", columns="rung",
                                values="cfs", aggfunc="mean")
        print("\n  on-manifold CFS by model_size x rung:")
        print(pivot.to_string())
    except Exception:
        pass
    return combined


def main() -> None:
    ap = argparse.ArgumentParser(
        description="FAITH-SAE model-size sweep: run the pipeline across the three "
                    "supervised-ViT sizes (S/B/L) and combine CFS vs (model_size x "
                    "dataset) into one CSV. Real path is the default; --smoke runs "
                    "the 3-size sweep on CPU with no timm/downloads.")
    ap.add_argument("--smoke", action="store_true",
                    help="offline CPU sweep across 3 widths (no timm, no downloads)")
    ap.add_argument("--out_dir", default=None,
                    help="where the combined CSV goes (default outputs_model_sweep/)")
    ap.add_argument("--python", default=None,
                    help="interpreter run_all_real.sh uses on the real path "
                         "(default: this interpreter)")
    args = ap.parse_args()
    run_model_size_sweep(smoke=args.smoke, out_dir=args.out_dir,
                         python_exe=args.python)


if __name__ == "__main__":
    main()
