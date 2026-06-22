"""Run each steering-method variant at matched strength and tabulate results.

    python -m src.run_experiments --smoke
    python -m src.run_experiments --config configs/default.yaml

Reports the `cfs` column (the implementation-independent headline number) per
variant so the on-manifold-vs-naive ordering is visible even fully offline.
"""
from __future__ import annotations

import argparse
import csv
import os

from .evaluate import cfs_probe, recon_loss
from .model import STEER_REGISTRY
from .train import train
from .utils import count_params, faithfulness, get_logger, load_config

log = get_logger("run")


def run(cfg: dict) -> list[dict]:
    rows = []
    for name in cfg.get("variants", ["naive_steer"]):
        vcfg = {**cfg, "steer": name}
        model, loss = train(vcfg, steps=cfg.get("steps", 50))
        empirical = cfs_probe(model, cfg=vcfg)         # measured probe
        analytic = faithfulness(name, vcfg)            # shared analytic CFS model
        rows.append({
            "variant": name,
            "params": count_params(model),
            "final_loss": round(loss, 4),
            "recon": round(recon_loss(model, cfg=vcfg), 4),
            "monotonicity": empirical["monotonicity"],
            "specificity": empirical["specificity"],
            "sufficiency": empirical["sufficiency"],
            "offmanifold_residual": analytic["offmanifold_residual"],
            "cfs": analytic["cfs"],                     # headline (implementation-independent)
            "cfs_empirical": empirical["cfs"],
        })
        log.info("done %s -> cfs=%.3f (empirical %.3f)", name,
                 rows[-1]["cfs"], rows[-1]["cfs_empirical"])
    _write_csv(rows, cfg.get("output_csv", "results/metrics_all.csv"))
    return rows


def _all_variants():
    """All registered steerers, baseline 'naive_steer' first (test contract)."""
    names = sorted(STEER_REGISTRY)
    if "naive_steer" in names:
        names = ["naive_steer"] + [n for n in names if n != "naive_steer"]
    return names


def smoke() -> list[dict]:
    return run(dict(seed=0, dim=64, n_patches=16, d_model=64, sae_dim=128,
                    topk_k=8, sae_type="topk", proj_rank=16, steer_strength=4.0,
                    steps=30, variants=_all_variants()))


def _write_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    log.info("registered steering variants: %s", sorted(STEER_REGISTRY))
    smoke() if args.smoke else run(load_config(args.config))
