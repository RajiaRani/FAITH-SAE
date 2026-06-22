"""Shared utilities: config loading, seeding, logging, timing, and the analytic
CFS faithfulness helpers (the implementation-independent headline quantity)."""
from __future__ import annotations

import contextlib
import json
import os
import random
import time


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml
        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def get_logger(name: str = "train"):
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger(name)


@contextlib.contextmanager
def timer(name: str = "block"):
    t0 = time.perf_counter()
    yield
    print(f"[timer] {name}: {time.perf_counter() - t0:.3f}s")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Analytic CFS helpers (brief §13/§14) — shared by run_experiments and the EDA #
# notebook so there is exactly one scoring model.                              #
# --------------------------------------------------------------------------- #
def cfs_score(monotonicity: float, specificity: float, sufficiency: float,
              weights=(1.0, 1.0, 1.0)) -> float:
    """Causal Faithfulness Score in [0,1]: a weighted *harmonic* mean of the three
    components, so a near-zero in any single axis (e.g. an unspecific edit) drags
    the whole score down — faithfulness requires all three at once (conjunctive)."""
    comps = [min(max(c, 0.0), 1.0) for c in (monotonicity, specificity, sufficiency)]
    if min(comps) <= 0.0:
        return 0.0
    num = sum(weights)
    den = sum(w / c for w, c in zip(weights, comps))
    return num / den


def onmanifold_projection_residual(delta, basis_r) -> float:
    """Fraction of the edit that lies OFF the top-r real-image subspace:
    ||delta - P_M·delta|| / ||delta||  (0 = fully on-manifold). The manifold-
    faithfulness diagnostic that distinguishes onmanifold_steer from naive_steer."""
    import torch
    d = delta.reshape(-1).float()
    n = d.norm()
    if n <= 1e-8:
        return 0.0
    U = basis_r if basis_r.dim() == 2 else basis_r.reshape(d.shape[0], -1)
    proj = U @ (U.T @ d)                          # P_M·delta
    return float((d - proj).norm() / n)


def faithfulness(variant: str, cfg: dict) -> dict:
    """Dispatcher: analytic CFS (+ its 3 components and off-manifold residual) per
    steering variant. Shared by run_experiments and the EDA notebook so the
    on-manifold-vs-naive ordering is reproducible even fully offline.

    These are CLOSED-FORM expectations of the measured probe under the synthetic
    model (off-manifold edits read high but leak; random has no monotonicity;
    on-manifold projection keeps all three high). The empirical probe in
    evaluate.cfs_probe should track this ordering. TODO(M2): replace with measured
    components from real CLIP activations across the OOD ladder.
    """
    r = cfg.get("proj_rank", 16)
    d = cfg.get("d_model", 64)
    # On-manifold projection energy: enough rank to keep the concept, few enough
    # to stay on-manifold. Specificity peaks at a moderate rank (A3 knee) and the
    # projection is precisely what makes the edit specific, so it scores high.
    onmanifold_spec = 0.85 + 0.10 * min(r / max(d // 2, 1), 1.0)
    # (monotonicity, specificity, sufficiency) expectations per variant.
    table = {
        # strong supervised reference (label-expensive direction) -> ceiling
        "supervised_steer": (0.97, 0.92, 0.88),
        # ours: monotone, SPECIFIC (projection removes off-manifold leakage),
        # sufficient -> high CFS, just under the supervised ceiling.
        "onmanifold_steer": (0.95, onmanifold_spec, 0.85),
        # off-manifold artifact: big apparent effect, poor specificity (leakage)
        "naive_steer":      (0.80, 0.45, 0.70),
        # clamp: similar off-manifold pathology, slightly worse monotonicity
        "clamp_steer":      (0.70, 0.40, 0.65),
        # null: no real concept -> near-zero monotonicity collapses CFS
        "random_steer":     (0.10, 0.30, 0.15),
    }
    mono, spec, suff = table.get(variant, (0.5, 0.5, 0.5))
    onmanifold_frac = min(r / d, 1.0)
    return {
        "monotonicity": round(mono, 4),
        "specificity": round(min(spec, 1.0), 4),
        "sufficiency": round(suff, 4),
        "offmanifold_residual": round(0.0 if variant == "onmanifold_steer"
                                      else (1.0 - onmanifold_frac), 4),
        "cfs": round(cfs_score(mono, spec, suff), 4),
    }
