"""step3_steer_and_residual.py — run the 4 steerers, measure off-manifold residual.

==============================================================================
WHAT THIS STEP DOES (in one sentence)
==============================================================================
It builds ONE raw concept edit Delta, runs each of the four steering methods on
it (passing the FIXED real-image subspace U_r from step2), and measures the
OFF-MANIFOLD RESIDUAL of the actual edit each method applies — naive flies off
the sheet, on-manifold stays on it.

==============================================================================
TEACH-FROM-ZERO: the terms this step adds
==============================================================================

THE RAW EDIT  Delta
  To "steer a concept up", we add a direction to the activation. That direction
  is Delta. Here Delta = the SAE concept direction d (the decoder column for the
  feature we picked). Naive steering adds the WHOLE Delta. On-manifold steering
  adds only the part of Delta that lies on the sheet: P_M @ Delta.

THE FOUR STEERERS (the pluggable component; names from DESIGN_BRIEF §12)
  * naive_steer      a' = a + s * d            (adds the whole raw edit; M3 baseline)
  * random_steer     a' = a + s * r_rand       (same form, a RANDOM direction; null)
  * clamp_steer      clamp the SAE feature to magnitude s, decode back (no projection)
  * onmanifold_steer a' = a + s * (P_M * d)    (OURS: project Delta onto the sheet)
  We DO NOT re-implement these — we pull them from the project's STEER_REGISTRY
  (src/blocks/__init__.py). That is the whole point of reusing src/.

THE FIXED real-image BASIS
  Every steerer's `steer(...)` accepts a `basis` argument. We pass basis = U_r
  (estimated ONCE in step2 from the real-image bank). This matters:
  on-manifold steering must project onto the subspace the FROZEN MODEL uses on
  REAL images, not onto the current batch. naive/random/clamp ignore `basis`
  (by design they do no projection) — that is exactly why they go off-manifold.

OFF-MANIFOLD RESIDUAL  (the diagnostic, reused from src.utils)
    onmanifold_projection_residual(edit, U_r)
       = || edit - P_M*edit ||  /  || edit ||      (0 = fully on-manifold, 1 = fully off)
  We measure the residual of the EFFECTIVE edit each method applies (a' - a),
  i.e. how much of what it actually did leaves the sheet.
  * naive/random: their effective edit is the raw (mostly off-sheet) direction
    -> residual is LARGE (the edit pushes the activation off the manifold).
  * on-manifold: its effective edit is P_M*Delta, already on the sheet
    -> residual ~ 0 (P_M*(P_M*Delta) = P_M*Delta; nothing left off-sheet).
  Tiny number (2-D sheet=x-axis): raw edit (0.6, 0.8). naive applies all of it
  -> residual = 0.8/1.0 = 0.8 (mostly off-sheet). on-manifold applies (0.6, 0)
  -> residual = 0.0.

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step3_steer_and_residual.py
Reads outputs/U_r.npy. Writes outputs/residuals.csv (variant, offmanifold_residual).
"""
from __future__ import annotations

import csv

import numpy as np

from _common import banner, load_cfg, outpath


def build_model_and_edit(cfg: dict):
    """Make the toy FaithSAE model and grab one raw concept edit Delta = d.

    We train the SAE a few steps so its decoder columns are meaningful concept
    directions, then take column `concept` as the raw edit direction.
    """
    import torch

    from src.train import train  # the project's tiny SAE trainer
    from src.utils import set_seed

    set_seed(int(cfg["seed"]))
    model, _ = train(cfg, steps=int(cfg["steps"]))
    concept = 0                                   # the SAE feature we steer
    with torch.no_grad():
        d = model.sae.concept_direction(concept)  # raw edit Delta (a [dim] vector)
        d = d / (d.norm() + 1e-8)                  # unit length (clean strength units)
    return model, concept, d


def effective_edit(model, concept, d, strength, variant, basis, x):
    """Apply one steerer and return the EFFECTIVE edit it made: (a' - a).

    We build the named steerer from the registry, run it with the FIXED real-
    image basis U_r, and subtract the un-steered activation. The result is the
    actual direction that method pushed the activation along.
    """
    import torch

    from src.model import build_steer

    steer = build_steer(variant, model.cfg)
    with torch.no_grad():
        a = model.activations(x)                  # un-steered activations
        a_steered = steer(a, d, strength, sae=model.sae, concept=concept, basis=basis)
        edit = (a_steered - a)                     # [B, n_patches, dim]
    # average over batch+patches -> one representative edit direction
    return edit.reshape(-1, edit.shape[-1]).mean(0)


def main() -> None:
    cfg = load_cfg()
    s = float(cfg["steer_strength"])
    banner("STEP 3 — steer with each method, measure the off-manifold residual")

    import torch

    from src.data import synthetic_batch
    from src.utils import onmanifold_projection_residual

    # The FIXED real-image subspace from step2 (the model's sheet on real images).
    U_r = torch.from_numpy(np.load(outpath("U_r.npy"))).float()    # [dim, r]
    print(f"  loaded U_r {tuple(U_r.shape)} (the fixed real-image sheet basis from step2)")

    model, concept, d = build_model_and_edit(cfg)
    print(f"  raw concept edit Delta = SAE concept_direction(0), unit length")

    # An input batch to steer (the synthetic 'image' inputs).
    x, _ = synthetic_batch(16, int(cfg["n_patches"]), int(cfg["dim"]), seed=9000)

    rows = []
    print(f"\n  matched steering strength s = {s} for ALL methods\n")
    print(f"  {'variant':<18} {'off-manifold residual':>22}   meaning")
    print("  " + "-" * 70)
    for variant in cfg["variants"]:
        edit = effective_edit(model, concept, d, s, variant, U_r, x)
        # Residual of the EFFECTIVE edit against the real-image subspace U_r.
        resid = onmanifold_projection_residual(edit, U_r)
        if variant == "onmanifold_steer":
            note = "edit stays ON the sheet (realistic)"
        elif variant == "random_steer":
            note = "random dir -> almost entirely OFF the sheet"
        else:
            note = "edit flies OFF the sheet (off-manifold)"
        print(f"  {variant:<18} {resid:>22.4f}   {note}")
        rows.append({"variant": variant, "offmanifold_residual": round(resid, 4)})

    out = outpath("residuals.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "offmanifold_residual"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  saved -> {out}")

    onm = next(r for r in rows if r["variant"] == "onmanifold_steer")["offmanifold_residual"]
    nv = next(r for r in rows if r["variant"] == "naive_steer")["offmanifold_residual"]
    print(f"\n  CHECK: on-manifold residual {onm:.4f}  <  naive residual {nv:.4f}  "
          f"-> {'PASS' if onm < nv else 'FAIL'}")
    print("\nSTEP 3 done. Next: step4 turns these into CFS and the final table.")


# REAL RUN (M4): U_r comes from the large real CLIP bank (step2). The raw edit
# Delta is a real SAE concept direction; steer real CLIP ViT-B/16 patch
# activations and measure the residual against the same fixed U_r at each OOD
# shift level. Everything else is unchanged.
if __name__ == "__main__":
    main()
