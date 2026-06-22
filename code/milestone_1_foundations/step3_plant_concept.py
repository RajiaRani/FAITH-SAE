"""step3_plant_concept.py
================================================================================
STEP 3 of 5 — PLANT A KNOWN CONCEPT  +  MEET THE DATA MANIFOLD
================================================================================
Run me:   /usr/bin/python3 step3_plant_concept.py

WHAT YOU LEARN HERE
-------------------
Why we plant a KNOWN concept (so we have a ground truth to score against), and
what the "data manifold" is — the single most important idea behind on-manifold
steering.

-------------------------------------------------------------------------------
WHY PLANT A KNOWN CONCEPT?
-------------------------------------------------------------------------------
In a real model we do not know which direction is "stripes" vs "fur" — so we
cannot tell whether a steer truly moved the right concept. The fix: in the
synthetic world we INJECT one concept ourselves, at a known direction and
magnitude. Now we have GROUND TRUTH. When we later steer and measure, we can
check the readout actually tracks the concept we planted.
  Analogy:    A doctor testing a thermometer puts it in water of a KNOWN
              temperature. Only because the truth is known can they grade the
              thermometer. The planted concept is our "known-temperature water".
  Tiny number: planted direction d = [0.5, 0.5, 0.5, 0.5] added with amplitude
              a=2.0 to an activation [0.1, 0.0, -0.1, 0.2] gives
              [1.1, 1.0, 0.9, 1.2]. The "+1.0 shove along d" is the planted signal.

-------------------------------------------------------------------------------
TERM 5 — THE DATA MANIFOLD, and ON- vs OFF-MANIFOLD EDITS
-------------------------------------------------------------------------------
  Definition: The data manifold is the (curved, lower-dimensional) region of
              activation space where REAL images actually land. Activation space
              is huge, but real-image activations cluster on a thin sheet inside
              it; that sheet is the manifold. An ON-manifold edit keeps you on the
              sheet (still looks like a real image to the model); an OFF-manifold
              edit shoves you off the sheet into "nonsense" the model never saw.
  Analogy:    Roads on a map. Towns (real activations) sit along a network of
              roads (the manifold). Driving town-to-town ON the roads = on-manifold
              and sensible. Driving straight through a lake/mountain because it is
              "shorter" = off-manifold: technically a direction, but no real trip
              looks like that.
  Tiny number: Suppose real activations only ever vary along axis-1 and axis-2
              (the manifold = the (x,y) plane); axis-3 is always ~0. An edit that
              moves you to (1.0, 0.5, 0.0) stays ON the plane. An edit to
              (1.0, 0.5, 9.9) jumps OFF it (axis-3=9.9 never happens in real data).
  WHY IT MATTERS: The whole paper's claim is that constraining a steer to the
              manifold ("on-manifold steering", method `onmanifold_steer`) makes
              the edit faithful, while naive steering ignores the manifold and so
              produces off-manifold artifacts. We make this measurable with the
              "off-manifold residual" = the fraction of the edit that left the
              sheet (0.0 = fully on-manifold).
"""
from __future__ import annotations

from _common import banner, load_cfg


def main() -> None:
    import torch
    cfg = load_cfg()
    from src.data import concept_readout, planted_concept, synthetic_batch
    from src.utils import onmanifold_projection_residual, set_seed

    set_seed(cfg["seed"])
    dim = cfg["dim"]
    banner("STEP 3 — PLANT A KNOWN CONCEPT AND MEASURE ON/OFF-MANIFOLD")

    # --- 1. The planted concept direction (our ground truth). Same one src.data
    # uses everywhere, so the readout below is consistent with the rest of src/.
    d = planted_concept(dim, seed=0)
    print(f"Planted concept direction (first 6 of {dim}): "
          f"[{', '.join(f'{v:+.2f}' for v in d[:6].tolist())}, ...]")
    print(f"Its length (norm) = {float(d.norm()):.3f}  (unit vector by construction)")

    # --- 2. Build a batch where the concept is planted at a KNOWN amplitude, then
    # read it out with the held-out probe. A bigger plant -> bigger readout: this
    # is the ground-truth relationship every steering method will be graded on.
    print("\nPlant amplitude  ->  mean concept readout  (probe = <activation, d>):")
    for amp in (0.0, 1.0, 2.0, 4.0):
        # concept_strength scales how hard the concept is planted into the batch.
        _, a_target = synthetic_batch(batch=64, n_patches=cfg["n_patches"], dim=dim,
                                      seed=7, concept_strength=amp)
        read = float(concept_readout(a_target, dim).mean())
        print(f"   plant={amp:<4}  ->  readout={read:+.3f}")
    print("Readout rises with the plant: that monotone link is the GROUND TRUTH")
    print("a faithful steer must reproduce (see step4's monotonicity component).")

    # --- 3. Demonstrate ON vs OFF manifold with the project's own residual helper.
    # Estimate the real-image manifold = top-r principal directions of real
    # activations, then measure how much of an edit lands OFF that subspace.
    _, a_real = synthetic_batch(batch=128, n_patches=cfg["n_patches"], dim=dim, seed=99)
    flat = a_real.reshape(-1, dim)
    flat = flat - flat.mean(0, keepdim=True)
    # Right singular vectors = the activation principal axes (same math the real
    # onmanifold_steer uses to build P_M = U_r U_r^T).
    _, _, Vh = torch.linalg.svd(flat, full_matrices=False)
    r = min(cfg["proj_rank"], Vh.shape[0])
    U_r = Vh[:r].T                                  # [dim, r] = the manifold basis

    # Edit A: a direction LIVING in the manifold (a mix of the top axes) -> on-manifold.
    on_edit = U_r @ torch.randn(r)
    # Edit B: a direction OUTSIDE the manifold (a tail/least-used axis) -> off-manifold.
    off_edit = Vh[-1]

    res_on = onmanifold_projection_residual(on_edit, U_r)
    res_off = onmanifold_projection_residual(off_edit, U_r)
    print(f"\nManifold rank r = {r}  (top-{r} real-image directions kept)")
    print(f"Off-manifold residual of an ON-manifold edit  : {res_on:.3f}  "
          f"(~0.0 => stays on the sheet)")
    print(f"Off-manifold residual of an OFF-manifold edit : {res_off:.3f}  "
          f"(~1.0 => left the sheet)")
    print("\nThis residual is the key diagnostic: onmanifold_steer keeps it ~0;")
    print("naive_steer ignores the manifold and lets it grow.")

    print("\n[STEP 3 OK]  You have a known planted concept (ground truth) and you")
    print("can measure how far any edit drifts off the data manifold.")
    print("Next: step4 STEERS the concept four ways and scores each with CFS.")


if __name__ == "__main__":
    main()
