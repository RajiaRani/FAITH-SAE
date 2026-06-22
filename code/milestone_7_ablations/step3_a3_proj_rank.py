"""step3_a3_proj_rank.py — ABLATION A3: on-manifold projection rank r (THE CORE KNOB).

==============================================================================
WHAT THIS ABLATION DOES (in one sentence)
==============================================================================
It turns exactly ONE knob — the on-manifold projection rank `r` (how many
real-image directions the projection keeps) — across a sweep, holds every other
dial fixed (the SAE is trained ONCE and reused), rebuilds the projection U_r for
each r, and MEASURES how CFS responds — locating the faithfulness "knee".

==============================================================================
TEACH-FROM-ZERO: the A3 knob — the projection rank r
==============================================================================
ON-MANIFOLD STEERING (one-paragraph recap; milestone 4 teaches it from zero)
  Real-image activations live on a thin SHEET inside the big 64-D space. To steer
  a concept up we add a direction; on-manifold steering first PROJECTS that edit
  onto the sheet (keeps only the part the model actually uses on real images),
  so the steered activation stays realistic. The projection keeps the top-r sheet
  directions: P_M = U_r U_r^T, where U_r holds r directions found by PCA.

r (THE PROJECTION RANK)
  r = how many of those sheet directions we keep. It is the dimension of the
  subspace the edit is allowed to live in.
  Analogy: a stencil with r holes. The edit can only "paint" through the holes.
  Too few holes (small r) and you can't draw the concept at all; the right number
  and the concept comes through cleanly; way too many holes (r -> 64) and the
  stencil is gone — you're back to painting anywhere (naive steering).
  Tiny number: the true sheet here is 8-D. r=2 covers only a sliver of it; r=8
  covers it exactly; r=64 covers the whole space (no constraint = naive).

OVER- vs UNDER-CONSTRAINED (the two failure modes A3 reveals)
  * r TOO SMALL (over-constrained): the projection throws away real sheet
    directions the concept needs. The edit can barely move the concept -> the
    EFFECT DIES -> sufficiency/monotonicity drop -> CFS low.
  * r ABOUT RIGHT (~ the true sheet rank, 8): the projection keeps the whole
    sheet and nothing more. The edit moves the concept (monotone, sufficient)
    WITHOUT smearing off-sheet (specific) -> CFS peaks.
  * r TOO BIG (-> 64, under-constrained): P_M -> identity, the projection does
    nothing, on-manifold DEGENERATES INTO NAIVE -> the edit drifts off-manifold,
    specificity leaks -> CFS sags back toward the naive level.
  This is the design brief's CORE ablation (A3): "low r over-constrains (effect
  dies), high r lets the edit drift off-manifold; locate the CFS knee."

THE KNEE / SWEET SPOT (how to read this curve)
  Plot CFS vs r. It rises from a starved low-r value, PEAKS near the true sheet
  rank, then declines toward the naive level as r -> 64. The PEAK is the sweet
  spot; the elbow where it stops improving is the KNEE. Pick r at the peak.
  Tiny number: CFS by r might read 0.30 (r=1), 0.55 (r=2), 0.78 (r=4),
  0.86 (r=8), 0.80 (r=16), 0.70 (r=32), 0.61 (r=64=naive) -> peak at r=8.

THE DIAGNOSTIC WE REPORT ALONGSIDE CFS
  Off-manifold residual of the effective edit = ||edit - P_M edit|| / ||edit||
  (0 = fully on the sheet, 1 = fully off). It RISES as r shrinks below the sheet
  rank (the edit can't even reach the sheet's used part) AND the naive baseline's
  residual stays high throughout — the on-manifold residual is ~0 in the sweet
  spot. This is the manifold-faithfulness diagnostic from src.utils.

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step3_a3_proj_rank.py
Appends rows to outputs/ablations.csv (ablation_id=A3) — one row per (r, steerer).
"""
from __future__ import annotations

from _common import (banner, build_a3_bank, estimate_U_r, load_cfg,
                     measure_cfs, train_probes, train_sae_decoder)


def run(cfg: dict) -> list:
    rows = []
    # A3 uses a PURPOSE-BUILT bank (see _common.build_a3_bank) whose target concept
    # is SPREAD across the whole sheet (so low r truncates it) and carries an
    # OFF-sheet leak (so high r re-admits it). edit_dir is the raw steering edit;
    # U_ref is the FIXED true-sheet basis used only for the residual diagnostic.
    acts, labels, dirs, edit_dir, U_ref = build_a3_bank(cfg)
    W, b, accs = train_probes(acts, labels, seed=int(cfg["seed"]))
    print(f"  probe held-out accuracies = {[round(a, 3) for a in accs]}")

    # Train the SAE ONCE at the baseline; A3 changes ONLY the projection rank r,
    # not the SAE. (Holding the SAE fixed isolates the projection-rank effect.)
    dec, recon_mse, model = train_sae_decoder({}, acts, cfg)
    print(f"  trained baseline SAE once (recon MSE = {recon_mse:.4f}); "
          f"now sweeping the projection rank r only.\n")
    dim = int(cfg["dim"])

    print(f"  {'r':>4} {'variant':<18} {'off_resid':>10} "
          f"{'mono':>6} {'spec':>6} {'suff':>6} {'CFS':>7}")
    print("  " + "-" * 66)
    for r in cfg["a3_proj_ranks"]:
        r = min(int(r), dim)
        # === THE ONE KNOB WE TURN: the projection rank r ==========================
        # Rebuild the on-manifold subspace U_r with exactly r directions, and tell
        # the model's onmanifold_steer to use this r (its cfg drives the steerer).
        # naive_steer ignores r (no projection) -> its row is the constant
        # off-manifold reference the curve declines toward as r -> dim.
        U_r = estimate_U_r(acts, r)
        model.cfg["proj_rank"] = r          # onmanifold_steer reads proj_rank
        for variant in cfg["ablation_variants"]:
            U_use = U_r if variant == "onmanifold_steer" else estimate_U_r(acts, dim)
            m = measure_cfs(variant, cfg, acts, dirs, dec, W, b, U_use, model,
                            target_concept=int(cfg["target_concept"]),
                            edit_dir_override=edit_dir, resid_basis=U_ref)
            print(f"  {r:>4} {variant:<18} {m['offmanifold_residual']:>10.4f} "
                  f"{m['monotonicity']:>6.3f} {m['specificity']:>6.3f} "
                  f"{m['sufficiency']:>6.3f} {m['cfs']:>7.4f}")
            rows.append({
                "ablation_id": "A3",
                "knob_value": r,
                "variant": variant,
                "cfs": m["cfs"],
                "diagnostic": m["offmanifold_residual"],
                "diagnostic_name": "offmanifold_residual",
                "monotonicity": m["monotonicity"],
                "specificity": m["specificity"],
                "sufficiency": m["sufficiency"],
                "offmanifold_residual": m["offmanifold_residual"],
            })
    return rows


def main() -> list:
    cfg = load_cfg()
    banner("ABLATION A3 — projection rank r (THE CORE KNOB): find the CFS knee")
    rows = run(cfg)
    onm = [r for r in rows if r["variant"] == "onmanifold_steer"]
    best = max(onm, key=lambda r: r["cfs"])
    print(f"\n  A3 takeaway: on-manifold CFS peaks at r = {best['knob_value']} "
          f"(CFS = {best['cfs']:.3f}); the true sheet rank is {cfg['true_manifold_rank']}.")
    print("    r too small => effect dies; r => 64 degenerates to naive "
          "(P_M -> I).")
    return rows


# REAL RUN (M7): U_r comes from PCA of a LARGE real CLIP activation bank; sweep r
# and report CFS + off-manifold residual at each r. This is the offline analog of
# the paper's core A3 x steering-strength design grid (fig3 / fig5).
if __name__ == "__main__":
    main()
