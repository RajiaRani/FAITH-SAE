"""step4_score_cfs.py — turn residuals into CFS and write method_compare.csv.

==============================================================================
WHAT THIS STEP DOES (in one sentence)
==============================================================================
For each steering method it computes the Causal Faithfulness Score (CFS) and
joins it with the MEASURED off-manifold residual from step3, producing the
headline table outputs/method_compare.csv (variant, off-manifold residual, CFS).

==============================================================================
TEACH-FROM-ZERO: the CFS metric (the project's measuring stick)
==============================================================================

WHY A STEER NEEDS A SCORE
  A steering method with NO metric is unfalsifiable: a big, plausible-looking
  change in the readout could be a REAL causal effect OR an off-manifold mirage.
  CFS makes "is this edit faithful?" a number you can compare.

THE THREE INGREDIENTS OF FAITHFULNESS (each in [0,1])
  * Monotonicity = does turning the knob UP move the concept readout UP, smoothly
    and in order? (measured as a rank correlation between knob and readout.)
    Tiny example: knob 0,1,2,3 -> readout 0.1,0.4,0.7,0.9 is monotone (~1.0);
    readout 0.1,0.9,0.2,0.8 is jagged (~0.0).
  * Specificity = does ONLY the target concept move, while OFF-target probes stay
    put? (1 - off-target drift.) Off-manifold edits smear into other concepts ->
    low specificity.
  * Sufficiency = is the effect BIG enough to match the concept's claimed meaning?
    (a standardized effect size.) A real concept knob should produce an ample,
    not a token, change.

CFS = HARMONIC MEAN of the three  (DESIGN_BRIEF §13/§14)
  The harmonic mean is "conjunctive": if ANY one ingredient is near zero, CFS is
  near zero — faithfulness requires ALL THREE at once. (A plain average would let
  a high score in one axis hide a failure in another.)
  Tiny example: HM(0.9, 0.9, 0.9) = 0.90; but HM(0.9, 0.9, 0.05) = 0.13 — one
  weak axis tanks the whole score. We REUSE the project's cfs_score() for this.

HOW THE METHODS SCORE (and WHY on-manifold wins)
  * onmanifold_steer: the projection removes off-sheet leakage, so the edit is
    SPECIFIC; it is monotone and sufficient too -> high CFS, residual ~ 0.
  * naive_steer / clamp_steer: big apparent effect but the off-sheet part leaks
    into other concepts -> low specificity -> low CFS, residual large.
  * random_steer: no real concept -> ~zero monotonicity -> CFS collapses.
  We obtain the (monotonicity, specificity, sufficiency, CFS) per variant from
  the project's SHARED scoring dispatcher src.utils.faithfulness(...), so the
  EDA notebook and this milestone agree exactly. We then OVERWRITE the residual
  column with the value we actually MEASURED in step3.

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step4_score_cfs.py
Reads outputs/residuals.csv. Writes outputs/method_compare.csv.
"""
from __future__ import annotations

import csv

from _common import banner, load_cfg, outpath


def read_measured_residuals() -> dict:
    """variant -> measured off-manifold residual (from step3)."""
    out = {}
    with open(outpath("residuals.csv"), newline="") as f:
        for row in csv.DictReader(f):
            out[row["variant"]] = float(row["offmanifold_residual"])
    return out


def main() -> None:
    cfg = load_cfg()
    banner("STEP 4 — score CFS per method and assemble method_compare.csv")

    from src.utils import faithfulness    # the project's shared scoring dispatcher

    measured = read_measured_residuals()
    print(f"  measured off-manifold residuals from step3: {measured}\n")

    rows = []
    print(f"  {'variant':<18} {'mono':>6} {'spec':>6} {'suff':>6} "
          f"{'off-resid':>10} {'CFS':>7}")
    print("  " + "-" * 62)
    for variant in cfg["variants"]:
        f = faithfulness(variant, cfg)        # mono/spec/suff/cfs (+ analytic resid)
        resid = measured.get(variant, f["offmanifold_residual"])  # prefer MEASURED
        row = {
            "variant": variant,
            "monotonicity": f["monotonicity"],
            "specificity": f["specificity"],
            "sufficiency": f["sufficiency"],
            "offmanifold_residual": round(resid, 4),
            "cfs": f["cfs"],
        }
        print(f"  {variant:<18} {row['monotonicity']:>6.3f} {row['specificity']:>6.3f} "
              f"{row['sufficiency']:>6.3f} {row['offmanifold_residual']:>10.4f} "
              f"{row['cfs']:>7.4f}")
        rows.append(row)

    out = outpath("method_compare.csv")
    fields = ["variant", "monotonicity", "specificity", "sufficiency",
              "offmanifold_residual", "cfs"]
    with open(out, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  saved -> {out}")

    onm = next(r for r in rows if r["variant"] == "onmanifold_steer")
    nv = next(r for r in rows if r["variant"] == "naive_steer")
    cfs_ok = onm["cfs"] > nv["cfs"]
    resid_ok = onm["offmanifold_residual"] < nv["offmanifold_residual"]
    print(f"\n  SUCCESS CRITERION:")
    print(f"    on-manifold CFS {onm['cfs']:.4f}  >  naive CFS {nv['cfs']:.4f}  "
          f"-> {'PASS' if cfs_ok else 'FAIL'}")
    print(f"    on-manifold residual {onm['offmanifold_residual']:.4f} (~0)  <  "
          f"naive residual {nv['offmanifold_residual']:.4f}  "
          f"-> {'PASS' if resid_ok else 'FAIL'}")
    print("\nSTEP 4 done. Next: step5 draws the figure.")


# REAL RUN (M4): swap the shared analytic dispatcher for the EMPIRICAL probe
# (src.evaluate.cfs_probe) measured on real CLIP activations — sweep the knob,
# measure monotonicity/specificity/sufficiency directly, combine with cfs_score —
# and report CFS at every OOD shift level. The MEASURED off-manifold residual
# from step3 carries over unchanged.
if __name__ == "__main__":
    main()
