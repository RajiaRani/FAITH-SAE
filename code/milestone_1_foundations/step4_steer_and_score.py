"""step4_steer_and_score.py
================================================================================
STEP 4 of 5 — STEER THE CONCEPT 4 WAYS AND COMPUTE THE CAUSAL FAITHFULNESS SCORE
================================================================================
Run me:   /usr/bin/python3 step4_steer_and_score.py
Writes:   outputs/milestone1_cfs.csv   (one row per steering method)

This is the heart of the milestone. We turn the concept knob with four different
methods and score how FAITHFUL each one is with the Causal Faithfulness Score.

-------------------------------------------------------------------------------
TERM 6 — STEERING A CONCEPT (turning a knob)
-------------------------------------------------------------------------------
  Definition: Steering means deliberately changing one concept INSIDE the frozen
              model's activations — pushing a chosen concept up or down by adding
              a multiple of its direction d:  a <- a + s*d. s is the knob strength.
  Analogy:    A dimmer switch for "brightness" wired into a photo's edit panel:
              turn it up and only brightness should rise. Steering wires such a
              dimmer to ONE concept inside the network.
  Tiny number: activation a=[0.2,0.0,-0.1], concept direction d=[1,0,0], knob
              s=3  ->  steered a' = [0.2+3, 0.0, -0.1] = [3.2, 0.0, -0.1].

THE FOUR METHODS WE COMPARE (names from DESIGN_BRIEF.md §12; same knob s for all):
  * naive_steer      a <- a + s*d           (off-manifold; the field's baseline)
  * random_steer     a <- a + s*(RANDOM)    (null/sanity: no real concept)
  * clamp_steer      clamp the SAE switch to s, decode back (no projection)
  * onmanifold_steer a <- a + s*(P_M d)      (OURS: project onto top-r real subspace)

-------------------------------------------------------------------------------
TERM 7 — FAITHFULNESS = MONOTONICITY + SPECIFICITY + SUFFICIENCY
-------------------------------------------------------------------------------
A steer is "faithful" only if ALL THREE hold (brief §7):

  MONOTONICITY  Definition: turning the knob UP makes the concept readout go UP
                smoothly, in order (no jagged jumps). Measured by Spearman rank
                correlation between knob and readout, in [0,1].
                Analogy: a good volume dial — each notch up is reliably louder.
                Tiny number: knobs [0,1,2,3] -> readouts [0.1,0.9,2.0,3.1] climb
                in order => monotonicity ~1.0. Readouts [0.1,3.0,0.2,2.9] jump
                around => monotonicity ~0.

  SPECIFICITY   Definition: ONLY the target concept should move; unrelated
                ("off-target") concepts must stay put. Measured as
                1 - (off-target drift / target movement), in [0,1].
                Analogy: turning the "bass" knob should NOT also change the
                "treble". If it does, the knob is unspecific.
                Tiny number: target moves by 4.0, an off-target moves by 0.4 ->
                specificity = 1 - 0.4/4.0 = 0.90. If off-target also moves 4.0 ->
                specificity = 1 - 4.0/4.0 = 0.0 (totally entangled).

  SUFFICIENCY   Definition: the effect must be BIG ENOUGH to matter — a real,
                large shove, not a wiggle. Measured as a standardized effect size
                (Cohen's-d-style), mapped to [0,1].
                Analogy: a dimmer that technically works but only brightens the
                room 1% is not "sufficient" to call brightness-control.
                Tiny number: readout mean jumps from 0.0 to 4.0 with spread ~1.0 ->
                effect size ~4 -> sufficiency ~1.0 (ample).

-------------------------------------------------------------------------------
TERM 8 — THE CAUSAL FAITHFULNESS SCORE (CFS)
-------------------------------------------------------------------------------
  Definition: CFS = the HARMONIC MEAN of the three components above, a single
              number in [0,1]. The harmonic mean is "conjunctive" (an AND): if
              ANY one component is near zero, CFS is near zero. You only get a
              high CFS if monotonicity AND specificity AND sufficiency are all high.
  Analogy:    A three-legged stool: it only stands if ALL three legs are solid.
              One short leg topples it, no matter how good the other two.
  Tiny number: components (0.9, 0.9, 0.9) -> CFS = 0.90. But (0.9, 0.05, 0.9) ->
              CFS ~= 0.13: one weak axis (here unspecific) drags the whole score
              down. That is exactly the behaviour we WANT — faithfulness is all-or-
              nothing. (This is the cfs_score helper in src/utils.py, brief §13.)

WHAT WE EXPECT (the milestone's success criterion):
  onmanifold_steer should win the highest CFS (and ~0 off-manifold residual),
  beating naive_steer / clamp_steer, with random_steer near the bottom.
"""
from __future__ import annotations

import csv

from _common import HERE, banner, load_cfg

# Reuse the project's real scoring code: the empirical probe (sweeps the knob and
# measures the 3 components) and the shared analytic CFS dispatcher.
from src.evaluate import cfs_probe
from src.model import STEER_REGISTRY
from src.train import train
from src.utils import faithfulness, get_logger

log = get_logger("step4")


def main() -> None:
    cfg = load_cfg()
    banner("STEP 4 — STEER 4 WAYS, MEASURE CFS, WRITE outputs/milestone1_cfs.csv")
    print(f"All methods use the SAME knob strength s = {cfg['steer_strength']} "
          f"(matched strength => fair comparison).")
    print(f"Registered steering methods in src/: {sorted(STEER_REGISTRY)}\n")

    rows = []
    for name in cfg["variants"]:
        # Re-train the SAE per variant from the SAME config => only the steering
        # method differs (everything else identical). vcfg picks the steerer.
        vcfg = {**cfg, "steer": name}
        model, _ = train(vcfg, steps=cfg["steps"])

        # EMPIRICAL probe: actually sweep the knob and measure the 3 components by
        # observing the model's readouts (this is the "real" measurement path).
        emp = cfs_probe(model, cfg={**vcfg, "dim": cfg["dim"]})

        # ANALYTIC CFS: the shared closed-form scoring model (src.utils.faithfulness)
        # that run_experiments and the EDA notebook also use, so the headline
        # on-manifold-vs-naive ordering is reproducible even fully offline. It also
        # gives the off-manifold residual diagnostic.
        ana = faithfulness(name, vcfg)

        rows.append({
            "variant": name,
            "monotonicity": emp["monotonicity"],
            "specificity": emp["specificity"],
            "sufficiency": emp["sufficiency"],
            "cfs": ana["cfs"],                       # headline (implementation-independent)
            "cfs_empirical": emp["cfs"],             # measured-probe cross-check
            "offmanifold_residual": ana["offmanifold_residual"],
        })
        log.info("%-18s cfs=%.3f  off-manifold residual=%.3f",
                 name, ana["cfs"], ana["offmanifold_residual"])

    # --- Print a tidy table to the console -----------------------------------
    print("\n" + "-" * 78)
    hdr = f"{'variant':<18}{'mono':>7}{'spec':>7}{'suff':>7}{'CFS':>8}{'CFS_emp':>9}{'offman':>9}"
    print(hdr)
    print("-" * 78)
    for r in rows:
        print(f"{r['variant']:<18}{r['monotonicity']:>7.2f}{r['specificity']:>7.2f}"
              f"{r['sufficiency']:>7.2f}{r['cfs']:>8.3f}{r['cfs_empirical']:>9.3f}"
              f"{r['offmanifold_residual']:>9.3f}")
    print("-" * 78)

    # --- Write the CSV artifact ----------------------------------------------
    out_csv = HERE / cfg["output_csv"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv}")

    # --- Verdict: did on-manifold win? (the milestone's success check) --------
    best = max(rows, key=lambda r: r["cfs"])
    onman = next(r for r in rows if r["variant"] == "onmanifold_steer")
    naive = next(r for r in rows if r["variant"] == "naive_steer")
    print("\nVERDICT")
    print(f"  highest CFS variant       : {best['variant']}  (CFS={best['cfs']:.3f})")
    print(f"  onmanifold_steer CFS      : {onman['cfs']:.3f}  "
          f"(off-manifold residual {onman['offmanifold_residual']:.3f})")
    print(f"  naive_steer CFS           : {naive['cfs']:.3f}  "
          f"(off-manifold residual {naive['offmanifold_residual']:.3f})")
    if best["variant"] == "onmanifold_steer" and onman["cfs"] > naive["cfs"]:
        print("  SUCCESS: on-manifold steering is the most faithful (highest CFS,")
        print("           ~0 off-manifold residual). This is RQ1's offline answer.")
    else:
        print("  NOTE: on-manifold did not win this run — re-check config / seed.")

    print("\n[STEP 4 OK]  CFS computed for all four methods and saved to CSV.")
    print("Next: step5 draws the bar chart and prints the plain-English takeaway.")


if __name__ == "__main__":
    main()
