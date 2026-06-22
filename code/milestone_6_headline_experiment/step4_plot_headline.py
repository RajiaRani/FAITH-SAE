"""step4_plot_headline.py — render the HEADLINE figure outputs/fig1_cfs_ood_sweep.png.

==============================================================================
WHAT THIS STEP DOES (in one sentence)
==============================================================================
It reads outputs/ood_cfs_sweep.csv and draws the paper's headline picture: the
Causal Faithfulness Score (CFS) on the y-axis against the OOD shift severity on
the x-axis (clean -> ImageNet-R -> Sketch -> ImageNet-C 1..5 -> ObjectNet), one
line per steering method, each with its bootstrap CONFIDENCE BAND, the usability
FLOOR drawn as a dashed line, and each method's COLLAPSE KNEE marked. The SHAPE
of these two curves is the project's answer to RQ3.

==============================================================================
HOW TO READ A DEGRADATION CURVE (the one-glance takeaway), FROM ZERO
==============================================================================
A "degradation curve" plots a quality number (here CFS) as conditions get HARDER
(here further out of distribution). Read it left-to-right:
  * HEIGHT  : how faithful the steer is at that rung (higher = better).
  * SLOPE   : how fast faithfulness is FALLING (steeper down = more fragile).
  * The KNEE: where the curve dives below the usability FLOOR -- the rung where
    the concept stops being trustworthy. A knee FAR to the right = robust; a knee
    near the left = fragile.
  * ROBUSTNESS: a curve that stays high and crosses the floor LATER (or never) is
    more robust. The headline comparison is: does the on-manifold curve sit ABOVE
    the naive curve and/or keep its knee FURTHER right?
The CI BAND around each line shows how sure we are given we measured only a
handful of concepts: a narrow band = a confident reading; overlapping bands at a
rung = the two methods are not clearly different there.

==============================================================================
BOTH PUBLISHABLE OUTCOMES (either way the figure is the answer)
==============================================================================
  * Faithfulness SURVIVES: the curves stay above the floor far out -> vision-SAE
    concept steering is trustworthy under shift (a green light for the field).
  * Faithfulness COLLAPSES: the curves dive below the floor early -> a warning
    that clean-data faithfulness does NOT transfer OOD. Either way, on-manifold
    sitting above naive (or kneeing later) is the method contribution.

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step4_plot_headline.py
Reads outputs/ood_cfs_sweep.csv. Writes outputs/fig1_cfs_ood_sweep.png.
"""
from __future__ import annotations

from _common import banner, load_cfg, outpath


def main() -> None:
    cfg = load_cfg()
    banner("STEP 4 — render the HEADLINE curve outputs/fig1_cfs_ood_sweep.png")

    import matplotlib
    matplotlib.use("Agg")                          # headless: never blocks on a display
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(outpath(cfg["output_csv"].split("/")[-1]))
    floor = float(cfg["cfs_floor"])
    variants = cfg["variants"]

    # A stable colour + label per method; on-manifold (ours) in standout green.
    style = {
        "onmanifold_steer": ("#2ca02c", "on-manifold (ours)", "o"),
        "naive_steer":      ("#d62728", "naive (off-manifold)", "s"),
    }

    fig, ax = plt.subplots(figsize=(11, 6))

    # x-axis = the ordered shift ladder; tick labels = the benchmark names.
    order = (df[["severity_index", "shift_level"]]
             .drop_duplicates().sort_values("severity_index"))
    xs = order["severity_index"].tolist()
    xlabels = order["shift_level"].tolist()

    # vertical offsets for the knee labels so two methods kneeing at the same rung
    # do not overprint each other.
    knee_dy = {"onmanifold_steer": 0.16, "naive_steer": -0.18}
    for variant in variants:
        sub = df[df.variant == variant].sort_values("severity_index")
        color, label, marker = style.get(variant, ("#1f77b4", variant, "^"))
        x = sub["severity_index"].to_numpy()
        y = sub["cfs"].to_numpy()
        lo = sub["cfs_ci_lo"].to_numpy()
        hi = sub["cfs_ci_hi"].to_numpy()
        ax.plot(x, y, marker=marker, color=color, lw=2.2, label=label, zorder=3)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, zorder=1,
                        label=f"{label} {int(cfg['ci_pct'])}% CI")

        # Mark this method's COLLAPSE KNEE (first rung below the floor).
        below = sub[sub.cfs < floor]
        if len(below):
            kx = int(below.iloc[0]["severity_index"])
            ky = float(below.iloc[0]["cfs"])
            ax.axvline(kx, color=color, ls=":", lw=1.3, alpha=0.8, zorder=2)
            dy = knee_dy.get(variant, 0.14)
            ax.annotate(f"{label}\ncollapse knee", xy=(kx, ky),
                        xytext=(kx - 1.7, ky + dy), color=color, fontsize=9,
                        ha="left",
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.4))

    # The usability FLOOR (below it = no longer faithful).
    ax.axhline(floor, color="black", ls="--", lw=1.2, alpha=0.7)
    ax.text(xs[0], floor + 0.012, f"usability floor = {floor}", fontsize=9,
            color="black", va="bottom")

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, rotation=25, ha="right")
    ax.set_xlabel("out-of-distribution shift  (clean -> rendition -> sketch -> "
                  "corruption sev 1..5 -> ObjectNet)")
    ax.set_ylabel("Causal Faithfulness Score  (CFS, 0..1; higher = more faithful)")
    ax.set_ylim(0, 1.0)
    ax.set_title("FAITH-SAE M6 (HEADLINE / RQ3): does steering faithfulness survive "
                 "distribution shift?\n(offline synthetic OOD ladder — measured CFS, "
                 "not illustrative)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = outpath(cfg["output_png"].split("/")[-1])
    fig.savefig(out, dpi=200)
    print(f"  saved -> {out}")
    print("  Read it: higher line = more faithful; the curve that stays above the")
    print("  floor further out (knee to the RIGHT) is the more OOD-robust method.")
    print("\nSTEP 4 done. This PNG is the offline analog of the paper's fig1_cfs_ood_sweep.png.")


# REAL RUN (M6): identical figure, real numbers -- this IS the paper's
# fig1_cfs_ood_sweep.png once real CLIP + ImageNet-shift activations replace the
# synthetic ladder. The x-axis ticks are already the real benchmark names.
if __name__ == "__main__":
    main()
