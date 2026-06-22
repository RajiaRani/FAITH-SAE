"""step4_plot.py — draw the grouped-bar figure outputs/cfs_breakdown.png.

==============================================================================
WHAT THIS STEP DOES (in one sentence)
==============================================================================
It reads cfs_breakdown.csv (the MEASURED components from step3) and renders a
grouped-bar chart: for each steering method, four bars side by side —
monotonicity, specificity, sufficiency, and the combined CFS — so you can see at
a glance WHICH component each method wins or loses, and that on-manifold is among
the most faithful overall.

==============================================================================
HOW TO READ THE PICTURE (the one-glance takeaway)
==============================================================================
  * Each METHOD is a cluster of 4 bars: [mono | spec | suff | CFS].
  * The CFS bar (the 4th, darkest) is the harmonic mean of the other three, so it
    can never sit above the WEAKEST of them — that is the conjunctive penalty.
  * naive/clamp usually have a tall sufficiency/monotonicity but a SHORT
    specificity bar (their off-manifold edit leaks into other concepts) -> the
    weak specificity drags their CFS down.
  * on-manifold keeps all three bars tall -> tall CFS, near the supervised ceiling.
  * random's monotonicity bar is on the floor (no real concept) -> CFS collapses.
    That is the sanity check: the metric does not reward "big change" alone.

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step4_plot.py
Reads outputs/cfs_breakdown.csv. Writes outputs/cfs_breakdown.png.
"""
from __future__ import annotations

from _common import banner, load_cfg, outpath


def main() -> None:
    load_cfg()
    banner("STEP 4 — render outputs/cfs_breakdown.png (grouped bars)")

    import matplotlib
    matplotlib.use("Agg")                       # headless: no display needed
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.read_csv(outpath("cfs_breakdown.csv"))
    # Sort tallest CFS first so the figure reads left-to-right "best -> worst".
    df = df.sort_values("cfs", ascending=False).reset_index(drop=True)

    variants = df["variant"].tolist()
    components = ["monotonicity", "specificity", "sufficiency", "cfs"]
    comp_colors = {
        "monotonicity": "#4C72B0",   # blue
        "specificity": "#55A868",    # green
        "sufficiency": "#C44E52",    # red
        "cfs": "#000000",            # black = the combined headline score
    }

    n_methods = len(variants)
    n_comp = len(components)
    x = np.arange(n_methods)
    width = 0.8 / n_comp

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, comp in enumerate(components):
        vals = df[comp].to_numpy()
        offs = x + (i - (n_comp - 1) / 2) * width
        bars = ax.bar(offs, vals, width, label=comp, color=comp_colors[comp],
                      edgecolor="black", linewidth=0.4)
        for bx, v in zip(offs, vals):
            ax.text(bx, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=18, ha="right")
    ax.set_ylabel("score in [0, 1]")
    ax.set_ylim(0, 1.05)
    ax.set_title("FAITH-SAE M5: MEASURED CFS breakdown by steering method\n"
                 "(monotonicity / specificity / sufficiency -> harmonic-mean CFS; "
                 "offline synthetic run)")
    ax.legend(title="component", ncol=4, loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Mark the on-manifold cluster as OURS so the reader's eye lands on it.
    if "onmanifold_steer" in variants:
        idx = variants.index("onmanifold_steer")
        ax.annotate("ours", xy=(idx, 1.0), ha="center", va="bottom",
                    fontsize=10, color="#2ca02c", fontweight="bold")

    fig.tight_layout()
    out = outpath("cfs_breakdown.png")
    fig.savefig(out, dpi=200)
    print(f"  saved -> {out}")
    print("  Read it: the CFS (black) bar can never top a method's WEAKEST")
    print("  component — that is the harmonic mean's conjunctive penalty in action.")
    print("\nSTEP 4 done. The milestone is complete.")


# REAL RUN (M5): same figure, real numbers — this is the offline analog of the
# paper's fig7_by_method_bar.png (mean CFS by steering variant). The OOD-sweep
# figure fig1_cfs_ood_sweep.png comes from milestone 6's shift loop.
if __name__ == "__main__":
    main()
