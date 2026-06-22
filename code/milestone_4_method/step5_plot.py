"""step5_plot.py — draw the headline figure outputs/method_compare.png.

==============================================================================
WHAT THIS STEP DOES (in one sentence)
==============================================================================
It reads method_compare.csv and renders a two-panel PNG: (left) a bar chart of
CFS per steering method, (right) a scatter of off-manifold residual (x) vs CFS
(y) — on-manifold should sit in the top-LEFT corner (high CFS, ~0 residual) while
naive/clamp/random sit lower-right (off the sheet, low CFS).

==============================================================================
HOW TO READ THE PICTURE (the one-glance takeaway)
==============================================================================
  * LEFT bar chart: taller bar = more faithful (higher CFS). on-manifold's bar is
    the tallest of the four runnable methods; random's is near the floor.
  * RIGHT scatter: the GOAL corner is top-left = "faithful AND on-manifold".
      - on-manifold  -> top-left  (CFS high, residual ~ 0): edits stay realistic.
      - naive/clamp  -> lower-right (CFS low, residual large): off-manifold mirage.
      - random       -> bottom    (no real concept).
    The arrow from naive to on-manifold is the contribution: projecting the edit
    onto the real-image sheet drags it from "off-manifold, unfaithful" up into
    "on-manifold, faithful".

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step5_plot.py
Reads outputs/method_compare.csv. Writes outputs/method_compare.png.
"""
from __future__ import annotations

import csv

from _common import banner, load_cfg, outpath


def read_table():
    rows = []
    with open(outpath("method_compare.csv"), newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "variant": row["variant"],
                "offmanifold_residual": float(row["offmanifold_residual"]),
                "cfs": float(row["cfs"]),
            })
    return rows


def main() -> None:
    load_cfg()
    banner("STEP 5 — render outputs/method_compare.png")

    import matplotlib
    matplotlib.use("Agg")                       # headless: no display needed
    import matplotlib.pyplot as plt

    rows = read_table()
    variants = [r["variant"] for r in rows]
    cfs = [r["cfs"] for r in rows]
    resid = [r["offmanifold_residual"] for r in rows]

    # A stable color per method; on-manifold (ours) in a standout green.
    palette = {
        "onmanifold_steer": "#2ca02c",   # ours
        "naive_steer": "#d62728",        # main competitor
        "clamp_steer": "#ff7f0e",
        "random_steer": "#7f7f7f",
        "supervised_steer": "#1f77b4",
    }
    colors = [palette.get(v, "#9467bd") for v in variants]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- LEFT: CFS bar chart ------------------------------------------------
    bars = axL.bar(range(len(variants)), cfs, color=colors)
    axL.set_xticks(range(len(variants)))
    axL.set_xticklabels(variants, rotation=20, ha="right")
    axL.set_ylabel("Causal Faithfulness Score (CFS)")
    axL.set_ylim(0, 1.0)
    axL.set_title("CFS by steering method (higher = more faithful)")
    for b, c in zip(bars, cfs):
        axL.text(b.get_x() + b.get_width() / 2, c + 0.01, f"{c:.2f}",
                 ha="center", va="bottom", fontsize=9)

    # ---- RIGHT: residual vs CFS scatter (goal corner = top-left) -----------
    for v, x, y, c in zip(variants, resid, cfs, colors):
        axR.scatter(x, y, s=160, color=c, edgecolor="black", zorder=3)
        axR.annotate(v, (x, y), textcoords="offset points", xytext=(8, 6),
                     fontsize=9)
    # arrow naive -> on-manifold: the contribution of projecting onto the sheet
    try:
        nv = next(r for r in rows if r["variant"] == "naive_steer")
        onm = next(r for r in rows if r["variant"] == "onmanifold_steer")
        axR.annotate("", xy=(onm["offmanifold_residual"], onm["cfs"]),
                     xytext=(nv["offmanifold_residual"], nv["cfs"]),
                     arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2))
        axR.text(0.52, 0.55, "project onto\nreal-image sheet",
                 color="#2ca02c", fontsize=9, transform=axR.transAxes)
    except StopIteration:
        pass
    axR.set_xlabel("off-manifold residual  (0 = on the sheet -> 1 = off the sheet)")
    axR.set_ylabel("Causal Faithfulness Score (CFS)")
    axR.set_xlim(-0.05, 1.05)
    axR.set_ylim(0, 1.0)
    axR.set_title("Faithful AND on-manifold lives in the TOP-LEFT corner")
    axR.grid(True, alpha=0.3)

    fig.suptitle("FAITH-SAE M4: on-manifold steering is faithful where naive is not "
                 "(illustrative offline synthetic run)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = outpath("method_compare.png")
    fig.savefig(out, dpi=200)
    print(f"  saved -> {out}")
    print("\nSTEP 5 done. Open the PNG: on-manifold = tallest bar / top-left dot.")


# REAL RUN (M4): same figure, real numbers — this is the offline analog of the
# paper's fig7_by_method_bar.png / fig2_faithfulness_pareto.png.
if __name__ == "__main__":
    main()
