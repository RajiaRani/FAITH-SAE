"""step5_plot_and_interpret.py
================================================================================
STEP 5 of 5 — DRAW THE CFS BAR CHART AND EXPLAIN THE RESULT
================================================================================
Run me:   /usr/bin/python3 step5_plot_and_interpret.py
Reads:    outputs/milestone1_cfs.csv   (produced by step4)
Writes:   outputs/milestone1_cfs.png

WHAT YOU LEARN HERE
-------------------
How to read the result of the whole miniature pipeline: a bar of CFS per steering
method, and what "on-manifold wins" means for the research claim.

This step is pure presentation + interpretation; all the science happened in
step4. If the CSV is missing, run step4 first (or run.py runs the whole chain).
"""
from __future__ import annotations

import csv

from _common import HERE, banner, load_cfg


def main() -> None:
    cfg = load_cfg()
    banner("STEP 5 — PLOT CFS PER METHOD AND INTERPRET")

    csv_path = HERE / cfg["output_csv"]
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}. Run step4 first:\n"
                         f"   /usr/bin/python3 step4_steer_and_score.py")

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    variants = [r["variant"] for r in rows]
    cfs = [float(r["cfs"]) for r in rows]
    residual = [float(r["offmanifold_residual"]) for r in rows]

    # --- Draw the bar chart. matplotlib's "Agg" backend renders to a file with
    # NO display/window needed (works headless on any machine).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # color on-manifold (ours) distinctly so the winner is obvious at a glance.
    colors = ["#d98c5f" if v != "onmanifold_steer" else "#2f8f4e" for v in variants]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(range(len(variants)), cfs, color=colors)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=15, ha="right")
    ax.set_ylabel("Causal Faithfulness Score (CFS), [0,1]")
    ax.set_ylim(0, 1.0)
    ax.set_title("Milestone 1 (synthetic): on-manifold steering is most faithful")
    for b, c, res in zip(bars, cfs, residual):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.02,
                f"CFS {c:.2f}\noffman {res:.2f}", ha="center", va="bottom", fontsize=8)
    fig.text(0.99, 0.01, "For research and educational purposes only.",
             ha="right", va="bottom", fontsize=7, color="gray")
    fig.tight_layout()

    out_png = HERE / cfg["output_png"]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")

    # --- Plain-English interpretation ----------------------------------------
    best = max(rows, key=lambda r: float(r["cfs"]))
    print("\nHOW TO READ THIS")
    print("  * Taller bar = more faithful steering (the steer is monotone, specific,")
    print("    and sufficient all at once).")
    print("  * 'offman' label = off-manifold residual; ~0 means the edit stayed on")
    print("    the real-image manifold; ~1 means it drifted off into artifact land.")
    print(f"  * Winner this run: {best['variant']} (CFS {float(best['cfs']):.3f}).")
    print("\nWHY IT MATTERS FOR THE RESEARCH (RQ1):")
    print("  on-manifold steering scoring highest with ~0 off-manifold residual is")
    print("  the miniature, synthetic version of the paper's headline claim — that")
    print("  constraining a steer to the data manifold makes it causally faithful,")
    print("  where naive off-manifold steering only LOOKS effective but leaks.")

    print("\n[STEP 5 OK]  Milestone 1 complete: you ran the whole FAITH-SAE pipeline")
    print("in miniature and understand every object in it.")


if __name__ == "__main__":
    main()
