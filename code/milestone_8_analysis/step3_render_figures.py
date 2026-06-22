#!/usr/bin/env python3
# ===========================================================================
#  step3_render_figures.py  —  Milestone 8 (Analysis), Step 3
#  Render the REAL (measured-on-this-run) versions of two paper figures into
#  outputs/, matching paper/figures/make_figures.py style (same colorblind
#  palette, same titles, same shape):
#    * fig1_cfs_ood_sweep.png  — CFS vs shift severity, on-manifold vs naive,
#                                with bootstrap CI bands and the collapse knee.
#    * fig7_by_method_bar.png  — mean CFS by steering method on the CLEAN rung,
#                                with bootstrap 95% CI error bars.
#  These replace the paper's "illustrative placeholder" figures with measured ones.
#  FAITH-SAE  ·  author: Rajia Rani  ·  for research and educational purposes only
# ===========================================================================
#
#  ============ HOW TO READ THE TWO PICTURES ============
#
#  fig1_cfs_ood_sweep.png  (THE HEADLINE — RQ3)
#    x-axis = shift severity, walking clean -> ImgNet-R -> Sketch -> C-3 -> C-5 ->
#    ObjectNet (left=easy, right=hardest). y-axis = CFS in [0,1] (higher=more
#    faithful). Two lines: on-manifold (blue) and naive (vermillion). Each line
#    has a SHADED BAND = its bootstrap 95% CI (how much the mean could wobble).
#    A dashed "usability floor" marks CFS=0.50. The COLLAPSE KNEE (a ring) is the
#    first rung where a method's mean CFS drops below the floor. The takeaway in
#    one glance: on-manifold stays higher and crosses the floor LATER (or never).
#
#  fig7_by_method_bar.png  (RQ1)
#    One bar per steering method (supervised / on-manifold / clamp / naive /
#    random) showing mean CFS on the CLEAN rung, with a bootstrap 95% CI ERROR
#    BAR on top. Bars whose error bars do NOT overlap differ for real. Takeaway:
#    on-manifold sits close behind supervised and clearly above clamp/naive/random.
#
#  We REUSE the exact palette + labels from paper/figures/make_figures.py so the
#  offline figures look like the paper's, just with measured numbers.
#
#  RUN:  /usr/bin/python3 step3_render_figures.py   (needs step1 + step2 CSVs)
#  ========================================================================

from __future__ import annotations

import csv

from _common import banner, load_cfg, outpath

# Same colorblind-friendly palette + labels as paper/figures/make_figures.py.
C = {
    "onmanifold_steer": "#0072B2",   # ours (blue)
    "supervised_steer": "#009E73",   # green (TCAV gold reference)
    "clamp_steer":      "#E69F00",   # orange
    "naive_steer":      "#D55E00",   # vermillion (main competitor)
    "random_steer":     "#999999",   # grey (null)
}
LABEL = {
    "onmanifold_steer": "On-manifold (ours)",
    "supervised_steer": "Supervised (TCAV)",
    "clamp_steer":      "Raw clamp",
    "naive_steer":      "Naive off-manifold",
    "random_steer":     "Random direction",
}


def read_bootstrap(path: str):
    """variant -> shift -> {mean_cfs, ci_low, ci_high}, plus shift order."""
    from collections import defaultdict
    table = defaultdict(dict)
    shift_order, seen = [], set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table[row["variant"]][row["shift"]] = {
                "mean": float(row["mean_cfs"]),
                "lo": float(row["ci_low"]),
                "hi": float(row["ci_high"]),
            }
            if row["shift"] not in seen:
                seen.add(row["shift"])
                shift_order.append(row["shift"])
    return table, shift_order


def read_residual_by_method(path: str, clean_shift: str):
    """variant -> mean off-manifold residual on the clean rung (from step1's
    per-concept table). The residual is the genuine MEASURED separator between
    on-manifold (~0) and naive/clamp (large), even where CFS is close."""
    from collections import defaultdict
    acc = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["shift"] == clean_shift:
                acc[row["variant"]].append(float(row["offmanifold_residual"]))
    return {v: sum(xs) / len(xs) for v, xs in acc.items()}


def collapse_knee(shifts, means, floor):
    """First index where the mean CFS dips below the usability floor (the
    'collapse knee'). Returns None if the method never crosses the floor."""
    for i, m in enumerate(means):
        if m < floor:
            return i
    return None


def fig1_ood_sweep(cfg, table, shift_order):
    """CFS vs shift for on-manifold vs naive, with bootstrap CI bands + knee."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    floor = float(cfg["usability_floor"])
    x = np.arange(len(shift_order))
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    for variant in ("onmanifold_steer", "naive_steer"):
        means = np.array([table[variant][s]["mean"] for s in shift_order])
        los = np.array([table[variant][s]["lo"] for s in shift_order])
        his = np.array([table[variant][s]["hi"] for s in shift_order])
        ax.fill_between(x, los, his, color=C[variant], alpha=0.18)
        marker = "-o" if variant == "onmanifold_steer" else "-s"
        ax.plot(x, means, marker, color=C[variant], lw=2.6, ms=6,
                label=LABEL[variant])
        # mark this method's collapse knee (first rung below the floor)
        knee = collapse_knee(shift_order, means, floor)
        if knee is not None:
            ax.scatter([x[knee]], [means[knee]], s=170, facecolor="none",
                       edgecolor=C[variant], lw=2.4, zorder=5)
            ax.annotate(f"{LABEL[variant].split(' (')[0].lower()}\ncollapse knee",
                        (x[knee], means[knee]),
                        xytext=(x[knee], means[knee] - 0.18),
                        fontsize=9, color=C[variant], ha="center",
                        arrowprops=dict(arrowstyle="-|>", color=C[variant], lw=1.6))

    ax.axhline(floor, ls="--", lw=1.4, color="#444444")
    ax.text(0.1, floor + 0.015, "usability floor", fontsize=9.5, color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels(shift_order, rotation=35, ha="right", fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Causal Faithfulness Score (CFS)")
    ax.set_xlabel("distribution-shift severity  (clean -> out-of-distribution)")
    ax.set_title("Faithfulness vs shift: on-manifold vs naive (measured, 95% CI band)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    out = outpath("fig1_cfs_ood_sweep.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  saved -> {out}")


def fig7_by_method_bar(cfg, table, shift_order):
    """Mean CFS by steering method on the CLEAN rung, with bootstrap 95% CI bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    clean = shift_order[0]                       # the in-distribution rung
    order = list(cfg["variants"])                # supervised, onmanifold, clamp, naive, random
    means = np.array([table[v][clean]["mean"] for v in order])
    los = np.array([table[v][clean]["lo"] for v in order])
    his = np.array([table[v][clean]["hi"] for v in order])
    # asymmetric error bars: distance from mean down to ci_low and up to ci_high.
    yerr = np.vstack([means - los, his - means])
    x = np.arange(len(order))
    cols = [C[v] for v in order]

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.bar(x, means, yerr=yerr, capsize=5, color=cols, edgecolor="white", lw=0.8,
           error_kw=dict(ecolor="#333333", lw=1.5))
    for xi, m, hi in zip(x, means, his):
        ax.text(xi, hi + 0.02, f"{m:.2f}", ha="center", fontsize=10, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[v].replace(" (", "\n(") for v in order], fontsize=9)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Causal Faithfulness Score (CFS)")
    ax.set_title(f"Mean CFS by steering variant on {clean} (measured, bootstrap 95% CI)")
    ax.grid(True, axis="y", alpha=0.3)
    out = outpath("fig7_by_method_bar.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  saved -> {out}")


def main() -> None:
    cfg = load_cfg()
    banner("STEP 3 — render fig1_cfs_ood_sweep.png and fig7_by_method_bar.png")
    table, shift_order = read_bootstrap(outpath("bootstrap_ci.csv"))
    print(f"  shift ladder: {shift_order}")
    fig1_ood_sweep(cfg, table, shift_order)
    fig7_by_method_bar(cfg, table, shift_order)
    print("\nSTEP 3 done. Next: step4 writes FINDINGS.md (the plain-language answers).")


# REAL RUN (M8): same figures, real numbers. fig1/fig7 here are the offline
# analogs of paper/figures/fig1_cfs_ood_sweep.png and fig7_by_method_bar.png;
# once milestones 5-7 produce the real per-concept CFS, this renders the
# measured versions that replace the paper's "Illustrative" placeholders.
if __name__ == "__main__":
    main()
