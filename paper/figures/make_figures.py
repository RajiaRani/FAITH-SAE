#!/usr/bin/env python3
"""Render the 9 paper figures for FAITH-SAE (prospectus #25).

Two schematics (fig_overview, fig_method) drawn with matplotlib patches, plus
seven data plots carrying ILLUSTRATIVE placeholder numbers (fixed seed, captions
in paper.tex mark them "Illustrative"). Run with:  /usr/bin/python3 make_figures.py
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- global style
plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 220,
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 10.5,
})

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 25
rng = np.random.default_rng(SEED)

# Consistent colorblind-friendly palette: same colour = same method everywhere.
C = {
    "onmanifold": "#0072B2",   # ours (blue)
    "supervised": "#009E73",   # green (TCAV gold reference)
    "clamp":      "#E69F00",   # orange
    "naive":      "#D55E00",   # vermillion (main competitor)
    "random":     "#999999",   # grey (null)
}
LABEL = {
    "onmanifold": "On-manifold (ours)",
    "supervised": "Supervised (TCAV)",
    "clamp":      "Raw clamp",
    "naive":      "Naive off-manifold",
    "random":     "Random direction",
}


def save(fig, name):
    out = os.path.join(HERE, name)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name, os.path.getsize(out), "bytes")


# ============================================================ FIG: OVERVIEW
def fig_overview():
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 38)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333333", fs=11, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.18,rounding_size=0.8",
                     fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, weight=weight, wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                     arrowstyle="-|>", mutation_scale=20,
                     lw=2.4, color="#222222"))

    ax.text(50, 35.5, "FAITH-SAE: are vision-SAE concept directions causally faithful under shift?",
            ha="center", va="center", fontsize=15, weight="bold")

    blue, green, orange, purple, red = "#cfe2f3", "#d9ead3", "#fce5cd", "#e6d8f0", "#f4cccc"

    y = 19
    box(1.5, y, 14, 9, "Frozen\nCLIP ViT-B/16\n(patch activations)", blue, fs=11, weight="bold")
    arrow(15.5, y + 4.5, 19.5, y + 4.5)
    box(19.5, y, 13, 9, "TopK SAE\ndictionary\n(thousands of\nconcepts)", green, fs=10.5)
    arrow(32.5, y + 4.5, 36.5, y + 4.5)
    box(36.5, y, 14, 9, "Concept\ndirections d\n(SAE decoder)", green, fs=11)
    arrow(50.5, y + 4.5, 54.5, y + 4.5)
    box(54.5, y, 14, 9, "Select\ntestable concepts\n(~10-15%\nreliable)", orange, fs=10.5)
    arrow(68.5, y + 4.5, 72.5, y + 4.5)
    box(72.5, y, 12.5, 9, "On-manifold\nsteer\n$a' = a + s\\,P_M\\Delta$", purple, fs=10.5, weight="bold")

    # down to measurement row
    arrow(78.7, y, 78.7, 13.5)
    box(54.5, 4.5, 24.5, 9, "Measure CFS  =  HM(monotonicity,\nspecificity, sufficiency) $\\in[0,1]$",
        red, fs=10.5, weight="bold")
    arrow(54.5, 9, 50, 9)
    box(20, 4.5, 30, 9,
        "OOD sweep:  clean $\\to$ ImageNet-R $\\to$ Sketch\n$\\to$ ImageNet-C (sev. 1-5) $\\to$ ObjectNet",
        "#fff2cc", fs=10.5)

    ax.text(4, 31.5, "the dictionary", fontsize=10, style="italic", color="#555555")
    ax.text(73.0, 31.5, "the faithful edit", fontsize=10, style="italic", color="#555555")
    ax.text(21.5, 1.0, "the answer: where does faithfulness survive shift?",
            fontsize=10, style="italic", color="#555555")
    save(fig, "fig_overview.png")


# ============================================================ FIG: METHOD
def fig_method():
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333333", fs=11, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.2,rounding_size=0.8",
                     fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, weight=weight)

    def arrow(x0, y0, x1, y1, color="#222222", style="-|>"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                     arrowstyle=style, mutation_scale=18, lw=2.2, color=color))

    ax.text(50, 43.5, "On-manifold steering + CFS computation", ha="center", weight="bold", fontsize=15)

    blue, purple, green, orange, red = "#cfe2f3", "#e6d8f0", "#d9ead3", "#fce5cd", "#f4cccc"

    # --- top: the edit + projection -------------------------------------
    box(2, 30, 15, 9, "Raw edit\n$\\Delta = s\\cdot d$\n(concept dir. $d$)", blue, fs=11)
    arrow(17, 34.5, 22, 34.5)
    box(22, 30, 20, 9, "Project onto top-$r$\nreal-image subspace\n$P_M = U_r U_r^\\top$",
        purple, fs=11, weight="bold")
    arrow(42, 34.5, 47, 34.5)
    box(47, 30, 17, 9, "On-manifold edit\n$a' = a + P_M\\Delta$", purple, fs=11)
    arrow(64, 34.5, 69, 34.5)
    box(69, 30, 16, 9, "Steered frozen\nViT readout", blue, fs=11)

    # residual diagnostic callout
    ax.text(32, 27.0, "off-manifold residual  $\\|\\Delta-P_M\\Delta\\| / \\|\\Delta\\|$  (0 = faithful)",
            ha="center", fontsize=9.5, style="italic", color="#555555")

    # --- knob sweep feeding the three readouts --------------------------
    box(2, 16, 18, 8, "Sweep knob $s$\n(steering strength)", "#fff2cc", fs=11)
    arrow(20, 20, 26, 20)

    box(28, 16.5, 19, 8, "Monotonicity\n(knob $\\to$ readout,\nSpearman $\\rho$)", green, fs=10)
    box(50, 16.5, 19, 8, "Specificity\n(off-target probes\nstay flat)", green, fs=10)
    box(72, 16.5, 19, 8, "Sufficiency\n(effect size vs\nclaimed meaning)", green, fs=10)

    # connect readout box to the three
    arrow(77, 30, 56, 24.5)   # readout -> specificity area (representative)
    arrow(46.5, 20.5, 49.5, 20.5, color="#888888", style="-")
    arrow(68.5, 20.5, 71.5, 20.5, color="#888888", style="-")

    # --- combine into CFS ----------------------------------------------
    for xb in (37.5, 59.5, 81.5):
        arrow(xb, 16.5, 50, 9)
    box(33, 2, 34, 7,
        "CFS  =  harmonic mean$(\\,\\cdot\\,)\\in[0,1]$\n(conjunctive: faithful only if all three hold)",
        red, fs=11, weight="bold")

    ax.text(2.5, 11.5, "naive steering = the $r\\!\\to\\!\\infty$ ($P_M{=}I$) special case",
            fontsize=9.5, style="italic", color="#555555")
    save(fig, "fig_method.png")


# ============================================================ FIG 1 (HEADLINE)
def fig1_cfs_ood_sweep():
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    stages = ["clean", "ImgNet-R", "Sketch", "C-1", "C-2", "C-3", "C-4", "C-5", "ObjectNet"]
    x = np.arange(len(stages))

    on = np.array([0.86, 0.81, 0.77, 0.79, 0.74, 0.68, 0.61, 0.55, 0.49])
    na = np.array([0.71, 0.58, 0.46, 0.52, 0.43, 0.34, 0.26, 0.20, 0.16])
    on_ci = 0.035 + 0.004 * x
    na_ci = 0.045 + 0.005 * x

    ax.fill_between(x, on - on_ci, on + on_ci, color=C["onmanifold"], alpha=0.18)
    ax.fill_between(x, na - na_ci, na + na_ci, color=C["naive"], alpha=0.18)
    ax.plot(x, on, "-o", color=C["onmanifold"], lw=2.6, ms=6, label=LABEL["onmanifold"])
    ax.plot(x, na, "-s", color=C["naive"], lw=2.6, ms=6, label=LABEL["naive"])

    floor = 0.5
    ax.axhline(floor, ls="--", lw=1.4, color="#444444")
    ax.text(0.15, floor + 0.015, "usability floor", fontsize=9.5, color="#444444")

    # collapse knee where naive crosses the floor (between Sketch and C-1) and on-manifold near end
    knee_na = 2.0  # Sketch first below floor for naive
    ax.scatter([knee_na], [na[2]], s=170, facecolor="none", edgecolor=C["naive"], lw=2.4, zorder=5)
    ax.annotate("naive collapse\nknee", (knee_na, na[2]), xytext=(3.1, 0.30),
                fontsize=9.5, color=C["naive"], ha="left",
                arrowprops=dict(arrowstyle="-|>", color=C["naive"], lw=1.6))
    ax.scatter([7], [on[7]], s=170, facecolor="none", edgecolor=C["onmanifold"], lw=2.4, zorder=5)
    ax.annotate("on-manifold\ncollapse knee", (7, on[7]), xytext=(5.0, 0.86),
                fontsize=9.5, color=C["onmanifold"], ha="center",
                arrowprops=dict(arrowstyle="-|>", color=C["onmanifold"], lw=1.6))

    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=35, ha="right", fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Causal Faithfulness Score (CFS)")
    ax.set_xlabel("distribution-shift severity  (clean $\\to$ out-of-distribution)")
    ax.set_title("Faithfulness survives shift on-manifold, collapses naive")
    ax.legend(loc="upper right")
    save(fig, "fig1_cfs_ood_sweep.png")


# ============================================================ FIG 2 (PARETO)
def fig2_faithfulness_pareto():
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    # x = monotonicity (effect realism), y = specificity; up-right is better
    pts = {
        "supervised": (0.88, 0.83),
        "onmanifold": (0.84, 0.88),
        "clamp":      (0.62, 0.55),
        "naive":      (0.74, 0.40),
        "random":     (0.30, 0.22),
    }
    # label offsets tuned to avoid all collisions
    loff = {
        "supervised": (0.0, -0.075, "center"),
        "onmanifold": (0.0, 0.050, "center"),
        "clamp":      (0.0, 0.050, "center"),
        "naive":      (0.0, -0.075, "center"),
        "random":     (0.04, 0.050, "left"),
    }
    for k, (mx, my) in pts.items():
        ax.scatter(mx, my, s=230, color=C[k], edgecolor="white", lw=1.5, zorder=4)
        dx, dy, ha = loff[k]
        ax.annotate(LABEL[k], (mx, my), xytext=(mx + dx, my + dy),
                    ha=ha, fontsize=10, color=C[k], weight="bold")

    # frontier line through the two best
    fx = [pts["onmanifold"][0], pts["supervised"][0]]
    fy = [pts["onmanifold"][1], pts["supervised"][1]]
    ax.plot(fx, fy, "--", color="#555555", lw=1.6, zorder=2)
    ax.text(0.86, 0.965, "Pareto frontier", fontsize=9.5, color="#555555", ha="center")

    ax.set_xlim(0.2, 1.02)
    ax.set_ylim(0.1, 1.02)
    ax.set_xlabel("Monotonicity  (effect realism, Spearman $\\rho$)")
    ax.set_ylabel("Specificity  (1 - off-target drift)")
    ax.set_title("Specificity vs effect across steering methods")
    ax.annotate("better", xy=(0.55, 0.78), xytext=(0.40, 0.62),
                arrowprops=dict(arrowstyle="-|>", color="#222222", lw=2.0),
                fontsize=10, color="#222222", weight="bold")
    save(fig, "fig2_faithfulness_pareto.png")


# ============================================================ FIG 3 (KNEE)
def fig3_strength_sweep():
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    s = np.linspace(0, 6, 200)
    # rises then saturates/declines (magnitude-saturation onset = knee)
    cfs = 0.88 * (1 - np.exp(-1.7 * s)) - 0.10 * np.clip(s - 3.2, 0, None) ** 1.6 * 0.18
    cfs = np.clip(cfs, 0, 0.9)
    ax.plot(s, cfs, "-", color=C["onmanifold"], lw=2.8, label=LABEL["onmanifold"])

    # naive for contrast: high apparent effect early then degrades
    cfs_n = 0.62 * (1 - np.exp(-2.3 * s)) - 0.12 * np.clip(s - 2.0, 0, None) ** 1.5 * 0.3
    cfs_n = np.clip(cfs_n, 0, 0.9)
    ax.plot(s, cfs_n, "-", color=C["naive"], lw=2.4, label=LABEL["naive"])

    knee_s = 3.2
    knee_y = np.interp(knee_s, s, cfs)
    ax.scatter([knee_s], [knee_y], s=180, facecolor="none", edgecolor=C["onmanifold"], lw=2.6, zorder=5)
    ax.annotate("knee\n(saturation onset)", (knee_s, knee_y), xytext=(4.0, 0.55),
                fontsize=10, ha="center", color=C["onmanifold"],
                arrowprops=dict(arrowstyle="-|>", color=C["onmanifold"], lw=1.7))
    ax.axvline(knee_s, ls=":", color="#888888", lw=1.3)

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Steering strength  $s$")
    ax.set_ylabel("Causal Faithfulness Score (CFS)")
    ax.set_title("CFS vs steering strength: the saturation knee")
    ax.legend(loc="lower right")
    save(fig, "fig3_strength_sweep.png")


# ============================================================ FIG 4 (RELIABILITY)
def fig4_concept_reliability():
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    # bimodal: large unreliable mass near 0.2, small reliable tail near 0.8
    n = 4000
    unreliable = rng.beta(2.2, 6.0, int(n * 0.86)) * 0.6
    reliable = 0.62 + rng.beta(2.5, 2.0, int(n * 0.14)) * 0.36
    vals = np.clip(np.concatenate([unreliable, reliable]), 0, 1)

    thr = 0.6
    bins = np.linspace(0, 1, 36)
    counts, edges = np.histogram(vals, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cols = [C["onmanifold"] if c >= thr else "#bdbdbd" for c in centers]
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.92, color=cols, edgecolor="white", lw=0.4)

    ax.axvline(thr, ls="--", color="#444444", lw=1.6)
    frac = 100 * np.mean(vals >= thr)
    ax.text(thr + 0.02, ax.get_ylim()[1] * 0.92,
            f"reliable tail\n({frac:.0f}% of concepts,\nCFS $\\geq$ {thr:.1f})",
            fontsize=10, color=C["onmanifold"], va="top")
    ax.text(0.16, ax.get_ylim()[1] * 0.78, "unreliable\n/ polysemantic\nmass",
            fontsize=10, color="#777777", ha="center")

    ax.set_xlim(0, 1)
    ax.set_xlabel("per-concept CFS")
    ax.set_ylabel("number of SAE concepts")
    ax.set_title("Only a thin tail of concepts steers faithfully")
    save(fig, "fig4_concept_reliability.png")


# ============================================================ FIG 5 (HEATMAP)
def fig5_ood_heatmap():
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    rows = ["clean", "ImageNet-R", "Sketch", "ImageNet-C", "ObjectNet"]
    cols = ["Random", "Naive", "Raw clamp", "On-manifold", "Supervised"]
    # method quality increases left->right; shift degrades top->bottom.
    method_base = np.array([0.22, 0.50, 0.58, 0.85, 0.88])
    shift_drop = np.array([0.00, 0.10, 0.17, 0.27, 0.36])
    M = np.zeros((len(rows), len(cols)))
    for i, d in enumerate(shift_drop):
        for j, b in enumerate(method_base):
            # methods that are more on-manifold degrade more gracefully
            grace = 0.45 + 0.55 * (j / (len(cols) - 1))
            M[i, j] = np.clip(b - d * (1.4 - grace), 0.05, 0.95)

    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("CFS")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=25, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("steering method")
    ax.set_ylabel("distribution shift")
    ax.set_title("CFS by shift type and steering method")
    ax.grid(False)

    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.62 else "black", fontsize=9.5)

    # best cell: on-manifold, clean (col 3, row 0). Box it and label inside-down
    # so the "best" tag never collides with the title.
    bi, bj = 0, 3
    ax.add_patch(Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor="#D55E00", lw=3))
    ax.annotate("best", xy=(bj, bi), xytext=(bj + 0.95, bi + 0.62),
                color="#D55E00", ha="left", fontsize=10, weight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#D55E00", lw=1.8))
    save(fig, "fig5_ood_heatmap.png")


# ============================================================ FIG 6 (MONOTONICITY)
def fig6_monotonicity_curve():
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    knob = np.linspace(-3, 3, 60)
    # on-manifold: smooth monotone sigmoid-ish readout
    on = 1.0 / (1 + np.exp(-1.4 * knob))
    # naive: jagged, non-monotone (off-manifold artifacts)
    na = 1.0 / (1 + np.exp(-1.0 * knob))
    na = na + 0.16 * np.sin(2.6 * knob) + rng.normal(0, 0.05, knob.size)
    na = np.clip(na, 0, 1.05)

    ax.plot(knob, on, "-o", color=C["onmanifold"], lw=2.6, ms=4, label=LABEL["onmanifold"])
    ax.plot(knob, na, "-s", color=C["naive"], lw=1.8, ms=3.5, alpha=0.9, label=LABEL["naive"])

    ax.text(-2.9, 0.92, "smooth, monotone $\\Rightarrow$ high CFS", fontsize=9.5, color=C["onmanifold"])
    ax.text(0.2, 0.14, "jagged, non-monotone\n$\\Rightarrow$ off-manifold artifact",
            fontsize=9.5, color=C["naive"])

    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("steering knob  $s$  (concept down $\\to$ up)")
    ax.set_ylabel("held-out concept readout (normalized)")
    ax.set_title("Concept readout vs knob for an example concept")
    ax.legend(loc="upper left")
    save(fig, "fig6_monotonicity_curve.png")


# ============================================================ FIG 7 (BAR)
def fig7_by_method_bar():
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    order = ["supervised", "onmanifold", "clamp", "naive", "random"]
    means = np.array([0.84, 0.85, 0.57, 0.49, 0.21])
    ci = np.array([0.03, 0.035, 0.05, 0.055, 0.04])
    x = np.arange(len(order))
    cols = [C[k] for k in order]
    ax.bar(x, means, yerr=ci, capsize=5, color=cols, edgecolor="white", lw=0.8,
           error_kw=dict(ecolor="#333333", lw=1.5))

    for xi, m, e in zip(x, means, ci):
        ax.text(xi, m + e + 0.02, f"{m:.2f}", ha="center", fontsize=10, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[k].replace(" (", "\n(").replace(" direction", "\ndir.")
                        .replace(" off-manifold", "\noff-manifold") for k in order],
                       fontsize=9)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Causal Faithfulness Score (CFS)")
    ax.set_title("Mean CFS by steering variant (bootstrap 95% CI)")
    save(fig, "fig7_by_method_bar.png")


if __name__ == "__main__":
    fig_overview()
    fig_method()
    fig1_cfs_ood_sweep()
    fig2_faithfulness_pareto()
    fig3_strength_sweep()
    fig4_concept_reliability()
    fig5_ood_heatmap()
    fig6_monotonicity_curve()
    fig7_by_method_bar()
    print("all figures rendered")
