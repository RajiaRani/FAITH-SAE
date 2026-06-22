#!/usr/bin/env python3
# ===========================================================================
#  figures_real.py  —  FAITH-SAE real-run figure renderer (publication-grade)
#
#  Renders the REAL-DATA versions of the two data figures this stage owns, in
#  the EXACT visual style of paper/figures/make_figures.py (same rcParams, same
#  colour-per-method palette, same save() helper), but driven by the MEASURED
#  result CSVs instead of illustrative placeholder numbers:
#
#    outputs/fig1_cfs_ood_sweep.png  — HEADLINE (RQ3): CFS vs distribution-shift
#        severity, on-manifold vs naive, bootstrap CI bands, usability floor,
#        and the collapse knee (first rung below the floor) marked per method.
#    outputs/fig7_by_method_bar.png  — RQ1: mean CFS by steering variant with
#        bootstrap 95% CI error bars over concepts.
#
#  It reuses analysis_real.bootstrap_by_method / ood_degradation so the figures
#  and the FINDINGS text are computed from ONE statistics path (no drift between
#  the number in the bar and the number in the paper).
#
#  Author: Rajia Rani
#  For research and educational purposes only.
# ===========================================================================
from __future__ import annotations

import argparse
import pathlib
import sys

# --- contract: make project root (src/) and this dir importable -------------
_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: render straight to PNG, no display needed
import matplotlib.pyplot as plt  # noqa: E402

# Reuse the analysis layer so the figure numbers == the FINDINGS numbers.
from analysis_real import (  # noqa: E402
    bootstrap_by_method, bootstrap_ci, ood_degradation,
    _ensure_cfs_column, _load_csv, _pick_col, _method_col, _concept_col,
    METHOD_ORDER, METHOD_LABEL,
)

# --------------------------------------------------------------------------- #
#  GLOBAL STYLE — copied verbatim from paper/figures/make_figures.py so the    #
#  real-data figures are visually indistinguishable from the paper's draft     #
#  placeholders (same fonts, dpi, grid, colour semantics).                     #
# --------------------------------------------------------------------------- #
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

# Same colour = same method everywhere (matches make_figures.py exactly).
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
# Map the registry variant names -> the short palette keys above.
VARIANT_KEY = {
    "onmanifold_steer": "onmanifold",
    "supervised_steer": "supervised",
    "clamp_steer":      "clamp",
    "naive_steer":      "naive",
    "random_steer":     "random",
}


def save(fig, out_dir, name):
    """Identical to make_figures.save: white bg, tight bbox, report bytes."""
    out = pathlib.Path(out_dir) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", str(out), out.stat().st_size, "bytes")
    return out


# ====================================================================== FIG 1
def fig1_cfs_ood_sweep(per_df, out_dir, floor: float = 0.5, n_boot: int = 2000,
                       ci_pct: float = 95.0):
    """HEADLINE: measured CFS vs OOD shift for on-manifold vs naive steering.

    For each shift rung we bootstrap a 95% CI on the mean CFS OVER CONCEPTS, so
    the band is the real per-concept uncertainty (not a placeholder). The
    collapse knee = the first rung whose mean CFS falls below the usability
    floor, marked per method.
    """
    per_df = _ensure_cfs_column(per_df.copy())
    method_col = _method_col(per_df)
    concept_col = _concept_col(per_df)
    level_col = _pick_col(per_df, ["shift", "shift_level", "rung", "level",
                                   "ood_level", "dataset"])
    sev_col = _pick_col(per_df, ["shift_noise", "severity_index", "shift_index",
                                 "severity"])

    # Ordered list of shift rungs (by severity if available, else first-seen).
    if sev_col:
        order = (per_df[[level_col, sev_col]].drop_duplicates()
                 .sort_values(sev_col)[level_col].tolist())
    else:
        order = list(dict.fromkeys(per_df[level_col].tolist()))
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    plotted_any = False
    for variant in ["onmanifold_steer", "naive_steer"]:
        key = VARIANT_KEY[variant]
        sub = per_df[per_df[method_col] == variant]
        if not len(sub):
            continue
        means, los, his = [], [], []
        for lvl in order:
            cell = sub[sub[level_col] == lvl]
            # one CFS per concept at this rung -> bootstrap the mean over concepts
            if concept_col and concept_col in cell.columns:
                vec = cell.groupby(concept_col)["cfs"].mean().to_numpy()
            else:
                vec = cell["cfs"].to_numpy()
            m, lo, hi = bootstrap_ci(vec, n=n_boot, ci_pct=ci_pct)
            means.append(m); los.append(lo); his.append(hi)
        means = np.asarray(means); los = np.asarray(los); his = np.asarray(his)

        ax.fill_between(x, los, his, color=C[key], alpha=0.18)
        marker = "-o" if variant == "onmanifold_steer" else "-s"
        ax.plot(x, means, marker, color=C[key], lw=2.6, ms=6, label=LABEL[key])

        # collapse knee = first rung whose mean CFS dips below the floor.
        below = np.where(means < floor)[0]
        if below.size:
            ki = int(below[0])
            ax.scatter([ki], [means[ki]], s=170, facecolor="none",
                       edgecolor=C[key], lw=2.4, zorder=5)
            # nudge the annotation up for ours, down for naive, to avoid overlap.
            ytxt = means[ki] + 0.22 if variant == "onmanifold_steer" else means[ki] - 0.18
            ax.annotate(f"{LABEL[key].split(' ')[0].lower()}\ncollapse knee",
                        (ki, means[ki]),
                        xytext=(min(ki + 0.4, len(order) - 1.2), np.clip(ytxt, 0.06, 0.92)),
                        fontsize=9.5, color=C[key], ha="left",
                        arrowprops=dict(arrowstyle="-|>", color=C[key], lw=1.6))
        plotted_any = True

    if not plotted_any:
        ax.text(0.5, 0.5, "no on-manifold / naive rows in results",
                ha="center", va="center", transform=ax.transAxes)

    ax.axhline(floor, ls="--", lw=1.4, color="#444444")
    ax.text(0.15, floor + 0.015, "usability floor", fontsize=9.5, color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_level(l) for l in order], rotation=35,
                       ha="right", fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Causal Faithfulness Score (CFS)")
    ax.set_xlabel("distribution-shift severity  (clean $\\to$ out-of-distribution)")
    ax.set_title("Faithfulness survives shift on-manifold, collapses naive")
    if plotted_any:
        ax.legend(loc="upper right")
    return save(fig, out_dir, "fig1_cfs_ood_sweep.png")


# ====================================================================== FIG 7
def fig7_by_method_bar(per_df, out_dir, n_boot: int = 2000, ci_pct: float = 95.0):
    """RQ1: measured mean CFS by steering variant with bootstrap 95% CI bars.

    The bootstrap (over concepts) is the SAME bootstrap_by_method the FINDINGS
    writer uses, so the bar heights and CIs match the paper text exactly.
    """
    boot = bootstrap_by_method(per_df, n=n_boot, ci_pct=ci_pct)
    # Keep the canonical left->right method order where present.
    boot = boot.set_index("variant")
    order = [m for m in METHOD_ORDER if m in boot.index]
    order += [m for m in boot.index if m not in order]
    boot = boot.loc[order].reset_index()

    means = boot["mean_cfs"].to_numpy(dtype=float)
    # asymmetric error bars from the CI (yerr wants distances from the mean).
    lo_err = np.clip(means - boot["ci_low"].to_numpy(dtype=float), 0, None)
    hi_err = np.clip(boot["ci_high"].to_numpy(dtype=float) - means, 0, None)
    yerr = np.vstack([lo_err, hi_err])

    x = np.arange(len(order))
    cols = [C[VARIANT_KEY.get(v, "random")] for v in boot["variant"]]

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.bar(x, means, yerr=yerr, capsize=5, color=cols, edgecolor="white", lw=0.8,
           error_kw=dict(ecolor="#333333", lw=1.5))
    for xi, m, he in zip(x, means, hi_err):
        ax.text(xi, m + he + 0.02, f"{m:.2f}", ha="center", fontsize=10,
                weight="bold")

    # Wrap labels onto two lines exactly like make_figures.py.
    xlabels = [LABEL[VARIANT_KEY.get(v, "random")]
               .replace(" (", "\n(").replace(" direction", "\ndir.")
               .replace(" off-manifold", "\noff-manifold")
               for v in boot["variant"]]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Causal Faithfulness Score (CFS)")
    ax.set_title("Mean CFS by steering variant (bootstrap 95% CI)")
    return save(fig, out_dir, "fig7_by_method_bar.png")


# --------------------------------------------------------------------------- #
#  pretty x-axis labels: map the real dataset rung names to compact tick text  #
# --------------------------------------------------------------------------- #
def _pretty_level(name):
    s = str(name)
    table = {
        # the student's domain-shift ladder (ordered by shift strength)
        "in1k": "IN-1k", "in100": "IN-100", "food101": "Food-101",
        "cifar100": "CIFAR-100",
        # legacy open_clip ladder (still supported)
        "clean": "clean", "imagenet": "clean", "imagenet_val": "clean",
        "imagenet_r": "ImgNet-R", "imgnet-r": "ImgNet-R",
        "imagenet_sketch": "Sketch", "sketch": "Sketch",
        "imagenet_c_s1": "C-1", "imagenet_c_s2": "C-2", "imagenet_c_s3": "C-3",
        "imagenet_c_s4": "C-4", "imagenet_c_s5": "C-5",
        "c-1": "C-1", "c-2": "C-2", "c-3": "C-3", "c-4": "C-4", "c-5": "C-5",
        "objectnet": "ObjectNet",
    }
    return table.get(s.lower(), s)


# --------------------------------------------------------------------------- #
#  render both figures from a results dir (the public entry the orchestrator   #
#  and the CLI both call).                                                     #
# --------------------------------------------------------------------------- #
def make_real_figures(results_dir, out_dir, floor: float = 0.5, n_boot: int = 2000,
                      ci_pct: float = 95.0):
    """Render fig1 + fig7 from the measured per_concept_cfs.csv in results_dir.

    Returns the list of written PNG paths. Mirrors the analysis layer's CSV
    schema so the heavy modules and these figures stay in lockstep.
    """
    results_dir = pathlib.Path(results_dir)
    per_df = _load_csv(results_dir / "per_concept_cfs.csv")
    if per_df is None:
        raise FileNotFoundError(
            f"no per_concept_cfs.csv in {results_dir} "
            f"(run the heavy modules first, or use --smoke)")
    written = []
    written.append(fig1_cfs_ood_sweep(per_df, out_dir, floor=floor,
                                      n_boot=n_boot, ci_pct=ci_pct))
    written.append(fig7_by_method_bar(per_df, out_dir, n_boot=n_boot,
                                      ci_pct=ci_pct))
    return written


# =========================================================================== #
#  SMOKE: fabricate small real-shaped results, render both PNGs on CPU         #
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(
        description="FAITH-SAE real-run figures: render fig1 (OOD CFS sweep) and "
                    "fig7 (CFS by method) from measured CSVs, in the paper style.")
    ap.add_argument("--results-dir", default=str(_HERE / "outputs"),
                    help="dir holding per_concept_cfs.csv")
    ap.add_argument("--out-dir", default=str(_HERE / "outputs"),
                    help="where the PNGs are written")
    ap.add_argument("--floor", type=float, default=0.5,
                    help="usability floor drawn on fig1 + used for the knee")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--ci-pct", type=float, default=95.0)
    ap.add_argument("--smoke", action="store_true",
                    help="fabricate a tiny real-shaped results df and render both "
                         "PNGs on CPU (no model, no GPU)")
    args = ap.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        # Reuse the analysis layer's fabricator so smoke data has ONE definition.
        from analysis_real import _fabricate_results
        results_dir = _fabricate_results(out_dir)
        print(f"[smoke] fabricated results in {results_dir}")

    written = make_real_figures(results_dir, out_dir, floor=args.floor,
                                n_boot=args.n_boot, ci_pct=args.ci_pct)
    print("rendered", len(written), "figures")


if __name__ == "__main__":
    main()
