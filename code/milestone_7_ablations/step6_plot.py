"""step6_plot.py — draw the multi-panel figure outputs/ablations.png (one panel per ablation).

==============================================================================
WHAT THIS STEP DOES (in one sentence)
==============================================================================
It reads outputs/ablations.csv and renders a 5-panel PNG — one panel per ablation
A1..A5 — each plotting the measured Causal Faithfulness Score (CFS) against that
ablation's knob (with the off-manifold naive reference and the relevant diagnostic).

==============================================================================
HOW TO READ THE PICTURE (the one-glance takeaways)
==============================================================================
  * A1 (SAE type): two bars per steerer (topk vs l1). Taller on-manifold bar =
    that SAE family steers more faithfully.
  * A2 (TopK k): CFS-vs-k curve. The PEAK is the sparsity sweet spot; too small k
    starves the effect, too large k blurs (polysemantic).
  * A3 (rank r): the CORE curve. CFS rises, PEAKS near the true sheet rank, then
    declines toward naive as r -> dim (P_M -> I). The peak is the KNEE — the sweet
    spot. A green dashed line marks the true sheet rank.
  * A4 (threshold tau): two lines — mean CFS of survivors (rises with tau) and the
    reliable-concept fraction (falls with tau). Stricter filter = fewer but more
    faithful concepts (the ~10-15% tail).
  * A5 (layer|token): bars per attachment point. Cleaner LATE layer + per-PATCH
    tokens are the tallest = the best place to attach the SAE.
In every panel the on-manifold (ours) series sits ABOVE the naive reference — the
projection is what buys the faithfulness, and the ablation isolates how each knob
moves it.

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step6_plot.py
Reads outputs/ablations.csv. Writes outputs/ablations.png.
"""
from __future__ import annotations

import csv
from collections import defaultdict

from _common import banner, load_cfg, outpath

# stable colours: on-manifold (ours) in green, naive reference in red.
COL = {"onmanifold_steer": "#2ca02c", "naive_steer": "#d62728"}


def read_rows():
    rows = []
    with open(outpath("ablations.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _series(rows, ablation_id, variant):
    """Return (knob_values, cfs, diagnostic) for one (ablation, steerer), in order."""
    sub = [r for r in rows if r["ablation_id"] == ablation_id
           and r["variant"] == variant]
    xs = [r["knob_value"] for r in sub]
    cfs = [float(r["cfs"]) for r in sub]
    diag = [float(r["diagnostic"]) if r["diagnostic"] != "" else float("nan")
            for r in sub]
    return xs, cfs, diag


def main() -> None:
    cfg = load_cfg()
    banner("STEP 6 — render outputs/ablations.png (one panel per ablation)")

    import matplotlib
    matplotlib.use("Agg")                       # headless: no display needed
    import matplotlib.pyplot as plt

    rows = read_rows()
    variants = cfg["ablation_variants"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axA1, axA2, axA3, axA4, axA5, axTxt = axes.ravel()

    # ---- A1: SAE type — grouped bars (topk vs l1) per steerer ---------------
    types = list(cfg["a1_sae_types"])
    width = 0.35
    for i, v in enumerate(variants):
        xs, cfs, _ = _series(rows, "A1", v)
        order = {t: c for t, c in zip(xs, cfs)}
        vals = [order.get(t, 0.0) for t in types]
        pos = [j + (i - 0.5) * width for j in range(len(types))]
        axA1.bar(pos, vals, width=width, color=COL.get(v, "#888"),
                 label=v.replace("_steer", ""))
        for p, val in zip(pos, vals):
            axA1.text(p, val + 0.01, f"{val:.2f}", ha="center", fontsize=8)
    axA1.set_xticks(range(len(types))); axA1.set_xticklabels(types)
    axA1.set_ylim(0, 1.0); axA1.set_ylabel("CFS"); axA1.set_xlabel("SAE type")
    axA1.set_title("A1 — SAE type (TopK vs L1)"); axA1.legend(fontsize=8)

    # ---- A2: TopK k — CFS vs k -----------------------------------------------
    for v in variants:
        xs, cfs, _ = _series(rows, "A2", v)
        kx = [int(x) for x in xs]
        axA2.plot(kx, cfs, "o-", color=COL.get(v, "#888"),
                  label=v.replace("_steer", ""))
    om_x, om_c, _ = _series(rows, "A2", "onmanifold_steer")
    if om_c:
        bi = max(range(len(om_c)), key=lambda j: om_c[j])
        axA2.scatter([int(om_x[bi])], [om_c[bi]], s=180, facecolors="none",
                     edgecolors="#2ca02c", linewidths=2, zorder=5)
        axA2.annotate("sweet spot", (int(om_x[bi]), om_c[bi]),
                      textcoords="offset points", xytext=(6, 8), fontsize=8,
                      color="#2ca02c")
    axA2.set_xscale("log", base=2); axA2.set_ylim(0, 1.0)
    axA2.set_xlabel("k (features ON at once)"); axA2.set_ylabel("CFS")
    axA2.set_title("A2 — TopK sparsity k"); axA2.legend(fontsize=8); axA2.grid(alpha=0.3)

    # ---- A3: projection rank r — CFS vs r (the CORE knee) --------------------
    for v in variants:
        xs, cfs, _ = _series(rows, "A3", v)
        rx = [int(x) for x in xs]
        axA3.plot(rx, cfs, "o-", color=COL.get(v, "#888"),
                  label=v.replace("_steer", ""))
    om_x, om_c, _ = _series(rows, "A3", "onmanifold_steer")
    if om_c:
        bi = max(range(len(om_c)), key=lambda j: om_c[j])
        axA3.scatter([int(om_x[bi])], [om_c[bi]], s=180, facecolors="none",
                     edgecolors="#2ca02c", linewidths=2, zorder=5)
        axA3.annotate("knee / sweet spot", (int(om_x[bi]), om_c[bi]),
                      textcoords="offset points", xytext=(6, -14), fontsize=8,
                      color="#2ca02c")
    axA3.axvline(int(cfg["true_manifold_rank"]), ls="--", color="green", alpha=0.5)
    axA3.text(int(cfg["true_manifold_rank"]) * 1.05, 0.05, "true sheet rank",
              color="green", fontsize=8, rotation=90)
    axA3.set_xscale("log", base=2); axA3.set_ylim(0, 1.0)
    axA3.set_xlabel("projection rank r"); axA3.set_ylabel("CFS")
    axA3.set_title("A3 — manifold-projection rank r (core knob)")
    axA3.legend(fontsize=8); axA3.grid(alpha=0.3)

    # ---- A4: selection threshold — mean CFS (up) + kept fraction (down) ------
    xs, cfs, frac = _series(rows, "A4", "onmanifold_steer")
    tx = [float(x) for x in xs]
    axA4.plot(tx, cfs, "s-", color="#2ca02c", label="mean CFS of survivors")
    axA4.set_ylim(0, 1.0); axA4.set_xlabel("selection threshold tau")
    axA4.set_ylabel("mean CFS (kept concepts)", color="#2ca02c")
    axA4b = axA4.twinx()
    axA4b.plot(tx, frac, "^--", color="#9467bd", label="reliable-concept fraction")
    axA4b.set_ylim(0, 1.05); axA4b.set_ylabel("kept fraction", color="#9467bd")
    axA4.set_title("A4 — concept-selection threshold")
    l1, lab1 = axA4.get_legend_handles_labels()
    l2, lab2 = axA4b.get_legend_handles_labels()
    axA4.legend(l1 + l2, lab1 + lab2, fontsize=7, loc="center left")
    axA4.grid(alpha=0.3)

    # ---- A5: layer|token — grouped bars per attachment point -----------------
    tags = [f"{c['layer']}|{c['token']}" for c in cfg["a5_layer_tokens"]]
    for i, v in enumerate(variants):
        xs, cfs, _ = _series(rows, "A5", v)
        order = {t: c for t, c in zip(xs, cfs)}
        vals = [order.get(t, 0.0) for t in tags]
        pos = [j + (i - 0.5) * width for j in range(len(tags))]
        axA5.bar(pos, vals, width=width, color=COL.get(v, "#888"),
                 label=v.replace("_steer", ""))
    axA5.set_xticks(range(len(tags)))
    axA5.set_xticklabels(tags, rotation=20, ha="right", fontsize=8)
    axA5.set_ylim(0, 1.0); axA5.set_ylabel("CFS")
    axA5.set_title("A5 — backbone layer & token"); axA5.legend(fontsize=8)

    # ---- text panel: what each ablation isolates -----------------------------
    axTxt.axis("off")
    axTxt.text(
        0.0, 1.0,
        "Each panel turns ONE knob, holds all else fixed:\n\n"
        "A1  SAE type (TopK vs L1) — which sparsity recipe steers\n"
        "      more faithfully (diagnostic: reconstruction MSE).\n\n"
        "A2  TopK k — the sparsity sweet spot; too small starves\n"
        "      the effect, too large blurs (polysemantic).\n\n"
        "A3  rank r (CORE) — CFS peaks at the true sheet rank;\n"
        "      r->dim degenerates on-manifold into naive.\n\n"
        "A4  selection threshold — stricter filter = fewer but\n"
        "      more faithful concepts (the ~10-15% reliable tail).\n\n"
        "A5  layer & token — late layer + patch tokens steer best.\n\n"
        "Green = on-manifold (ours); Red = naive (off-manifold)\n"
        "reference. Ours sits above naive in every panel.",
        va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle("FAITH-SAE M7: five ablations A1-A5 — how CFS responds when ONE knob "
                 "turns (illustrative offline synthetic run)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = outpath("ablations.png")
    fig.savefig(out, dpi=200)
    print(f"  saved -> {out}")
    print("\nSTEP 6 done. Open the PNG: one panel per ablation, ours (green) above naive (red).")


# REAL RUN (M7): same 5-panel layout, real numbers — the offline analog of the
# paper's ablation table / fig5 design grid.
if __name__ == "__main__":
    main()
