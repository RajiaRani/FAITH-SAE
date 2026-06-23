#!/usr/bin/env python
# =============================================================================
# make_figures.py  --  turn the real result CSVs into the paper's figures.
# No GPU needed. Run on the login node:
#     conda activate sae_research
#     python make_figures.py
# Then download every fig*_REAL.png to your Mac with scp.
# =============================================================================
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    if not os.path.exists(path):
        print(f"  (missing {path} -- skipping its figure)")
        return None
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]                       # header, data rows


made = []

# ---- Fig 1 : faithfulness vs OOD severity (M6) --------------------------------
r = read_csv("faith_sae_severity_results.csv")     # severity,naive_offman,naive_ci,onman_offman,onman_ci,...
if r:
    _, d = r
    sev = [row[0] for row in d]
    nv = [float(row[1]) for row in d]; nvc = [float(row[2]) for row in d]
    om = [float(row[3]) for row in d]; omc = [float(row[4]) for row in d]
    x = range(len(sev))
    plt.figure(figsize=(6, 4))
    plt.errorbar(x, nv, yerr=nvc, marker="o", capsize=3, label="naive steering")
    plt.errorbar(x, om, yerr=omc, marker="s", capsize=3, label="on-manifold steering")
    plt.xticks(list(x), sev); plt.xlabel("distribution-shift severity")
    plt.ylabel("downstream off-manifold fraction\n(lower = more faithful)")
    plt.title("Fig 1 — faithfulness vs OOD severity"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("fig1_severity_REAL.png", dpi=200); plt.close()
    made.append("fig1_severity_REAL.png")

# ---- Fig 2 : on-manifold CFS vs manifold rank (M4) ----------------------------
r = read_csv("faith_sae_sweep_results.csv")        # rank,domain,naive_cfs,naive_ci,onman_cfs,onman_ci,...
if r:
    _, d = r
    ranks, naive, onman, onci = [], [], [], []
    for row in d:
        if row[1] == "clean":
            ranks.append(int(row[0])); naive.append(float(row[2]))
            onman.append(float(row[4])); onci.append(float(row[5]))
    plt.figure(figsize=(6, 4))
    plt.axhline(naive[0], color="C0", ls="--", label="naive (ignores manifold)")
    plt.errorbar(ranks, onman, yerr=onci, marker="s", color="C1", capsize=3, label="on-manifold")
    plt.xscale("log", base=2); plt.xticks(ranks, ranks)
    plt.xlabel("manifold rank r"); plt.ylabel("CFS (higher = more faithful)")
    plt.title("Fig 2 — on-manifold CFS vs manifold rank")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("fig2_rank_sweep_REAL.png", dpi=200); plt.close()
    made.append("fig2_rank_sweep_REAL.png")

# ---- Fig 3 : CLIP faithfulness by steering direction (M8) -- THE KEY FIGURE ----
r = read_csv("faith_sae_control_results.csv")      # domain,naive,naive_ci,onmanifold,onman_ci,diffmeans,diffmeans_ci
if r:
    _, d = r
    doms = [row[0] for row in d]
    nv = [float(row[1]) for row in d]; nvc = [float(row[2]) for row in d]
    om = [float(row[3]) for row in d]; omc = [float(row[4]) for row in d]
    dm = [float(row[5]) for row in d]; dmc = [float(row[6]) for row in d]
    x = np.arange(len(doms)); w = 0.25
    plt.figure(figsize=(6, 4))
    plt.bar(x - w, nv, w, yerr=nvc, capsize=3, label="naive (SAE)")
    plt.bar(x,     om, w, yerr=omc, capsize=3, label="on-manifold")
    plt.bar(x + w, dm, w, yerr=dmc, capsize=3, label="diff-of-means (TCAV)")
    plt.axhline(0, color="k", lw=0.8)
    plt.xticks(x, doms); plt.ylabel("semantic faithfulness (cosine)")
    plt.title("Fig 3 — CLIP faithfulness by steering direction")
    plt.legend(); plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig("fig3_clip_faithfulness_REAL.png", dpi=200); plt.close()
    made.append("fig3_clip_faithfulness_REAL.png")

# ---- Fig 4 : downstream realism, naive vs on-manifold (M5) --------------------
r = read_csv("faith_sae_downstream_results.csv")   # domain,variant,eff,on_eff,off_eff,offman,offman_ci
if r:
    _, d = r
    labels = [f"{row[0]}\n{row[1]}" for row in d]
    off = [float(row[5]) for row in d]; ci = [float(row[6]) for row in d]
    x = np.arange(len(labels))
    plt.figure(figsize=(6, 4))
    plt.bar(x, off, yerr=ci, capsize=3, color=["C0", "C1", "C0", "C1"])
    plt.xticks(x, labels); plt.ylabel("downstream off-manifold fraction\n(lower = more faithful)")
    lo = min(off) - 0.02; plt.ylim(max(0, lo), max(off) + 0.02)
    plt.title("Fig 4 — downstream realism of the output change")
    plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig("fig4_downstream_REAL.png", dpi=200); plt.close()
    made.append("fig4_downstream_REAL.png")

print("\nfigures written:")
for m in made:
    print("   -", m)
print(f"\n{len(made)} real figures ready. Download them to your Mac with scp.")
