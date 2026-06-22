#!/usr/bin/env /usr/bin/python3
# =============================================================================
# step2_eda.py
# Milestone 2 (FAITH-SAE) -- STEP 2 of 2: Exploratory Data Analysis (EDA).
# Author: Rajia Rani  ()
#
# WHAT THIS FILE DOES:
#   It LOADS the activation bank built by step1 and LOOKS AT IT before any model
#   touches it. "EDA" = Exploratory Data Analysis = the habit of computing simple
#   summaries and plots of your data so you understand its shape, scale, and
#   quirks before training. Analogy: a chef tastes and smells the ingredients
#   before cooking -- you never train a model on data you've never inspected.
#
#   Concretely it reports, all defined from zero in the README:
#     - per-dimension MEAN and VARIANCE (is each of the 768 dims centered? how
#       much does it wobble?),
#     - SPARSITY (what fraction of activation values are ~0 -- SAEs love sparse
#       inputs),
#     - TOKEN COUNT per image (a sanity check: should be 197),
#     - a 2-D PCA SCATTER (squash 768 dims down to 2 so we can SEE the data and
#       whether the planted concepts separate),
#   and finally builds a CLEAN-vs-OOD comparison so you literally see what a
#   distribution shift does to the activations.
#
# Run step1 first (it writes outputs/activations.npz). Then run this.
# =============================================================================

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend: write PNGs, never open a window.
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# ---- project-root import (see step1 for the explanation) --------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import load_config, set_seed   # noqa: E402

# Import the bank builder so we can make a SHIFTED twin for the OOD comparison.
import step1_build_synthetic_bank as step1     # noqa: E402


# =============================================================================
# Core EDA statistics.
# =============================================================================
def per_dimension_stats(acts2d: np.ndarray):
    """Per-dimension mean and variance over all (image,token) rows.

    `acts2d` is [n_rows, dim] where n_rows = n_images * n_tokens. We treat every
    token of every image as one sample of the dim-dimensional activation vector.

      mean[j]     = average value of dimension j across all rows.
                    Healthy activations are roughly CENTERED (mean ~ 0), because
                    a feature that is always-on carries no information.
      variance[j] = how much dimension j wobbles across rows.
                    A few HIGH-variance dims (the manifold's strong directions)
                    and many LOW-variance dims is the hallmark of real data
                    living on a low-dimensional manifold.
    """
    mean = acts2d.mean(axis=0)                  # [dim]
    var = acts2d.var(axis=0)                    # [dim]
    return mean, var


def sparsity_fraction(acts2d: np.ndarray, eps: float) -> float:
    """Fraction of activation VALUES whose magnitude is below `eps` (≈ zero).

    SPARSITY = "mostly zeros". If 90% of the numbers are ~0, the data is 90%
    sparse. Sparse Autoencoders (the next milestone) are built precisely to
    represent each input with only a few non-zero features, so knowing the raw
    input's natural sparsity is a useful baseline.

    Tiny example: values [0.01, 2.3, -0.02, 0.0, -1.7] with eps=0.1
      -> 3 of 5 are below 0.1 in magnitude -> sparsity = 0.60.
    """
    return float((np.abs(acts2d) < eps).mean())


def pca_2d(acts2d: np.ndarray, n_components: int = 2, seed: int = 0):
    """Reduce dim-D activations to n_components-D via PCA, for plotting.

    PCA (Principal Component Analysis) finds the directions of GREATEST variance
    in the data and re-expresses every point in those directions. The 1st
    component is the single direction along which the cloud is most stretched;
    the 2nd is the next-most (perpendicular to the 1st); and so on. Keeping the
    top 2 lets us draw a flat scatter that preserves as much spread as possible.

    Analogy: a 3-D object casts a 2-D shadow; PCA picks the camera angle that
    makes the shadow as informative (spread-out) as possible.

    Returns (coords [n_rows, n_components], explained_variance_ratio [n_components]).
    """
    pca = PCA(n_components=n_components, random_state=seed)
    coords = pca.fit_transform(acts2d)
    return coords, pca.explained_variance_ratio_


# =============================================================================
# Plotting: one multi-panel "EDA overview" figure.
# =============================================================================
def make_eda_figure(acts: np.ndarray, labels: np.ndarray, cfg: dict, out_path: str):
    """Build a 4-panel EDA overview PNG.

    Panels:
      (A) histogram of per-dimension VARIANCE -- shows the few-strong/many-weak
          spectrum of a low-rank manifold.
      (B) histogram of all activation VALUES -- shows the central peak near 0
          (the source of sparsity) and the spread.
      (C) per-image TOKEN COUNT bar -- sanity check that every image has the same
          number of tokens (197).
      (D) 2-D PCA SCATTER -- each point is one (image,token) activation, colored
          by whether concept #0 is present; if the planted concept is real, the
          two colors separate.
    """
    n_images, n_tokens, dim = acts.shape
    acts2d = acts.reshape(n_images * n_tokens, dim)

    mean, var = per_dimension_stats(acts2d)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    # (A) per-dimension variance histogram.
    ax[0, 0].hist(var, bins=cfg["hist_bins"], color="#3b7dd8", edgecolor="white")
    ax[0, 0].set_title("(A) Per-dimension variance\n(few strong dims = the manifold)")
    ax[0, 0].set_xlabel("variance of a dimension")
    ax[0, 0].set_ylabel("# of dimensions")

    # (B) distribution of all activation values.
    ax[0, 1].hist(acts2d.reshape(-1), bins=cfg["hist_bins"],
                  color="#5aa469", edgecolor="white")
    ax[0, 1].axvline(0.0, color="black", lw=1, ls="--")
    ax[0, 1].set_title("(B) All activation values\n(peak near 0 -> sparsity)")
    ax[0, 1].set_xlabel("activation value")
    ax[0, 1].set_ylabel("count")

    # (C) token count per image (should be flat at n_tokens).
    token_counts = np.full(n_images, n_tokens)
    ax[1, 0].bar(np.arange(min(n_images, 40)), token_counts[:40],
                 color="#d98a3b")
    ax[1, 0].axhline(n_tokens, color="black", lw=1, ls="--")
    ax[1, 0].set_title(f"(C) Tokens per image (first 40)\nall = {n_tokens} "
                       f"(196 patch + 1 CLS)")
    ax[1, 0].set_xlabel("image index")
    ax[1, 0].set_ylabel("# tokens")
    ax[1, 0].set_ylim(0, n_tokens * 1.2)

    # (D) 2-D PCA scatter, colored by concept #0 presence.
    # Subsample rows so the scatter is readable and fast.
    rng = np.random.default_rng(cfg["seed"])
    n_rows = acts2d.shape[0]
    n_plot = min(4000, n_rows)
    idx = rng.choice(n_rows, size=n_plot, replace=False)
    coords, evr = pca_2d(acts2d[idx], n_components=2, seed=cfg["seed"])
    # Map each sampled row back to its image's concept-0 label.
    img_of_row = (idx // n_tokens)
    color_label = labels[img_of_row, 0]                 # 1.0 / 0.0
    for val, col, name in [(1.0, "#c0392b", "concept #0 present"),
                           (0.0, "#7f8c8d", "concept #0 absent")]:
        m = color_label == val
        ax[1, 1].scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.5,
                         c=col, label=name)
    ax[1, 1].set_title(f"(D) 2-D PCA scatter\nPC1+PC2 hold "
                       f"{100*(evr[0]+evr[1]):.1f}% of variance")
    ax[1, 1].set_xlabel(f"PC1 ({100*evr[0]:.1f}% var)")
    ax[1, 1].set_ylabel(f"PC2 ({100*evr[1]:.1f}% var)")
    ax[1, 1].legend(loc="best", fontsize=8)

    fig.suptitle("FAITH-SAE M2 -- EDA of CLIP-shaped patch-token activations "
                 "(synthetic, offline)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return mean, var, evr


def make_ood_figure(clean_acts: np.ndarray, ood_acts: np.ndarray,
                    basis: np.ndarray, cfg: dict, out_path: str):
    """Side-by-side PCA of a CLEAN bank vs an OOD-SHIFTED bank.

    The point: a distribution shift MOVES and SPREADS the activation cloud and
    pushes mass OFF the clean manifold. We show:
      - left:  PCA of clean activations,
      - right: the SAME PCA axes applied to shifted activations (so the move is
               visible on a common frame),
      - and we print the "off-manifold residual" -- the fraction of each cloud's
        energy that lives OUTSIDE the clean manifold subspace. OOD inflates it.
    """
    def flat(a):
        return a.reshape(a.shape[0] * a.shape[1], a.shape[2])

    clean2d, ood2d = flat(clean_acts), flat(ood_acts)

    # Fit PCA on CLEAN data; project BOTH onto the same clean axes.
    pca = PCA(n_components=2, random_state=cfg["seed"])
    clean_xy = pca.fit_transform(clean2d)
    ood_xy = pca.transform(ood2d)

    # Off-manifold residual: how much energy is OUTSIDE the clean manifold basis.
    def off_manifold_frac(a2d):
        # project each row onto the clean basis, measure the leftover.
        coords = a2d @ basis                      # [rows, rank]
        on = coords @ basis.T                     # reconstruction inside manifold
        resid = np.linalg.norm(a2d - on, axis=1)
        total = np.linalg.norm(a2d, axis=1) + 1e-8
        return float((resid / total).mean())

    clean_off = off_manifold_frac(clean2d)
    ood_off = off_manifold_frac(ood2d)

    rng = np.random.default_rng(cfg["seed"])
    n_plot = min(3000, clean_xy.shape[0])
    idx = rng.choice(clean_xy.shape[0], size=n_plot, replace=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    ax[0].scatter(clean_xy[idx, 0], clean_xy[idx, 1], s=6, alpha=0.4, c="#2c7fb8")
    ax[0].set_title(f"CLEAN (in-distribution)\noff-manifold residual = {clean_off:.3f}")
    ax[1].scatter(ood_xy[idx, 0], ood_xy[idx, 1], s=6, alpha=0.4, c="#d95f0e")
    ax[1].set_title(f"OOD-SHIFTED (shift={cfg['ood_demo_shift']})\n"
                    f"off-manifold residual = {ood_off:.3f}")
    for a in ax:
        a.set_xlabel("PC1 (clean axes)")
        a.set_ylabel("PC2 (clean axes)")
    fig.suptitle("FAITH-SAE M2 -- what distribution shift does to activations "
                 "(same clean PCA frame)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return clean_off, ood_off


# =============================================================================
# Main entry point.
# =============================================================================
def main():
    cfg_path = os.path.join(_HERE, "config.yaml")
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])

    npz_path = os.path.join(_HERE, cfg["activations_npz"])
    if not os.path.exists(npz_path):
        print(f"ERROR: {npz_path} not found. Run step1_build_synthetic_bank.py first.")
        sys.exit(1)

    data = np.load(npz_path)
    acts = data["acts"]                         # [n_images, n_tokens, dim]
    labels = data["concept_labels"]             # [n_images, n_concepts]
    basis = data["basis"]                       # [dim, manifold_rank]
    n_images, n_tokens, dim = acts.shape
    acts2d = acts.reshape(n_images * n_tokens, dim)

    print("=" * 70)
    print("FAITH-SAE Milestone 2  --  STEP 2: EDA of the activation bank")
    print("=" * 70)
    print(f"Loaded: {os.path.relpath(npz_path, _HERE)}")
    print(f"  acts shape   : {acts.shape}   (images x tokens x dims)")
    print(f"  total samples: {acts2d.shape[0]:,} activation vectors of dim {dim}")
    print("-" * 70)

    # --- numeric EDA ---------------------------------------------------------
    mean, var = per_dimension_stats(acts2d)
    sparsity = sparsity_fraction(acts2d, cfg["sparsity_eps"])
    token_counts = np.full(n_images, n_tokens)

    print("PER-DIMENSION SUMMARY (over all image x token rows):")
    print(f"  mean of per-dim means     : {mean.mean():+.4f}  "
          f"(healthy ~ 0 => activations are centered)")
    print(f"  per-dim variance: min {var.min():.4f}  median {np.median(var):.4f}  "
          f"max {var.max():.4f}")
    print(f"  -> a few high-variance dims + many low = a low-rank manifold.")
    print(f"SPARSITY (|value| < {cfg['sparsity_eps']}): {100*sparsity:.1f}% of values ~ 0")
    print(f"TOKENS PER IMAGE: min {token_counts.min()}  max {token_counts.max()}  "
          f"(expect 197 = 196 patch + 1 CLS)")
    print(f"CONCEPT PREVALENCE (fraction of images per concept): "
          f"{labels.mean(axis=0).round(3).tolist()}")
    print("-" * 70)

    # --- per-dimension table to CSV -----------------------------------------
    df = pd.DataFrame({
        "dim_index": np.arange(dim),
        "mean": mean,
        "variance": var,
    }).sort_values("variance", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(_HERE, cfg["eda_summary_csv"])
    df.to_csv(csv_path, index=False)
    print(f"Per-dimension EDA table -> {os.path.relpath(csv_path, _HERE)}")
    print("  top-5 highest-variance dimensions (the manifold's strong directions):")
    print(df.head(5).to_string(index=False))
    print("-" * 70)

    # --- EDA overview figure -------------------------------------------------
    fig_eda = os.path.join(_HERE, cfg["fig_eda"])
    _, _, evr = make_eda_figure(acts, labels, cfg, fig_eda)
    print(f"PCA: PC1 holds {100*evr[0]:.1f}% of variance, "
          f"PC2 {100*evr[1]:.1f}% (top-2 = {100*(evr[0]+evr[1]):.1f}%).")
    print(f"EDA overview figure -> {os.path.relpath(fig_eda, _HERE)}")
    print("-" * 70)

    # --- clean-vs-OOD comparison --------------------------------------------
    # Rebuild a SHIFTED twin of the SAME bank (same seed, same concepts) so the
    # only difference is the OOD shift -- a clean controlled comparison.
    print(f"Building an OOD-shifted twin (shift={cfg['ood_demo_shift']}) for comparison...")
    ood_bank = step1.build_activation_bank(
        n_images=cfg["n_images"], n_tokens=cfg["n_tokens"], dim=cfg["dim"],
        manifold_rank=cfg["manifold_rank"], manifold_scale=cfg["manifold_scale"],
        noise_scale=cfg["noise_scale"], n_concepts=cfg["n_concepts"],
        concept_strength=cfg["concept_strength"],
        concept_prevalence=cfg["concept_prevalence"],
        ood_shift=cfg["ood_demo_shift"], seed=cfg["seed"],
    )
    fig_ood = os.path.join(_HERE, cfg["fig_ood"])
    clean_off, ood_off = make_ood_figure(acts, ood_bank["acts"], basis, cfg, fig_ood)
    print(f"OFF-MANIFOLD RESIDUAL (energy outside the clean manifold):")
    print(f"  clean bank : {clean_off:.3f}")
    print(f"  OOD bank   : {ood_off:.3f}   <- distribution shift pushes mass OFF the manifold")
    print(f"Clean-vs-OOD figure -> {os.path.relpath(fig_ood, _HERE)}")
    print("-" * 70)

    # --- success criterion ---------------------------------------------------
    # The "low-rank manifold" property does NOT show up in RAW per-dimension
    # variance (each of the 768 raw dims is a MIXTURE of all manifold directions,
    # so raw variances look flat). It shows up in PCA / COMPONENT space: the top
    # few principal components should capture the bulk of the variance. So we
    # test concentration there -- the top `manifold_rank` components should hold
    # most of the variance, far more than the same number of random dims would.
    rank = int(data["meta"][4]) if "meta" in data else cfg["manifold_rank"]
    pca_full = PCA(n_components=min(2 * rank, dim, acts2d.shape[0]),
                   random_state=cfg["seed"]).fit(acts2d)
    cum_evr = np.cumsum(pca_full.explained_variance_ratio_)
    top_rank_evr = float(cum_evr[min(rank, len(cum_evr)) - 1])   # variance in top `rank` PCs

    ok_shape = (n_tokens == cfg["n_tokens"] and dim == cfg["dim"])
    ok_centered = abs(float(mean.mean())) < 0.25
    ok_manifold = top_rank_evr > 0.80         # top-`rank` PCs hold >80% => low-rank
    ok_ood = ood_off > clean_off               # shift increased off-manifold residual
    all_ok = ok_shape and ok_centered and ok_manifold and ok_ood

    print("SUCCESS CRITERION:")
    print(f"  [{'PASS' if ok_shape else 'FAIL'}] shape matches CLIP ViT-B/16 "
          f"(197 tokens x 768 dims)")
    print(f"  [{'PASS' if ok_centered else 'FAIL'}] activations centered (|mean| < 0.25)")
    print(f"  [{'PASS' if ok_manifold else 'FAIL'}] low-rank manifold "
          f"(top-{rank} PCs hold {100*top_rank_evr:.1f}% of variance, want >80%)")
    print(f"  [{'PASS' if ok_ood else 'FAIL'}] OOD shift increases off-manifold "
          f"residual ({clean_off:.3f} -> {ood_off:.3f})")
    print("=" * 70)
    print("STEP 2 complete." if all_ok else "STEP 2 finished WITH WARNINGS (see FAIL above).")
    print("=" * 70)

    if not all_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# For research and educational purposes only.
# Author: Rajia Rani
# -----------------------------------------------------------------------------
