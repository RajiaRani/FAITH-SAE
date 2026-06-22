"""step2_estimate_clean_subspace.py — freeze the CLEAN sheet U_r + clean probes.

==============================================================================
WHAT THIS STEP DOES (in one sentence)
==============================================================================
On the CLEAN bank ONLY, it (a) finds the thin clean SHEET with PCA, keeps the
top `manifold_rank` directions as U_r, and FREEZES it; and (b) trains one linear
PROBE (a "ruler") per concept so we can read each concept off any activation.
BOTH are estimated on clean data and then reused UNCHANGED at every OOD rung --
which is exactly why a steer tuned on clean data can collapse under shift.

==============================================================================
TEACH-FROM-ZERO: every term, defined before it is used
==============================================================================

PRINCIPAL COMPONENT ANALYSIS (PCA) -- "find the main directions"
  Given a cloud of points, PCA finds the direction the cloud spreads out MOST
  (PC1), then the next-most-spread perpendicular direction (PC2), and so on. Each
  PC has a VARIANCE = how much the cloud spreads along it. The first few PCs trace
  the sheet; the rest have near-zero variance (the cloud is flat there).
  Tiny number: if PC1 explains 60% of the spread, PC2 30%, PC3 8%, rest 2%, the
  top 3 PCs capture 98% -- the sheet is ~3-D.

U_r (the FROZEN clean sheet basis)
  Stack the top-r principal components as the COLUMNS of a [dim, r] matrix U_r.
  Its columns are orthonormal (perpendicular, length 1). U_r IS our estimate of
  the CLEAN sheet's directions. We compute it ONCE here and FREEZE it.

P_M = U_r U_r^T  (project onto the clean sheet)
  Projecting v onto the sheet = the closest sheet point to v (drop a perpendicular).
    coordinates inside the sheet : c = U_r^T v      (r numbers)
    rebuild in the big space     : v_proj = U_r c = U_r (U_r^T v) = P_M v
  Properties: P_M is symmetric, P_M @ P_M = P_M (project twice = once), trace = r.

WHY U_r IS FROZEN ON CLEAN (the seed of OOD COLLAPSE)
  In real life you can only estimate the manifold on the clean data you HAVE; you
  cannot peek at every future shift. So U_r is fixed on clean and reused for every
  rung. When shifted activations drift OFF the clean sheet, P_M = U_r U_r^T no
  longer matches them -- the on-manifold projection is now pointing at the WRONG
  sheet. That growing mismatch is the headline mechanism milestone 6 measures.

A LINEAR PROBE (a "ruler" for a concept)
  A linear probe is a weight vector w (and a bias b) such that  w . a + b  is high
  when the concept is present and low when absent -- a ruler you hold up to an
  activation to read ONE concept off it. We train one per concept with logistic
  regression on the clean (activation, label) pairs from step1. The probes are
  also FROZEN on clean and reused at every rung (you calibrate your ruler on the
  data you have).
  Tiny number: if w points exactly along the concept direction d and b=0, then for
  an activation with the concept added you get a clearly higher w.a than for one
  without -- the ruler separates present from absent.

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step2_estimate_clean_subspace.py
Reads  outputs/clean_acts.npy, labels.npy, sheet_basis.npy (step1).
Writes outputs/U_r.npy           ([dim, r] FROZEN clean sheet basis),
       outputs/probe_weights.npy ([n_concepts, dim] one ruler per concept),
       outputs/probe_bias.npy    ([n_concepts] the rulers' offsets).
"""
from __future__ import annotations

import numpy as np

from _common import banner, load_cfg, outpath


def estimate_U_r(acts: np.ndarray, r: int):
    """PCA the CLEAN bank and return (U_r [dim, r], explained_variance_ratio).

    sklearn's PCA centers the data and returns components_ as ROWS; we transpose
    so the top-r PCs become the COLUMNS of U_r (the brief's convention).
    """
    from sklearn.decomposition import PCA
    dim = acts.shape[1]
    r = min(int(r), dim)
    pca_full = PCA(n_components=dim, svd_solver="full")
    pca_full.fit(acts)
    evr = pca_full.explained_variance_ratio_          # [dim], sums to 1
    U_r = pca_full.components_[:r].T.copy()           # [dim, r], columns = top-r PCs
    return U_r.astype(np.float32), evr


def subspace_overlap(U_r: np.ndarray, B_true: np.ndarray) -> float:
    """How well does the estimated clean sheet U_r cover the TRUE planted sheet?

    Project each true direction onto the estimated subspace and measure how much
    length survives (1.0 = fully captured); average over true directions. Only
    computable in the synthetic setting where we planted B_true.
    """
    P = U_r @ U_r.T
    captured = [np.linalg.norm(P @ B_true[:, j]) / (np.linalg.norm(B_true[:, j]) + 1e-12)
                for j in range(B_true.shape[1])]
    return float(np.mean(captured))


def train_probes(acts: np.ndarray, labels: np.ndarray):
    """Train one logistic-regression PROBE per concept on the CLEAN bank.

    Returns (W [n_concepts, dim], b [n_concepts]). Each probe is a ruler:
    w . a + b reads ONE concept off an activation. We use the RAW linear score
    (not the squashed probability) downstream because it is smooth and unbounded --
    exactly what we want to watch rise as the steering knob turns.
    """
    from sklearn.linear_model import LogisticRegression
    n_c = labels.shape[1]
    W, b = [], []
    for c in range(n_c):
        y = labels[:, c].astype(int)
        if y.min() == y.max():            # degenerate (all one class) -> zero ruler
            W.append(np.zeros(acts.shape[1], dtype=np.float32)); b.append(0.0); continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(acts, y)
        W.append(clf.coef_[0].astype(np.float32))
        b.append(float(clf.intercept_[0]))
    return np.stack(W).astype(np.float32), np.asarray(b, dtype=np.float32)


def main() -> None:
    cfg = load_cfg()
    r = int(cfg["manifold_rank"])
    banner("STEP 2 — FREEZE the clean sheet U_r (PCA) + train clean probes")

    acts = np.load(outpath("clean_acts.npy"))
    labels = np.load(outpath("labels.npy"))
    print(f"  loaded clean acts {acts.shape}, labels {labels.shape} "
          f"(rerun step1 if missing)")
    print(f"  keeping the top r = manifold_rank = {r} clean PCs as U_r")

    # ---- (a) estimate + freeze the clean sheet U_r --------------------------
    U_r, evr = estimate_U_r(acts, r)
    print(f"\n  U_r shape = {U_r.shape}  (each COLUMN is one clean-sheet direction)")
    cum = float(np.sum(evr[:r]))
    true_r = int(cfg["true_manifold_rank"])
    print(f"  variance explained by the top {r} PCs       = {cum * 100:.1f}%")
    print(f"  variance explained by the first {true_r} PCs = "
          f"{float(np.sum(evr[:true_r])) * 100:.1f}%  (true clean sheet is {true_r}-D)")
    try:
        B_true = np.load(outpath("sheet_basis.npy"))
        ov = subspace_overlap(U_r, B_true)
        print(f"  SELF-GRADE (synthetic only): U_r captures {ov * 100:.1f}% of the "
              f"TRUE planted clean sheet (~100% => PCA found it).")
    except FileNotFoundError:
        print("  (no planted sheet_basis.npy -> skipping self-grade; normal for a REAL run.)")

    # projection identities, for the reader's trust
    P = U_r @ U_r.T
    print(f"  projection check: P_M @ P_M - P_M max error = "
          f"{float(np.max(np.abs(P @ P - P))):.2e}  (~0 => project twice = once)")
    print(f"                    trace(P_M) = {float(np.trace(P)):.2f}  (should equal r = {r})")

    # ---- (b) train + freeze the clean concept probes (the rulers) -----------
    W, b = train_probes(acts, labels)
    # quick accuracy readout so the reader trusts the rulers actually work
    accs = []
    for c in range(W.shape[0]):
        score = acts @ W[c] + b[c]
        pred = (score > 0).astype(np.float32)
        accs.append(float((pred == labels[:, c]).mean()))
    print(f"\n  trained {W.shape[0]} clean probes (rulers); "
          f"mean clean train accuracy = {float(np.mean(accs)):.3f} "
          f"(per concept: {[round(a, 2) for a in accs]})")

    np.save(outpath("U_r.npy"), U_r)
    np.save(outpath("probe_weights.npy"), W)
    np.save(outpath("probe_bias.npy"), b)
    print(f"\n  saved -> {outpath('U_r.npy')}           (FROZEN clean sheet; reused at EVERY rung)")
    print(f"  saved -> {outpath('probe_weights.npy')} (FROZEN clean rulers; reused at EVERY rung)")
    print(f"  saved -> {outpath('probe_bias.npy')}")
    print("\nSTEP 2 done. Next: step3 corrupts the bank rung by rung and MEASURES CFS.")


# REAL RUN (M6): estimate U_r ONCE from the large real CLEAN CLIP bank, and train
# the probes on real (activation, concept-label) pairs (or use SAE-discovered
# concepts). Both are frozen on clean and reused at every OOD shift level. The
# synthetic self-grade block (subspace_overlap) is dropped -- there is no planted
# ground-truth clean sheet in real life.
if __name__ == "__main__":
    main()
