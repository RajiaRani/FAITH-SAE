"""step2_estimate_subspace.py — find the sheet with PCA -> U_r and P_M.

==============================================================================
WHAT THIS STEP DOES (in one sentence)
==============================================================================
It takes the activation bank from step1 and discovers the thin SHEET the points
live on, using PCA, keeping the top `manifold_rank` directions as the columns of
a matrix U_r; then it builds the projection matrix P_M = U_r U_r^T.

==============================================================================
TEACH-FROM-ZERO: every term, defined before it is used
==============================================================================

SUBSPACE
  A flat slice through the big space that passes through the origin: a line, a
  plane, or a higher-dim "flat" inside the room. Our sheet, to first
  approximation, IS a subspace — the span of a few directions.
  Analogy: inside a 3-D room, the floor is a 2-D subspace and a single beam of
  light is a 1-D subspace.
  Tiny number: in 64-D, the span of 16 chosen directions is a 16-dimensional
  subspace; a point in it needs only 16 coordinates, not 64.

PRINCIPAL COMPONENT ANALYSIS (PCA) — "find the main directions"
  Given a cloud of points, PCA finds the direction the cloud spreads out MOST
  (principal component 1, PC1), then the next-most-spread direction
  perpendicular to it (PC2), and so on. Each PC comes with a variance = how much
  the cloud spreads along it. The first few PCs trace out the sheet; the rest
  have near-zero variance (the cloud is flat in those directions).
  Analogy: a thin frisbee floating in a room. PC1 and PC2 lie in the frisbee's
  flat face (lots of spread); PC3 is the frisbee's thin axis (almost no spread).
  Keeping PC1+PC2 keeps the frisbee; dropping PC3 drops only the thinness.
  Tiny number: if PC1 explains 60% of the spread, PC2 30%, PC3 8%, and the rest
  ~2%, then the top 3 PCs capture 98% of the data — the sheet is ~3-D.

U_r  (the estimated sheet basis)
  Stack the top-r principal components as the COLUMNS of a [dim, r] matrix.
  Its columns are orthonormal (perpendicular, length 1). U_r IS our estimate of
  the sheet's directions. In real life this is all you get; the true sheet is
  unknown.

PROJECTION  and  P_M = U_r U_r^T
  Projecting a vector v onto the subspace = finding the closest point to v that
  lies IN the subspace (drop a perpendicular onto it).
  * coordinates of v inside the subspace : c = U_r^T v   (r numbers)
  * rebuild that point in the big space   : v_proj = U_r c = U_r (U_r^T v)
  * so the one matrix that does both at once is  P_M = U_r U_r^T  (a [dim, dim]
    matrix), and  v_proj = P_M v.
  Properties you can rely on: P_M is symmetric, and P_M @ P_M = P_M (projecting
  twice changes nothing — you are already on the sheet). The trace of P_M equals
  r (it "passes through" r dimensions).
  Tiny number (the 2-D example again): sheet = x-axis, U_r = [[1],[0]].
    P_M = U_r U_r^T = [[1,0],[0,1*0... ]] = [[1,0],[0,0]].
    P_M @ (2,1) = (2,0)  -> keeps x, zeroes y. Exactly "project onto the x-axis".

THE RANK r KNOB
  r = manifold_rank = how many PCs we keep = the dimension of the sheet we trust.
  * r too small  -> we throw away real sheet directions; the edit can't move the
                    concept (effect dies; over-constrained).
  * r about right -> we keep the whole sheet, nothing more; edits stay realistic.
  * r -> dim      -> we keep EVERYTHING; P_M becomes the identity I; projecting
                    does nothing; on-manifold steering DEGENERATES into naive
                    steering. (Brief §14: naive = the r -> inf, P_M = I case.)

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step2_estimate_subspace.py
Reads  outputs/real_bank.npy (+ outputs/sheet_basis.npy to self-grade).
Writes outputs/U_r.npy  (the [dim, r] estimated sheet basis, used by every
       steerer in step3 as the FIXED real-image `basis`).
"""
from __future__ import annotations

import numpy as np

from _common import banner, load_cfg, outpath


def estimate_U_r(bank: np.ndarray, r: int):
    """PCA the bank and return (U_r [dim, r], explained_variance_ratio [dim]).

    We use scikit-learn's PCA (the textbook "find the main directions" tool).
    PCA centers the data (subtracts the mean) and returns components_ = the PCs
    as ROWS; we transpose so the PCs become COLUMNS of U_r (the convention in the
    brief: U_r is [dim, r] with directions in columns).
    """
    from sklearn.decomposition import PCA
    dim = bank.shape[1]
    r = min(int(r), dim)
    # Fit FULL PCA so we can also see how variance falls off past r (the knee).
    pca_full = PCA(n_components=dim, svd_solver="full")
    pca_full.fit(bank)
    evr = pca_full.explained_variance_ratio_          # [dim], sums to 1
    U_r = pca_full.components_[:r].T.copy()           # [dim, r], columns = top-r PCs
    return U_r.astype(np.float32), evr


def subspace_overlap(U_r: np.ndarray, B_true: np.ndarray) -> float:
    """How well does the estimated sheet U_r cover the TRUE sheet B_true?

    For each true direction, project it onto the estimated subspace and measure
    how much length survives (1.0 = fully captured). Average over true
    directions. This is only computable in the synthetic setting where we planted
    B_true; in a real run there is no ground truth to compare against.
    """
    # P_M projects onto the estimated subspace.
    P = U_r @ U_r.T                                   # [dim, dim]
    captured = []
    for j in range(B_true.shape[1]):
        b = B_true[:, j]
        proj = P @ b
        captured.append(np.linalg.norm(proj) / (np.linalg.norm(b) + 1e-12))
    return float(np.mean(captured))


def main() -> None:
    cfg = load_cfg()
    r = int(cfg["manifold_rank"])
    banner("STEP 2 — estimate the on-manifold subspace U_r by PCA")

    bank = np.load(outpath("real_bank.npy"))
    print(f"  loaded bank shape = {bank.shape}  (rerun step1 if this is missing)")
    print(f"  keeping the top r = manifold_rank = {r} principal components as U_r")

    U_r, evr = estimate_U_r(bank, r)
    print(f"\n  U_r shape = {U_r.shape}  (each COLUMN is one estimated sheet direction)")

    # How much of the data's spread lives in the top-r directions?
    cum = float(np.sum(evr[:r]))
    print(f"  variance explained by the top {r} PCs = {cum * 100:.1f}% of the total")
    true_r = int(cfg["true_manifold_rank"])
    print(f"  variance explained by the first {true_r} PCs = "
          f"{float(np.sum(evr[:true_r])) * 100:.1f}%  "
          f"(should be high: the true sheet is {true_r}-D)")
    print(f"  variance in PCs beyond #{true_r} (the off-sheet wobble) = "
          f"{float(np.sum(evr[true_r:])) * 100:.1f}%  (should be small)")

    # Self-grade against the planted sheet (synthetic only).
    try:
        B_true = np.load(outpath("sheet_basis.npy"))
        overlap = subspace_overlap(U_r, B_true)
        print(f"\n  SELF-GRADE (synthetic only): estimated U_r captures "
              f"{overlap * 100:.1f}% of the TRUE planted sheet "
              f"(~100% => PCA found the right sheet).")
    except FileNotFoundError:
        print("\n  (no planted sheet_basis.npy found -> skipping self-grade; "
              "this is normal for a REAL run.)")

    # Verify the projection identities on a tiny check, for the reader's trust.
    P = U_r @ U_r.T
    idempotent_err = float(np.max(np.abs(P @ P - P)))
    trace_P = float(np.trace(P))
    print(f"\n  projection check: P_M = U_r U_r^T  (shape {P.shape})")
    print(f"    P_M @ P_M - P_M max error = {idempotent_err:.2e}  (~0 => projecting twice = once)")
    print(f"    trace(P_M) = {trace_P:.2f}  (should equal r = {r}: passes through r dims)")

    np.save(outpath("U_r.npy"), U_r)
    print(f"\n  saved -> {outpath('U_r.npy')}  (the fixed real-image basis every steerer uses in step3)")
    print("\nSTEP 2 done. Next: step3 steers a concept and measures the off-manifold residual.")


# REAL RUN (M4): estimate U_r ONCE from the large real CLIP activation bank
# (step1's real_bank.npy). PCA over hundreds of thousands of CLIP ViT-B/16 patch
# activations is a one-time cost; cache U_r.npy and reuse it for every steerer,
# every concept, and every OOD shift level. The synthetic self-grade block
# (subspace_overlap) is dropped — there is no planted ground-truth sheet to grade
# against in real life.
if __name__ == "__main__":
    main()
