"""step1_build_bank.py — regenerate the synthetic REAL-IMAGE activation bank.

==============================================================================
WHAT THIS STEP DOES (in one sentence)
==============================================================================
It manufactures a pile of pretend "real-image activations" that live on a thin
low-dimensional SHEET inside a big space — so the next step can rediscover that
sheet with PCA. It also prints a tiny 2-D worked example of "on-manifold vs
off-manifold" you can check by hand.

==============================================================================
TEACH-FROM-ZERO: every term used below, defined before it is used
==============================================================================

ACTIVATION
  A vector of numbers a neural network produces inside itself while looking at
  an input. Think of it as the model's private notes about one image-patch.
  In real CLIP ViT-B/16 each note is 768 numbers long; here we use dim=64 so a
  laptop CPU runs it instantly. One activation = one point in a 64-dimensional
  space (a list of 64 numbers like [0.3, -1.2, ...]).

SPACE (the "big space")
  All possible 64-number lists. Picture a room, but with 64 directions instead
  of the usual 3 (left/right, up/down, forward/back). Every activation is one
  dot somewhere in that room.

THE DATA MANIFOLD (the "sheet")
  Real activations do NOT fill the whole room. A frozen model, fed real images,
  only ever lands in a thin curved SHEET inside the room — like a sheet of paper
  floating in a gym. The paper is 2-D even though the gym is 3-D. The model
  was "trained on the paper", so it only behaves sensibly for points ON the
  paper. Points off the paper (mid-air) are activations the model has never
  really seen and does not handle reliably.
  Analogy: handwriting. The set of all 28x28 pixel grids is enormous, but real
  handwritten digits occupy a tiny curved sheet inside it; random pixel noise
  (off the sheet) looks like static, not a digit.
  Tiny number: if the sheet is 8-dimensional inside a 64-dimensional room, then
  56 of the 64 directions are "off the sheet" — real activations have ~zero
  spread along those 56 directions.

WHY WE PLANT A KNOWN SHEET
  In real life we don't KNOW the sheet — we estimate it. Here we deliberately
  BUILD the bank to live on a known `true_manifold_rank`-dimensional sheet (plus
  a hair of off-sheet noise), so in step2 we can check that PCA recovers it.
  This is a controlled experiment: plant the answer, then test the method.

HOW WE BUILD ONE REAL-IMAGE ACTIVATION
  1. Pick `true_manifold_rank` fixed orthonormal "sheet directions" B (a basis
     for the sheet). Orthonormal = mutually perpendicular, each length 1.
  2. Draw random coordinates c ALONG the sheet (one random number per sheet
     direction). a_sheet = B @ c  lands exactly ON the sheet.
  3. Add a tiny Gaussian wobble across ALL 64 directions (noise_off_manifold)
     so the point sits a hair OFF the sheet — no real data is perfectly flat.
  Result: a bank of points that hug an 8-dim sheet inside a 64-dim room.

==============================================================================
THE 2-D WORKED EXAMPLE (do this by hand once; it is the whole idea in miniature)
==============================================================================
Shrink the room to 2-D (just an x-axis and a y-axis) and let the "sheet" be the
x-axis line (1-D sheet inside a 2-D room).
  * ON-manifold point :  P = (2.0, 0.0)         -- sits on the x-axis line.
  * Edit pushed ALONG the sheet :  Delta_on = (1.0, 0.0)
        P + Delta_on = (3.0, 0.0)   -- STILL on the line. Realistic.
  * Edit pushed OFF the sheet :  Delta_off = (0.0, 1.0)
        P + Delta_off = (2.0, 1.0)  -- floats OFF the line into empty air.
The projection onto the sheet (the x-axis) just keeps the x-part and zeroes the
y-part:  project((a, b)) = (a, 0).
  * project(Delta_on)  = (1.0, 0.0)  -> unchanged: it was already on the sheet.
  * project(Delta_off) = (0.0, 0.0)  -> killed: it was entirely off the sheet.
The OFF-MANIFOLD RESIDUAL = ||Delta - project(Delta)|| / ||Delta||:
  * for Delta_on  : ||(1,0)-(1,0)|| / ||(1,0)|| = 0/1 = 0.0   (fully on-sheet)
  * for Delta_off : ||(0,1)-(0,0)|| / ||(0,1)|| = 1/1 = 1.0   (fully off-sheet)
That single number 0.0-vs-1.0 is exactly what milestone 4 measures, just in 64-D
instead of 2-D. step1 prints this example so you can verify it yourself.

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step1_build_bank.py
Writes outputs/real_bank.npy  (the [bank_size, dim] activation bank) and
       outputs/sheet_basis.npy (the TRUE sheet basis B, for step2 to check
                                its PCA estimate against).
"""
from __future__ import annotations

import numpy as np

from _common import banner, load_cfg, outpath


def random_orthonormal_basis(dim: int, rank: int, seed: int) -> np.ndarray:
    """Return a [dim, rank] matrix whose columns are orthonormal (the sheet's
    directions). We draw a random matrix and orthonormalize it with QR.

    Orthonormal recap: each column has length 1 and any two columns are
    perpendicular (dot product 0). QR factorization hands us exactly that.
    """
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((dim, rank))
    Q, _ = np.linalg.qr(M)            # Q columns are orthonormal
    return Q[:, :rank]


def build_bank(cfg: dict):
    dim = int(cfg["dim"])
    n = int(cfg["bank_size"])
    true_r = int(cfg["true_manifold_rank"])
    noise = float(cfg["noise_off_manifold"])
    seed = int(cfg["seed"])

    # 1. The TRUE sheet: `true_r` orthonormal directions inside the dim-D room.
    B = random_orthonormal_basis(dim, true_r, seed=seed)      # [dim, true_r]

    # 2. Random coordinates ALONG the sheet -> points that sit ON the sheet.
    rng = np.random.default_rng(seed + 1)
    coords = rng.standard_normal((n, true_r))                 # [n, true_r]
    on_sheet = coords @ B.T                                   # [n, dim] ON the sheet

    # 3. A hair of off-sheet wobble in EVERY direction (no data is perfectly flat)
    wobble = noise * rng.standard_normal((n, dim))            # [n, dim]
    bank = on_sheet + wobble                                  # the "real" bank
    return bank.astype(np.float32), B.astype(np.float32)


def worked_2d_example() -> None:
    """Print the by-hand 2-D on/off-manifold example from the docstring."""
    banner("2-D WORKED EXAMPLE: on-manifold vs off-manifold (check by hand)")

    def project_to_xaxis(v):
        # The 1-D sheet is the x-axis; projecting keeps x, zeroes y.
        return np.array([v[0], 0.0])

    def residual(delta):
        d = np.asarray(delta, dtype=float)
        proj = project_to_xaxis(d)
        return float(np.linalg.norm(d - proj) / (np.linalg.norm(d) + 1e-12))

    P = np.array([2.0, 0.0])
    delta_on = np.array([1.0, 0.0])     # push ALONG the sheet
    delta_off = np.array([0.0, 1.0])    # push OFF the sheet
    print(f"  sheet = the x-axis line; point P = {tuple(P)} (on the sheet)")
    print(f"  edit ALONG sheet  Delta_on  = {tuple(delta_on)} -> "
          f"P+Delta = {tuple(P + delta_on)} (still on the line)")
    print(f"  edit OFF   sheet  Delta_off = {tuple(delta_off)} -> "
          f"P+Delta = {tuple(P + delta_off)} (floats off the line)")
    print(f"  off-manifold residual of Delta_on  = {residual(delta_on):.3f}  "
          f"(0 => fully on-manifold)")
    print(f"  off-manifold residual of Delta_off = {residual(delta_off):.3f}  "
          f"(1 => fully off-manifold)")
    print("  >>> on-manifold steering keeps only the (1,0)-style part of an edit;")
    print("      naive steering adds the whole edit, off-sheet part and all.")


def main() -> None:
    cfg = load_cfg()
    banner("STEP 1 — build the synthetic REAL-IMAGE activation bank")
    print(f"  room (activation space) dimension : dim            = {cfg['dim']}")
    print(f"  TRUE sheet dimension (planted)     : true_manifold  = {cfg['true_manifold_rank']}")
    print(f"  bank size (#real activations)      : bank_size      = {cfg['bank_size']}")
    print(f"  off-sheet wobble                   : noise          = {cfg['noise_off_manifold']}")

    bank, B = build_bank(cfg)
    print(f"\n  built bank shape = {bank.shape}  (each row is one 64-number activation)")
    # Quick sanity: how much of each point's energy is ON vs OFF the true sheet?
    on_energy = (bank @ B)                       # coordinates along the sheet
    on_norm = np.linalg.norm(on_energy, axis=1)
    full_norm = np.linalg.norm(bank, axis=1)
    frac_on = float(np.mean(on_norm / (full_norm + 1e-8)))
    print(f"  mean fraction of each point ON the true sheet = {frac_on:.3f} "
          f"(~1.0 => points really do hug the sheet)")

    np.save(outpath("real_bank.npy"), bank)
    np.save(outpath("sheet_basis.npy"), B)
    print(f"\n  saved -> {outpath('real_bank.npy')}")
    print(f"  saved -> {outpath('sheet_basis.npy')}  (the TRUE sheet, for step2 to grade itself)")

    worked_2d_example()
    print("\nSTEP 1 done. Next: step2 estimates the sheet with PCA (U_r).")


# REAL RUN (M4): replace build_bank() with a loader that streams a LARGE bank of
# REAL CLIP ViT-B/16 patch activations over ImageNet-val (e.g. via src.data.
# image_stream), cache it once to outputs/real_bank.npy, and DELETE the planted
# sheet_basis.npy step (in real life the sheet is unknown — PCA is the estimate,
# there is nothing to grade it against). Everything downstream is identical:
# step2 PCAs whatever real_bank.npy contains.
if __name__ == "__main__":
    main()
