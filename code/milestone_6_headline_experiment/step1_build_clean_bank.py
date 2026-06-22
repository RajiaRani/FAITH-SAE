"""step1_build_clean_bank.py — build the CLEAN, in-distribution activation bank.

==============================================================================
WHAT THIS STEP DOES (in one sentence)
==============================================================================
It manufactures a pile of pretend CLEAN "real-image activations" that live on a
thin low-dimensional SHEET inside a big 64-D space, with SEVERAL named concepts
planted in (one TARGET we will steer + several OFF-TARGET concepts that should
stay put). This is the in-distribution reference; every later rung of the OOD
ladder is a CORRUPTED version of this same bank.

==============================================================================
TEACH-FROM-ZERO: every term used below, defined before it is used
==============================================================================

ACTIVATION
  A vector of numbers a neural network produces inside itself while looking at an
  input -- the model's private notes about one image-patch. Real CLIP ViT-B/16
  notes are 768 numbers long; we use dim=64 so a laptop CPU runs it instantly.
  One activation = one point in a 64-dimensional space (a list of 64 numbers).

THE DATA MANIFOLD (the "sheet")
  Real activations do NOT fill the whole 64-D space. A frozen model, fed real
  photos, only ever lands in a thin curved SHEET inside it -- like a sheet of
  paper floating in a gym. The paper is (say) 8-D even though the gym is 64-D.
  The model was effectively trained ON the paper, so it only behaves sensibly for
  points ON the paper. Points OFF the paper (mid-air) are activations the model
  has never really seen -- it does not handle them reliably.
  Tiny number: if the sheet is 8-D inside a 64-D gym, then 56 of the 64 directions
  are "off the sheet" -- clean activations have ~zero spread along those 56.

IN-DISTRIBUTION vs OUT-OF-DISTRIBUTION (the WHOLE point of milestone 6)
  * IN-DISTRIBUTION (clean): the photos look like the photos the model was built
    on; their activations sit ON the clean sheet. This is the bank we build here.
  * OUT-OF-DISTRIBUTION (OOD): the photos are DIFFERENT in some way the model
    never trained on (a pencil SKETCH, an art RENDITION, a blurry/noisy CORRUPTED
    photo, an odd real-world POSE). Their activations drift OFF the clean sheet.
  Milestone 6 builds the clean bank now and CORRUPTS it (step3) to simulate that
  drift, rung by rung.

A CONCEPT, A CONCEPT DIRECTION, AND A LABEL
  A "concept" is a human-meaningful thing an image can have or not (e.g. "stripes",
  "wheel", "fur"). Inside the model it shows up as a fixed DIRECTION d: items that
  HAVE the concept have activations pushed a bit along d. We plant `n_concepts`
  such directions. For each item we flip a coin per concept (LABEL 1 = present,
  0 = absent) and, if present, add `concept_strength * d`. Because we know the
  labels, step2 can train a linear PROBE (a ruler) that reads each concept back.

TARGET vs OFF-TARGET CONCEPTS (this is what SPECIFICITY will test)
  ONE concept is the TARGET (`target_concept`) -- the one we will steer up. The
  rest are OFF-TARGET -- they should NOT move when we steer the target. A faithful
  steer moves ONLY the target; an unfaithful one smears into the off-target ones.

WHY THE CONCEPT DIRECTIONS ARE (MOSTLY) ON THE CLEAN SHEET, WITH A HAIR OFF
  Real concepts the model uses live on its sheet. We build each concept direction
  as MOSTLY on the clean sheet plus a tiny off-sheet sliver (`offsheet_frac`).
  That sliver is exactly what on-manifold steering's projection will trim away and
  naive steering will keep -- the seed of the on-manifold-vs-naive difference.

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step1_build_clean_bank.py
Writes outputs/clean_acts.npy   ([bank_size, dim] CLEAN activations),
       outputs/labels.npy        ([bank_size, n_concepts] 0/1 concept labels),
       outputs/read_dirs.npy     ([n_concepts, dim] GENUINE on-sheet concept dirs
                                  -- what the probe certifies),
       outputs/concept_dirs.npy  ([n_concepts, dim] SAE STEERING dirs d = read_dir
                                  + an off-sheet sliver -- what we steer),
       outputs/sheet_basis.npy   ([dim, true_manifold_rank] TRUE clean sheet B),
       outputs/style_basis.npy   ([dim, n_style_dirs] shared OFF-sheet subspace S
                                  the OOD shift floods in step3).
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


def style_subspace(B: np.ndarray, dim: int, n_style: int, seed: int) -> np.ndarray:
    """A small fixed OFF-sheet subspace S (the "style/rendition" directions).

    These are directions PERPENDICULAR to the clean sheet B. Two things will live
    in this same subspace -- (a) every concept's off-sheet SLIVER, and (b) the OOD
    corruption's style push (step3). That overlap is the honest mechanism: the
    SAME off-sheet directions a naive edit injects energy into are the ones the
    shift floods with junk, so naive's edit and the shift collide there. Returns a
    [dim, n_style] matrix with orthonormal columns, all off the clean sheet.
    """
    rng = np.random.default_rng(seed)
    P = B @ B.T                                   # projector onto the clean sheet
    G = rng.standard_normal((dim, n_style))
    G = G - P @ G                                 # strip the on-sheet part -> OFF
    Q, _ = np.linalg.qr(G)                        # orthonormalize the off-sheet dirs
    return Q[:, :n_style].astype(np.float32)


def make_concept_dirs(B: np.ndarray, S: np.ndarray, dim: int, n_concepts: int,
                      offsheet_frac: float, seed: int):
    """Build TWO aligned direction sets per concept (this split is the whole point).

    read_dir (ON the sheet ONLY) -- the GENUINE concept signal. This is what we
      plant into clean activations and what the probe (the ruler) certifies. A
      real concept the model truly represents lives on its manifold, so moving the
      genuine concept means moving ALONG read_dir, on the sheet.
    sae_dir  (= read_dir + an off-sheet SLIVER in S) -- the SAE-DISCOVERED steering
      direction d. Real SAE feature directions are not perfectly on-manifold; they
      carry an off-sheet sliver (here of size `offsheet_frac`, living in the shared
      style subspace S). This sliver does NOT correspond to genuine concept change
      -- it is the seed of an OFF-MANIFOLD MIRAGE.

    The contrast that drives the whole experiment:
      * on-manifold steering projects sae_dir onto the clean sheet -> recovers
        (essentially) read_dir -> moves the GENUINE concept, no mirage.
      * naive steering adds the WHOLE sae_dir -> moves the genuine concept AND
        injects the off-sheet sliver into S. On clean data that sliver is just
        wasted; under SHIFT (which floods S with junk) it corrupts the readout and
        leaks into off-target probes -> faithfulness collapses faster.
    Returns (read_dirs [n,dim] on-sheet, sae_dirs [n,dim] on-sheet+sliver).
    """
    rng = np.random.default_rng(seed)
    true_r = B.shape[1]
    n_style = S.shape[1]
    read_dirs, sae_dirs = [], []
    for _ in range(n_concepts):
        on = B @ rng.standard_normal(true_r)      # a random ON-sheet direction
        on = on / (np.linalg.norm(on) + 1e-8)     # read_dir: genuine concept dir
        off = S @ rng.standard_normal(n_style)    # a random OFF-sheet style dir
        off = off / (np.linalg.norm(off) + 1e-8)
        sae = (1.0 - offsheet_frac) * on + offsheet_frac * off  # SAE dir = on + sliver
        sae = sae / (np.linalg.norm(sae) + 1e-8)
        read_dirs.append(on)
        sae_dirs.append(sae)
    return (np.stack(read_dirs).astype(np.float32),
            np.stack(sae_dirs).astype(np.float32))


def build_clean_bank(cfg: dict):
    dim = int(cfg["dim"])
    n = int(cfg["bank_size"])
    n_c = int(cfg["n_concepts"])
    true_r = int(cfg["true_manifold_rank"])
    seed = int(cfg["seed"])
    cstr = float(cfg["concept_strength"])
    on_noise = float(cfg["noise_on_manifold"])
    off_noise = float(cfg["noise_off_manifold"])

    # 1. The TRUE CLEAN sheet: `true_r` orthonormal directions inside the 64-D room.
    B = random_orthonormal_basis(dim, true_r, seed=seed)         # [dim, true_r]

    # 2. The shared OFF-sheet STYLE subspace S (where the concept slivers AND the
    #    OOD shift both live -- the honest collision point under shift).
    S = style_subspace(B, dim, n_style=int(cfg["n_style_dirs"]), seed=seed + 3)

    # 3. The GENUINE on-sheet read directions + the SAE steering dirs (read+sliver).
    read_dirs, sae_dirs = make_concept_dirs(
        B, S, dim, n_c, offsheet_frac=float(cfg["concept_offsheet_frac"]), seed=seed + 5)

    # 4. Background ON-sheet activation cloud (the "everything else" the model
    #    represents on clean photos) + a tiny off-sheet wobble.
    rng = np.random.default_rng(seed + 1)
    coords = on_noise * rng.standard_normal((n, true_r))         # ON-sheet coords
    base = coords @ B.T                                          # [n, dim] ON sheet
    wobble = off_noise * rng.standard_normal((n, dim))          # tiny OFF-sheet
    acts = base + wobble

    # 5. Plant the GENUINE concept (the on-sheet read_dir) into clean activations:
    #    flip a coin per (item, concept); if present add read_dir at concept_strength.
    #    The probe trained on these certifies ON-sheet (genuine) concept movement.
    labels = (rng.random((n, n_c)) < 0.5).astype(np.float32)    # [n, n_concepts]
    acts = acts + (labels * cstr) @ read_dirs                   # [n, dim]

    return (acts.astype(np.float32), labels, read_dirs, sae_dirs,
            B.astype(np.float32), S.astype(np.float32))


def main() -> None:
    cfg = load_cfg()
    banner("STEP 1 — build the CLEAN (in-distribution) activation bank")
    print(f"  room (activation space) dimension : dim             = {cfg['dim']}")
    print(f"  TRUE clean sheet dimension         : true_manifold   = {cfg['true_manifold_rank']}")
    print(f"  bank size (#clean activations)     : bank_size       = {cfg['bank_size']}")
    print(f"  #concepts planted (1 target + off) : n_concepts      = {cfg['n_concepts']}")
    print(f"  target concept (the one we steer)  : target_concept  = {cfg['target_concept']}")
    print(f"  concept signal size                : concept_strength= {cfg['concept_strength']}")

    acts, labels, read_dirs, sae_dirs, B, S = build_clean_bank(cfg)
    print(f"\n  built CLEAN acts shape   = {acts.shape}  (each row = one 64-number activation)")
    print(f"  built labels shape       = {labels.shape}  (0/1 per concept per item)")
    print(f"  built read_dirs shape    = {read_dirs.shape}  (GENUINE on-sheet concept dirs; what the probe certifies)")
    print(f"  built sae_dirs shape     = {sae_dirs.shape}  (SAE steering dirs = read_dir + off-sheet sliver)")
    print(f"  built style_subspace S   = {S.shape}  (shared OFF-sheet dirs: concept slivers + shift live here)")

    # Sanity: how much of each clean point's energy is ON vs OFF the true sheet?
    on_energy = acts @ B                          # coordinates along the clean sheet
    frac_on = float(np.mean(np.linalg.norm(on_energy, axis=1)
                            / (np.linalg.norm(acts, axis=1) + 1e-8)))
    print(f"\n  mean fraction of each CLEAN point ON the true sheet = {frac_on:.3f} "
          f"(~1.0 => clean points really do hug the sheet)")
    # The genuine read_dir is fully on-sheet; the SAE steering dir has the sliver.
    P = B @ B.T
    tgt = int(cfg["target_concept"])
    read_off = float(np.linalg.norm(read_dirs[tgt] - P @ read_dirs[tgt])
                     / (np.linalg.norm(read_dirs[tgt]) + 1e-8))
    sae_off = float(np.linalg.norm(sae_dirs[tgt] - P @ sae_dirs[tgt])
                    / (np.linalg.norm(sae_dirs[tgt]) + 1e-8))
    print(f"  off-sheet fraction of the TARGET read_dir (genuine) = {read_off:.3f} "
          f"(~0: the genuine concept lives ON the sheet)")
    print(f"  off-sheet fraction of the TARGET sae_dir  (steered) = {sae_off:.3f} "
          f"(the sliver on-manifold TRIMS and naive KEEPS)")

    np.save(outpath("clean_acts.npy"), acts)
    np.save(outpath("labels.npy"), labels)
    np.save(outpath("read_dirs.npy"), read_dirs)
    np.save(outpath("concept_dirs.npy"), sae_dirs)   # the SAE steering dirs d
    np.save(outpath("sheet_basis.npy"), B)
    np.save(outpath("style_basis.npy"), S)
    print(f"\n  saved -> {outpath('clean_acts.npy')}")
    print(f"  saved -> {outpath('labels.npy')}")
    print(f"  saved -> {outpath('read_dirs.npy')}     (genuine on-sheet concept dirs)")
    print(f"  saved -> {outpath('concept_dirs.npy')}  (SAE steering dirs d = read_dir + sliver)")
    print(f"  saved -> {outpath('sheet_basis.npy')}  (the TRUE clean sheet, for step2 to self-grade)")
    print(f"  saved -> {outpath('style_basis.npy')}  (the OFF-sheet style subspace the OOD shift floods in step3)")
    print("\nSTEP 1 done. Next: step2 estimates the FROZEN clean sheet U_r by PCA.")


# REAL RUN (M6): replace build_clean_bank() with a loader that streams a LARGE
# bank of REAL CLIP ViT-B/16 patch activations over CLEAN ImageNet-val (e.g. via
# src.data.image_stream), with real concept labels. The GENUINE concept "read"
# direction is just the trained linear probe (no need to plant it); the SAE
# steering direction d is the matching SAE decoder column (its small off-manifold
# component is real, not injected -- so read_dirs/concept_dirs come from the data,
# not from make_concept_dirs). Cache clean_acts.npy + labels.npy once; DELETE
# sheet_basis.npy and style_basis.npy (in real life the clean sheet and the shift
# subspace are unknown -- PCA in step2 is the estimate, nothing to grade against,
# and the shift comes from REAL datasets in step3, not a planted subspace).
# Everything downstream is identical: step2 PCAs whatever clean_acts.npy contains.
if __name__ == "__main__":
    main()
