"""step1_build_bank.py — build the multi-concept activation bank + train the SAE.

==============================================================================
WHAT THIS STEP DOES (in one sentence)
==============================================================================
It manufactures a pile of pretend "real-image activations" in which SEVERAL
named concepts are planted (one TARGET we will steer, plus OFF-TARGET concepts
that should stay put), labels each activation with which concepts are present,
trains a tiny TopK Sparse Autoencoder on them, and saves everything for the
later steps to score.

This is the foundation for the whole CFS evaluation: you cannot measure whether
"only the target concept moved" until you have OTHER concepts to watch.

==============================================================================
TEACH-FROM-ZERO: every term used below, defined before it is used
==============================================================================

ACTIVATION
  A vector of numbers a neural network produces inside itself while looking at
  an input — the model's private notes about one image-patch. Real CLIP ViT-B/16
  notes are 768 numbers long; here we use dim=64 so a laptop CPU runs instantly.
  One activation = one point in a 64-dimensional space (a list of 64 numbers).

A CONCEPT, and its DIRECTION
  A "concept" is a human-meaningful property an image can have ("has stripes",
  "is a dog", "is outdoors"). Inside the activation space, a concept shows up as
  a fixed DIRECTION d (a unit-length 64-number arrow). An image that HAS the
  concept has its activation pushed a little ALONG d; an image without it does
  not. So "concept present" == "activation has a positive component along d".
  Tiny number: if d = (1, 0) in a 2-D space, an image with the concept might land
  at (3.0, 0.2) (big along d) and one without it at (0.1, -0.4) (small along d).

A READOUT / MEASUREMENT OF A CONCEPT
  A readout is a single number that says "how much of this concept is in this
  activation". The simplest readout is the DOT PRODUCT of the activation with the
  concept direction: readout(a) = <a, d> = a[0]*d[0] + a[1]*d[1] + ...  A big
  readout = lots of the concept; a small/negative readout = little of it.
  Tiny number: a = (3.0, 0.2), d = (1, 0)  ->  readout = 3.0*1 + 0.2*0 = 3.0.
  We will sweep the steering knob and watch this readout RISE — that is exactly
  what the monotonicity component (step3) scores. (A learned linear probe, taught
  in step2, is a smarter readout; the dot product is the bare-bones version.)

A LABEL
  For each activation we also store a 0/1 LABEL per concept: 1 = "this concept is
  present in this item", 0 = "absent". Labels are what let step2 train a probe
  (a probe needs examples of present-vs-absent to learn the boundary) and what
  let step3 measure an effect size (present group vs absent group).

THE DATA MANIFOLD (the "sheet") — why it matters here
  Real activations do NOT fill the whole 64-dim space. A frozen model, fed real
  images, only ever lands on a thin SHEET inside it (milestone 4 teaches this in
  full). We plant the bank to live on a `true_manifold_rank`-dimensional sheet
  (e.g. 8-D inside 64-D), plus a hair of off-sheet wobble. This is what makes
  on-manifold steering MEANINGFUL: there is a real sheet to project onto, and a
  real OFF-sheet region where a naive edit can go wrong.

THE PLANTED BANK (how we build one activation)
  1. Pick a random `true_manifold_rank`-D SHEET (an orthonormal basis B_sheet),
     and place all `n_concepts` concept directions ON that sheet (so the model
     genuinely represents each concept — a concept lives where real data lives).
  2. For each item, flip a fair coin per concept -> the 0/1 LABELS.
  3. Start from background ON-sheet Gaussian variation, add `concept_strength` *
     direction for each present concept, then add a small OFF-sheet wobble in the
     remaining 56 directions (no real data is perfectly flat).
  Why this geometry (NOT mutually-orthogonal-in-full-space): the OFF-TARGET probes
  are trained on real ON-sheet data, so they only "trust" readings on the sheet.
  A naive edit adds the WHOLE raw SAE direction, whose off-sheet part shoves the
  activation into the mid-air region the off-target probes were never fit on —
  and there their readings DRIFT (specificity leakage, the brief's error #2). The
  on-manifold projection deletes exactly that off-sheet part, so off-target probes
  stay flat. The leakage we measure is therefore a REAL property of the steerer
  meeting a REAL manifold, not a rigged direction overlap.
  Result: a labelled bank where concept 0 is the steering TARGET and concepts
  1..n-1 are OFF-TARGET watchers, all living on a known low-rank sheet.

THE TopK SPARSE AUTOENCODER (SAE), in one breath
  An autoencoder squeezes each activation through a narrow code and rebuilds it;
  a SPARSE one forces most of the code entries to be zero. TopK keeps only the k
  biggest code entries on (Gao et al. 2024). Each decoder column is a "concept
  direction the SAE discovered". We train the project's real TopK SAE here so the
  later steerers (clamp_steer, onmanifold_steer) have genuine SAE features to act
  on. (We REUSE src/ — we do not re-implement the SAE.)

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step1_build_bank.py
Writes:
  outputs/concept_dirs.npy   [n_concepts, dim]  the planted concept directions
  outputs/probe_acts.npy     [bank_size, dim]   labelled activations (for probes)
  outputs/probe_labels.npy   [bank_size, n_concepts]  the 0/1 labels
  outputs/sae_decoder.npy    [dim, sae_dim]     trained SAE decoder columns
"""
from __future__ import annotations

import numpy as np

from _common import banner, load_cfg, outpath


def random_orthonormal(dim: int, rank: int, seed: int) -> np.ndarray:
    """Return [dim, rank] with orthonormal columns (a basis for a `rank`-D sheet).

    We draw a random matrix and orthonormalize it with QR (the standard "make
    these arrows perpendicular and length-1" tool).
    """
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((dim, rank))
    Q, _ = np.linalg.qr(M)
    return Q[:, :rank].astype(np.float32)


def on_sheet_concept_dirs(B_sheet: np.ndarray, n_concepts: int,
                          seed: int) -> np.ndarray:
    """Return [n_concepts, dim]: unit concept directions that lie ON the sheet.

    We place each concept inside the column span of B_sheet (so the concept lives
    where real data lives) and orthonormalize the chosen sheet-coordinates so the
    concepts do not mechanically overlap WITHIN the sheet — any leakage we measure
    later is then the steerer pushing OFF the sheet, not a rigged in-sheet overlap.
    """
    dim, true_r = B_sheet.shape
    rng = np.random.default_rng(seed + 5)
    C = rng.standard_normal((true_r, n_concepts))     # coordinates within the sheet
    Qc, _ = np.linalg.qr(C)                            # perpendicular within-sheet
    dirs = (B_sheet @ Qc[:, :n_concepts]).T           # [n_concepts, dim], on-sheet
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
    return dirs.astype(np.float32)


def build_labelled_bank(cfg: dict):
    """Build (acts [N, dim], labels [N, n_concepts], dirs [n_concepts, dim]).

    The bank lives on a known low-rank SHEET. Concepts and the bulk of the
    variation sit ON the sheet, so the linear probes (step2) learn the genuine
    on-sheet concept directions. We ALSO plant a realistic POLYSEMANTIC twist:
    whenever the TARGET concept is present we add a fixed OFF-SHEET nuisance axis
    `g`. A Sparse Autoencoder trained on this bank then discovers a target feature
    that is partly off-sheet (entangled) — exactly the polysemantic feature real
    SAEs find (brief error #3). That off-sheet part is what a naive edit drags
    into the void (off-target probes misread -> specificity leakage, error #2),
    and exactly what the on-manifold projection deletes.
    """
    dim = int(cfg["dim"])
    n_c = int(cfg["n_concepts"])
    N = int(cfg["bank_size"])
    true_r = int(cfg["true_manifold_rank"])
    strength = float(cfg["concept_strength"])
    noise = float(cfg["noise_off_manifold"])
    ent = float(cfg.get("entanglement", 3.0))
    seed = int(cfg["seed"])

    # 1. The TRUE sheet: a random true_r-D subspace inside the dim-D space.
    B_sheet = random_orthonormal(dim, true_r, seed=seed)        # [dim, true_r]
    # 2. Concept directions placed ON that sheet (perpendicular within the sheet).
    dirs = on_sheet_concept_dirs(B_sheet, n_c, seed)            # [n_concepts, dim]
    # 2b. A fixed OFF-SHEET nuisance axis g (in the sheet's orthogonal complement).
    rng_g = np.random.default_rng(seed + 23)
    g = rng_g.standard_normal(dim).astype(np.float32)
    g = g - B_sheet @ (B_sheet.T @ g)                           # remove on-sheet part
    g = g / (np.linalg.norm(g) + 1e-8)                          # unit, fully off-sheet

    rng = np.random.default_rng(seed + 11)
    # 0/1 labels: fair coin per concept per item -> a balanced present/absent set.
    labels = (rng.random((N, n_c)) < 0.5).astype(np.float32)    # [N, n_concepts]

    # 3a. Background ON-SHEET variation (the "everything else" that real data has).
    on_coords = noise * rng.standard_normal((N, true_r)).astype(np.float32)
    acts = on_coords @ B_sheet.T                               # [N, dim], on-sheet
    # 3b. Inject present concepts (also on the sheet).
    acts += (labels @ dirs) * strength                          # [N, dim]
    # 3c. POLYSEMANTIC entanglement: target presence also lights up the off-sheet
    #     axis g (the target feature is NOT a clean on-sheet concept).
    tgt = int(cfg["target_concept"])
    acts += (labels[:, tgt:tgt + 1] * ent) * g[None, :]         # [N, dim]
    # 3d. A hair of OFF-SHEET wobble in EVERY direction (no data is perfectly flat).
    acts += 0.15 * noise * rng.standard_normal((N, dim)).astype(np.float32)
    # Save g + the sheet basis as diagnostics (the steering still uses the SAE dir).
    np.save(outpath("offsheet_axis.npy"), g)
    np.save(outpath("sheet_basis.npy"), B_sheet.astype(np.float32))
    return acts.astype(np.float32), labels, dirs


def train_sae(cfg: dict, acts: np.ndarray) -> np.ndarray:
    """Train the project's real TopK SAE on the bank; return decoder [dim, sae_dim].

    We reuse src.model.make_model (the FaithSAE harness). Only the SAE trains; the
    backbone is frozen. The decoder columns become the SAE's discovered concept
    directions that clamp_steer / onmanifold_steer steer.
    """
    import torch

    from src.model import make_model
    from src.utils import set_seed

    set_seed(int(cfg["seed"]))
    model = make_model(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-3)

    # The SAE trains directly on the activation bank (a stream of patch
    # activations). We do NOT need the frozen backbone here — the SAE just learns
    # to reconstruct these activations sparsely.
    A = torch.from_numpy(acts).float()
    bs = 256
    steps = int(cfg["steps"])
    for step in range(steps):
        idx = torch.randint(0, A.shape[0], (bs,))
        a = A[idx]
        _, _, loss = model.sae(a)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        dec = model.sae.dec.weight.detach().cpu().numpy()      # [dim, sae_dim]
    return dec.astype(np.float32)


def worked_readout_example() -> None:
    """Print a tiny by-hand 'readout = dot product' example."""
    banner("TINY WORKED EXAMPLE: a concept readout is just a dot product")
    d = np.array([1.0, 0.0])                # concept direction (2-D for clarity)
    a_has = np.array([3.0, 0.2])            # an activation that HAS the concept
    a_not = np.array([0.1, -0.4])           # one that does NOT
    print(f"  concept direction d        = {tuple(d)}")
    print(f"  activation WITH concept  a = {tuple(a_has)} "
          f"-> readout <a,d> = {float(a_has @ d):.2f}  (big -> present)")
    print(f"  activation WITHOUT it    a = {tuple(a_not)} "
          f"-> readout <a,d> = {float(a_not @ d):.2f}  (small -> absent)")
    print("  >>> 'steer the concept up' should make this readout RISE smoothly.")
    print("      That rise, scored by Spearman, is the MONOTONICITY component.")


def main() -> None:
    cfg = load_cfg()
    banner("STEP 1 — build the multi-concept activation bank + train the TopK SAE")
    print(f"  activation width            : dim            = {cfg['dim']}")
    print(f"  number of planted concepts  : n_concepts     = {cfg['n_concepts']}")
    print(f"  steering TARGET concept     : target_concept = {cfg['target_concept']} "
          f"(the rest are OFF-TARGET watchers)")
    print(f"  labelled bank size          : bank_size      = {cfg['bank_size']}")

    acts, labels, dirs = build_labelled_bank(cfg)
    print(f"\n  built bank: acts {acts.shape}, labels {labels.shape}, "
          f"concept_dirs {dirs.shape}")
    # Sanity: items WITH concept 0 should have a higher mean readout than items
    # without it (the planted signal is really there).
    tgt = int(cfg["target_concept"])
    read = acts @ dirs[tgt]                  # readout of every item on the target
    has = read[labels[:, tgt] == 1].mean()
    no = read[labels[:, tgt] == 0].mean()
    print(f"  sanity: target readout  present-mean = {has:.2f}  vs  "
          f"absent-mean = {no:.2f}  (present should be clearly higher)")

    print("\n  training the project's TopK SAE on the bank ...")
    dec = train_sae(cfg, acts)
    print(f"  trained SAE decoder shape = {dec.shape}  "
          f"(each column is a discovered concept direction)")

    np.save(outpath("concept_dirs.npy"), dirs)
    np.save(outpath("probe_acts.npy"), acts)
    np.save(outpath("probe_labels.npy"), labels)
    np.save(outpath("sae_decoder.npy"), dec)
    print(f"\n  saved -> {outpath('concept_dirs.npy')}  (planted concept directions)")
    print(f"  saved -> {outpath('probe_acts.npy')}     (labelled activations)")
    print(f"  saved -> {outpath('probe_labels.npy')}   (0/1 labels per concept)")
    print(f"  saved -> {outpath('sae_decoder.npy')}    (trained SAE decoder)")

    worked_readout_example()
    print("\nSTEP 1 done. Next: step2 trains the LINEAR PROBES (one ruler per concept).")


# REAL RUN (M5): replace build_labelled_bank() with REAL CLIP ViT-B/16 patch
# activations over ImageNet-val, and replace the 0/1 labels with real concept
# annotations (e.g. attribute labels, or CLIP-text-matched concept sets). Train
# the SAE on the real bank (or load the M4-trained SAE). Everything downstream is
# identical: probes, knob sweep, and CFS all read these saved arrays.
if __name__ == "__main__":
    main()
