#!/usr/bin/env /usr/bin/python3
# =============================================================================
# step1_build_synthetic_bank.py
# Milestone 2 (FAITH-SAE) -- STEP 1 of 2: build the activation bank.
# Author: Rajia Rani  ()
#
# WHAT THIS FILE DOES (in one sentence):
#   It MANUFACTURES a pile of numbers shaped EXACTLY like the activations a real
#   CLIP ViT-B/16 vision model would produce for a batch of images, and saves
#   them to outputs/activations.npz -- so the rest of the project (the Sparse
#   Autoencoder, the steering, the faithfulness metric) has data to train on
#   TODAY, with no GPU, no model download, and no internet.
#
# WHY FAKE DATA?
#   The real pipeline (run a CLIP model over ImageNet images, grab the inner
#   activations) needs a GPU, multi-GB model weights, and ~150 GB of image
#   datasets. None of that is available on this laptop. But the SHAPE and the
#   STATISTICAL FEEL of the data are what every downstream step actually cares
#   about. So we build a synthetic bank that has the same shape and the same
#   key properties (a low-dimensional "manifold", planted concept directions,
#   and an optional out-of-distribution shift knob). Code written against this
#   bank runs UNCHANGED on real activations once they exist.
#
# READ THE README FIRST. Every term below (activation, token, CLS, patch,
# manifold, concept direction, OOD, variance) is defined from absolute zero
# there. This file is the runnable companion to that explanation.
# =============================================================================

from __future__ import annotations

import os
import sys

import numpy as np

# ---- Make the project root importable so we can reuse src/utils.py ----------
# This milestone folder lives at <project>/code/milestone_2_data/. The shared
# helpers (config loader, seeding) live at <project>/src/. We add <project> to
# Python's import path so `from src.utils import ...` works no matter where the
# script is launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))           # .../code/milestone_2_data
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))  # .../25_..._FAITH_SAE
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import load_config, set_seed   # noqa: E402  (after sys.path tweak)


# =============================================================================
# The synthetic-activation generator.
# =============================================================================
def build_activation_bank(
    n_images: int,
    n_tokens: int,
    dim: int,
    manifold_rank: int,
    manifold_scale: float,
    noise_scale: float,
    n_concepts: int,
    concept_strength: float,
    concept_prevalence: float,
    ood_shift: float,
    seed: int,
):
    """Manufacture an activation bank shaped like CLIP ViT-B/16 patch tokens.

    Returns a dict of numpy arrays:
      acts            [n_images, n_tokens, dim]  the activations themselves.
      concept_dirs    [n_concepts, dim]          the planted concept directions
                                                 (unit vectors). Ground truth.
      concept_labels  [n_images, n_concepts]     1.0 if that concept is "present"
                                                 in that image, else 0.0.
      basis           [dim, manifold_rank]       orthonormal basis of the "real-
                                                 image manifold" subspace (used
                                                 later for on-manifold projection).

    The construction, piece by piece (all explained in the README):

      1. MANIFOLD CORE. Real activations do not scatter randomly through all
         `dim` directions; they live near a thin lower-dimensional sheet (the
         "manifold"). We imitate that: draw a random orthonormal `basis` of
         `manifold_rank` directions, then build each token as a random combo of
         ONLY those directions. Analogy: a sheet of paper (2-D) floating inside a
         room (3-D) -- points live on the paper, not everywhere in the room.

      2. PLANTED CONCEPTS. A "concept" is one meaningful direction. For each of
         `n_concepts` we pick a fixed unit vector INSIDE the manifold, then for a
         random `concept_prevalence` fraction of images we add that direction at
         strength `concept_strength`. That is the ground-truth "this image has
         stripes" signal we will later try to recover and steer.

      3. NOISE. A faint `noise_scale` of fully-random (off-manifold) wobble, so
         no dimension is ever perfectly silent -- real data is never that clean.

      4. OOD SHIFT (optional). If `ood_shift > 0` we deform the whole bank to
         imitate a distribution shift: rotate the manifold a little, rescale it,
         and inject extra off-manifold energy. That is what "the images got
         harder / came from a different distribution" does to activations.
    """
    rng = np.random.default_rng(seed)

    # --- 1. The manifold basis: `manifold_rank` orthonormal directions in dim-D.
    # A random Gaussian matrix's QR decomposition gives orthonormal columns.
    raw = rng.standard_normal((dim, manifold_rank))
    basis, _ = np.linalg.qr(raw)                 # basis: [dim, manifold_rank], orthonormal
    basis = basis[:, :manifold_rank]

    # Per-direction "importance": real manifolds have a few strong directions and
    # many weak ones (a decaying spectrum). We give the k-th direction weight
    # ~ 1/sqrt(k+1) so the first few PCA components dominate -- exactly what we
    # want the EDA's PCA scatter to reveal.
    spectrum = manifold_scale / np.sqrt(np.arange(1, manifold_rank + 1))   # [manifold_rank]

    # --- Random coordinates ON the manifold for every token of every image.
    # Shape [n_images, n_tokens, manifold_rank]: each token gets its own point.
    coords = rng.standard_normal((n_images, n_tokens, manifold_rank)) * spectrum

    # Lift the manifold coordinates back into the full dim-D space:
    #   acts = coords @ basis.T   -> [n_images, n_tokens, dim]
    acts = coords @ basis.T

    # --- 2. Planted concept directions (unit vectors living inside the manifold).
    concept_dirs = np.zeros((n_concepts, dim), dtype=np.float64)
    concept_labels = np.zeros((n_images, n_concepts), dtype=np.float64)
    for c in range(n_concepts):
        # A random combo of basis directions => guaranteed on-manifold concept.
        w = rng.standard_normal(manifold_rank)
        d = basis @ w
        d = d / (np.linalg.norm(d) + 1e-8)       # make it a unit vector
        concept_dirs[c] = d

        # Decide which images "have" this concept (a Bernoulli coin per image).
        present = rng.random(n_images) < concept_prevalence       # [n_images] bool
        concept_labels[:, c] = present.astype(np.float64)

        # Where present, add the concept direction to EVERY token of that image
        # at strength `concept_strength`. (Real concepts are diffuse across tokens.)
        add = present[:, None, None] * (concept_strength * d[None, None, :])
        acts = acts + add

    # --- 3. Faint off-manifold noise so nothing is perfectly degenerate.
    acts = acts + noise_scale * rng.standard_normal(acts.shape)

    # --- 4. Optional OOD shift: deform the bank to imitate a distribution shift.
    if ood_shift > 0.0:
        acts = _apply_ood_shift(acts, basis, ood_shift, rng)

    return {
        "acts": acts.astype(np.float32),
        "concept_dirs": concept_dirs.astype(np.float32),
        "concept_labels": concept_labels.astype(np.float32),
        "basis": basis.astype(np.float32),
    }


def _apply_ood_shift(acts: np.ndarray, basis: np.ndarray, shift: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Imitate a distribution shift on an activation bank.

    A distribution shift (OOD) means the new images come from a different source
    than training (renditions, sketches, corruptions, new viewpoints). In
    activation space this shows up three ways, which we reproduce:

      (a) ROTATION: the manifold tilts -- the same concept now points slightly
          elsewhere, so directions learned on clean data are a bit "off".
      (b) RESCALE: activation magnitudes drift (corruptions dim or inflate them).
      (c) OFF-MANIFOLD ENERGY: new content the clean manifold never modelled --
          the activation pops OFF the thin sheet it used to live on. This is the
          single most important fingerprint of OOD and the thing that breaks
          on-manifold methods.

    `shift` in [0, ~1] dials all three up together.
    """
    n_images, n_tokens, dim = acts.shape
    manifold_rank = basis.shape[1]

    # (a) Rotation: a small random rotation applied within the manifold subspace.
    #     Build a skew-symmetric matrix and exponentiate-ish via (I + shift*A),
    #     then re-orthonormalize. Keeps it a gentle, controllable tilt.
    A = rng.standard_normal((manifold_rank, manifold_rank))
    A = A - A.T                                    # skew-symmetric => pure rotation generator
    R = np.eye(manifold_rank) + shift * 0.5 * A
    R, _ = np.linalg.qr(R)                          # nearest orthonormal => a rotation
    rotated_basis = basis @ R                       # [dim, manifold_rank]
    # Re-express acts' on-manifold part through the rotated basis.
    coords = acts @ basis                           # project onto old basis: [.,.,rank]
    on_manifold = coords @ rotated_basis.T          # lift through rotated basis
    off_manifold = acts - (coords @ basis.T)        # whatever was already off-manifold

    # (b) Rescale: magnitudes drift by up to +/- shift.
    scale = 1.0 + shift * (rng.standard_normal((n_images, 1, 1)) * 0.5)

    # (c) Extra off-manifold energy: brand-new content the clean manifold lacks.
    off_extra = shift * 0.7 * rng.standard_normal(acts.shape)

    return (scale * on_manifold + off_manifold + off_extra).astype(acts.dtype)


# =============================================================================
# REAL RUN (M2): the real CLIP ViT-B/16 + ImageNet path.
# -----------------------------------------------------------------------------
# This function is DOCUMENTATION + a ready-to-uncomment template. It is NOT
# called by the default offline run and intentionally raises if invoked, so you
# never accidentally trigger a multi-GB download. To use it for real:
#   1) pip install open_clip_torch datasets pillow       (see requirements.txt)
#   2) Download ImageNet-val (and the OOD ladder) -- see the README §4.
#   3) Delete the `raise` line, fill in `imagenet_val_dir`, and run.
# =============================================================================
def build_real_clip_bank(cfg: dict):  # pragma: no cover  (never run offline)
    raise NotImplementedError(
        "REAL RUN (M2): this downloads CLIP weights + ImageNet. It is disabled "
        "for the offline default. See the README §4 and uncomment the body below."
    )
    # ---------------------------------------------------------------------
    # REAL RUN (M2): real CLIP ViT-B/16 patch-token extraction.
    # ---------------------------------------------------------------------
    # import torch, open_clip
    # from PIL import Image
    #
    # rc = cfg["real_run"]
    # # 1) Load the FROZEN CLIP vision tower (no training -- eval mode, no grad).
    # model, _, preprocess = open_clip.create_model_and_transforms(
    #     rc["clip_model"], pretrained=rc["clip_pretrained"])
    # model.eval()                              # freeze: we only read activations.
    # visual = model.visual                     # the ViT-B/16 image encoder.
    #
    # # 2) Register a forward hook on the chosen transformer block to CAPTURE the
    # #    token activations (shape [batch, n_tokens=197, dim=768]) as they flow.
    # captured = {}
    # layer = visual.transformer.resblocks[rc["backbone_layer"]]
    # def hook(_module, _inp, out):
    #     captured["acts"] = out.detach()       # [n_tokens, batch, dim] (CLIP layout)
    # handle = layer.register_forward_hook(hook)
    #
    # # 3) Stream images, preprocess to 224x224 tensors, run the frozen model.
    # banks = []
    # for img_path in iter_imagenet_val(rc["imagenet_val_dir"]):
    #     img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
    #     with torch.no_grad():
    #         visual(img)                       # forward pass; hook fills captured.
    #     a = captured["acts"].permute(1, 0, 2) # -> [batch, 197, 768]
    #     if rc["token_type"] == "patch":
    #         a = a[:, 1:, :]                   # drop CLS -> 196 PATCH tokens (SAE data)
    #     else:
    #         a = a[:, :1, :]                  # keep only the CLS token.
    #     banks.append(a.cpu().numpy())
    # handle.remove()
    # acts = np.concatenate(banks, axis=0)
    # return {"acts": acts, ...}               # same dict shape as the synthetic path.


# =============================================================================
# Main entry point.
# =============================================================================
def main():
    cfg_path = os.path.join(_HERE, "config.yaml")
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])

    out_dir = os.path.join(_HERE, cfg["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(_HERE, cfg["activations_npz"])

    print("=" * 70)
    print("FAITH-SAE Milestone 2  --  STEP 1: build the activation bank (OFFLINE)")
    print("=" * 70)
    print(f"Target shape (matches CLIP ViT-B/16 patch tokens): "
          f"[{cfg['n_images']} images x {cfg['n_tokens']} tokens x {cfg['dim']} dims]")
    print(f"  (= 196 patch tokens + 1 CLS token; width 768 = ViT-B/16 hidden size)")
    print(f"Planted concepts: {cfg['n_concepts']}  |  manifold rank: {cfg['manifold_rank']}"
          f"  |  OOD shift of this bank: {cfg['ood_shift']}")
    print("-" * 70)

    bank = build_activation_bank(
        n_images=cfg["n_images"],
        n_tokens=cfg["n_tokens"],
        dim=cfg["dim"],
        manifold_rank=cfg["manifold_rank"],
        manifold_scale=cfg["manifold_scale"],
        noise_scale=cfg["noise_scale"],
        n_concepts=cfg["n_concepts"],
        concept_strength=cfg["concept_strength"],
        concept_prevalence=cfg["concept_prevalence"],
        ood_shift=cfg["ood_shift"],
        seed=cfg["seed"],
    )

    # Save everything into one compressed .npz (numpy's multi-array container).
    np.savez_compressed(
        npz_path,
        acts=bank["acts"],
        concept_dirs=bank["concept_dirs"],
        concept_labels=bank["concept_labels"],
        basis=bank["basis"],
        # Stash the shape metadata so STEP 2 can sanity-check what it loaded.
        meta=np.array([cfg["n_images"], cfg["n_tokens"], cfg["dim"],
                       cfg["n_concepts"], cfg["manifold_rank"]], dtype=np.int64),
    )

    acts = bank["acts"]
    size_mb = os.path.getsize(npz_path) / 1e6
    print(f"acts array : shape {acts.shape}, dtype {acts.dtype}")
    print(f"value range: min {acts.min():+.3f}  max {acts.max():+.3f}  "
          f"mean {acts.mean():+.3f}  std {acts.std():.3f}")
    print(f"concept_labels prevalence per concept: "
          f"{bank['concept_labels'].mean(axis=0).round(3).tolist()}")
    print("-" * 70)
    print(f"SAVED -> {os.path.relpath(npz_path, _HERE)}  ({size_mb:.2f} MB)")
    print("STEP 1 complete. Next: run step2_eda.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# For research and educational purposes only.
# Author: Rajia Rani
# -----------------------------------------------------------------------------
