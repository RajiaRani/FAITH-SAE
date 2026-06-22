#!/usr/bin/env python3
# ===========================================================================
#  step1_train_sae.py  —  Milestone 3 (Baseline), Part A
#  Train a TopK Sparse Autoencoder (SAE) on a synthetic activation bank.
#  FAITH-SAE  ·  author: Rajia Rani  ·  educational use only
# ===========================================================================
#
#  ============ READ THIS FIRST: every term, from absolute zero ============
#
#  ACTIVATION
#    When a neural network looks at an input, every layer produces a list of
#    numbers — that list is an "activation vector". Think of it as the network's
#    private notes about the input. Example: a length-4 activation [0.2, -1.1,
#    0.0, 3.4]. Our bank is full of such vectors (length `dim` = 64 here).
#
#  AUTOENCODER
#    A network that learns to COPY its input to its output through a narrow (or,
#    here, a SPARSE) middle. Shape: input -> [encoder] -> middle code -> [decoder]
#    -> output. It is trained so output ~= input. Why copy? Because the middle
#    code is forced to be a compact/clean re-description of the input, and THAT
#    re-description is what we actually want.
#    Analogy: describe a photo to a friend in a few words ("beach, sunset, dog"),
#    and have them redraw it. If their redraw matches, your few words captured it.
#
#  ENCODER / DECODER
#    ENCODER = input vector -> feature code (a longer list of "concept" numbers,
#      mostly zero). DECODER = feature code -> reconstructed input vector.
#    Here encoder: 64 numbers -> 256 feature numbers; decoder: 256 -> 64.
#
#  FEATURE  /  CONCEPT DIRECTION  (the punchline of SAEs)
#    Each of the 256 features is "a concept". The DECODER is a matrix of shape
#    [64 x 256]; its column j is a length-64 vector — the direction in activation
#    space that feature j paints when it switches on. We call that column the
#    "concept direction" d_j. Steering a concept later = pushing activations along
#    its column. Tiny example: if column 5 is [0,1,0,...], turning feature 5 up
#    adds to the 2nd coordinate of the activation.
#
#  RECONSTRUCTION LOSS (MSE)
#    "How wrong is the copy?" We measure it with Mean Squared Error: take the
#    reconstructed vector minus the original, square each coordinate, average.
#    Example: original [1, 2], copy [1.5, 2]; errors [0.5, 0]; squares [0.25, 0];
#    MSE = 0.125. Training nudges weights to shrink this number. Lower = better
#    copy. (We watch it fall over training — that's the "loss curve".)
#
#  SPARSITY  &  WHY IT HELPS INTERPRETABILITY
#    "Sparse" = mostly zeros. We force only a few features to be nonzero per
#    activation. Why? If every input lit up all 256 features, each feature would
#    be a vague blur used for everything ("polysemantic"). Forcing few-active
#    pushes each feature to mean ONE clean thing ("monosemantic"), so a human can
#    name it. Analogy: a tidy toolbox with one labelled tool per slot beats a
#    drawer where every tool is half of three other tools.
#
#  TOP-K OPERATION
#    The exact way we enforce sparsity: of the 256 feature values, KEEP the k
#    largest (here k = 8) and SET THE REST TO ZERO. Example with k=2 on
#    [5, 1, 4, 0, 2]: the two largest are 5 and 4, so the kept code is
#    [5, 0, 4, 0, 0]. Simple, hard, no tuning knob to babysit (Gao et al. 2024).
#
#  DICTIONARY LEARNING (one line)
#    Learning a small set of reusable "atoms" (here: the decoder columns) so any
#    activation is a SHORT combination of a few atoms — that's all an SAE is.
#
#  LOSS CURVE
#    A plot of reconstruction loss (y) vs training step (x). A healthy curve
#    starts high and falls, fast at first then flattening as the SAE runs out of
#    easy wins. We save it as a PNG so you can SEE the SAE learned.
#
#  ------------------------------------------------------------------------
#  WHAT THIS SCRIPT DOES (Part A):
#    1. Regenerate a synthetic activation bank locally (no milestone_2 needed).
#    2. Build the project's real `TopKSAE` (from src/model.py).
#    3. Train it for a few hundred steps on CPU to minimise reconstruction MSE.
#    4. Save the checkpoint (outputs/sae_topk.pt) and the loss curve PNG.
#  ========================================================================

from __future__ import annotations

import argparse
import os
import sys

# --- Make the project's src/ importable, no matter where you run from. -----
# We add the repo ROOT (three folders up: milestone_3_baseline -> code -> repo)
# to Python's import path, so `import src...` finds the shared modules. This is
# the "Build on src/ via sys.path" rule from the contract.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Reuse the project's TopK SAE and seeding/config helpers (do NOT re-implement).
from src.model import _build  # noqa: E402  (_build returns the FaithSAE class)
from src.utils import load_config, set_seed  # noqa: E402


# ---------------------------------------------------------------------------
#  The synthetic activation bank — regenerated HERE so milestone_3 is fully
#  independent of milestone_2 (contract: do NOT depend on milestone_2 outputs).
#  Same SPIRIT as milestone 2: Gaussian "patch activations" with a handful of
#  planted concept directions injected at random strengths. The SAE's job is to
#  rediscover those planted directions as clean features.
# ---------------------------------------------------------------------------
def make_activation_bank(cfg: dict):
    """Return a tensor of shape [n_images, n_patches, dim] of synthetic
    activations, plus the planted ground-truth concept directions [n_concepts, dim].

    A "planted concept" is a fixed unit vector we deliberately add into some
    images, so we KNOW the answer the SAE should find. Real CLIP activations have
    such structure naturally; we inject it by hand here for an offline test."""
    import torch

    g = torch.Generator().manual_seed(cfg["seed"])
    n_img, n_pat, dim = cfg["n_images"], cfg["n_patches"], cfg["dim"]

    # --- The "manifold": real activations do NOT fill the whole 64-D space; they
    # live near a thin LOW-dimensional surface. We model that with a random
    # `manifold_dim`-dimensional subspace M. EVERY real activation here lies in M
    # (plus tiny noise). This is what makes naive off-manifold steering leak: a
    # raw edit a + s*d points partly OUTSIDE M, into space the model never sees.
    manifold_dim = 24
    raw_basis = torch.randn(dim, manifold_dim, generator=g)
    U, _ = torch.linalg.qr(raw_basis)        # [dim, manifold_dim], orthonormal M

    # A few planted concept directions that LIE IN the manifold M and are only
    # MILDLY separated (NOT orthogonalised) — real concepts share directions and
    # partly overlap, which is exactly why steering one nudges its neighbours.
    n_concepts = 6
    coeff = torch.randn(manifold_dim, n_concepts, generator=g)
    concepts = (U @ coeff).T                  # [n_concepts, dim], inside M
    concepts = concepts / (concepts.norm(dim=1, keepdim=True) + 1e-8)

    # Base activations: random combinations WITHIN the manifold M (+ small off-M
    # noise), so the bank's natural directions are exactly M's directions.
    latent = torch.randn(n_img, n_pat, manifold_dim, generator=g)
    bank = latent @ U.T                       # [n_img, n_pat, dim], lives in M
    bank = bank + 0.05 * torch.randn(n_img, n_pat, dim, generator=g)  # tiny off-M

    # For each image, switch ON a random subset of concepts at random strengths,
    # added to every patch. This gives features the SAE can latch onto.
    for c in range(n_concepts):
        present = (torch.rand(n_img, 1, 1, generator=g) < 0.4).float()   # ~40% on
        amp = present * (1.5 + 2.0 * torch.rand(n_img, 1, 1, generator=g))
        bank = bank + amp * concepts[c].view(1, 1, dim)

    return bank, concepts


def iter_batches(bank, batch: int, seed: int):
    """Yield random mini-batches [batch, n_patches, dim] from the bank forever."""
    import torch

    g = torch.Generator().manual_seed(seed)
    n = bank.shape[0]
    while True:
        idx = torch.randint(0, n, (batch,), generator=g)
        yield bank[idx]


def train_sae(cfg: dict):
    import matplotlib
    matplotlib.use("Agg")                    # headless backend: write PNG, no GUI
    import matplotlib.pyplot as plt
    import torch

    set_seed(cfg["seed"])
    os.makedirs(os.path.join(HERE, cfg["out_dir"]), exist_ok=True)

    # 1) Build the activation bank (the data we train the SAE on).
    bank, planted = make_activation_bank(cfg)
    print(f"[bank] activations shape = {tuple(bank.shape)}  "
          f"(n_images x n_patches x dim); planted concepts = {planted.shape[0]}")

    # 2) Build the project's FaithSAE wrapper, then grab its TopK SAE submodule.
    #    FaithSAE bundles a frozen backbone + the SAE + a steerer; for SAE
    #    training we only need the SAE part (we feed it activations directly).
    FaithSAE = _build()
    model = FaithSAE(cfg)
    sae = model.sae                          # the real TopKSAE from src/model.py
    print(f"[sae]  dictionary size = {cfg['sae_dim']} features, "
          f"top-k = {cfg['topk_k']} active per vector, type = {cfg['sae_type']}")

    # 3) Optimizer: AdamW updates only the SAE's weights (the only trainable part).
    opt = torch.optim.AdamW(sae.parameters(), lr=cfg["lr"])

    # 4) The training loop. Each step: take a batch, reconstruct it, measure MSE,
    #    backpropagate, update. We log the loss so we can plot the curve.
    batches = iter_batches(bank, cfg["batch"], seed=cfg["seed"] + 100)
    losses = []
    for step in range(cfg["steps"]):
        a = next(batches)                    # [batch, n_patches, dim] activations
        a_hat, z, loss = sae(a)              # reconstruct + get sparse code + MSE
        opt.zero_grad()
        loss.backward()                      # compute gradients of the loss
        opt.step()                           # nudge weights to shrink the loss
        losses.append(float(loss.detach()))
        if step % max(1, cfg["steps"] // 8) == 0 or step == cfg["steps"] - 1:
            # Average number of ACTIVE (nonzero) features — should sit near k.
            active = (z != 0).float().sum(-1).mean().item()
            print(f"  step {step:4d}  recon_MSE = {losses[-1]:.4f}  "
                  f"avg_active_features = {active:.1f}")

    # 5) Save the checkpoint: the learned weights + enough config to rebuild it,
    #    AND the planted concepts so step 2/3 can sanity-check rediscovery.
    ckpt_path = os.path.join(HERE, cfg["sae_ckpt"])
    torch.save({
        "state_dict": sae.state_dict(),
        "cfg": {k: cfg[k] for k in ("dim", "d_model", "sae_dim", "topk_k",
                                    "sae_type", "seed", "n_patches", "n_images")},
        "planted_concepts": planted,
        "final_loss": losses[-1],
    }, ckpt_path)
    print(f"[save] SAE checkpoint -> {ckpt_path}  (final recon_MSE = {losses[-1]:.4f})")

    # 6) Plot and save the reconstruction-loss curve.
    png_path = os.path.join(HERE, cfg["loss_png"])
    plt.figure(figsize=(6, 4))
    plt.plot(losses, lw=1.5)
    plt.xlabel("training step")
    plt.ylabel("reconstruction MSE (lower = better copy)")
    plt.title("TopK SAE reconstruction loss — Milestone 3 baseline")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()
    print(f"[save] loss curve PNG  -> {png_path}")

    return losses[-1]


def main():
    ap = argparse.ArgumentParser(description="Train a TopK SAE on a synthetic bank.")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    final = train_sae(cfg)
    print(f"\nDONE step 1. Final reconstruction MSE = {final:.4f}. "
          f"Now run: /usr/bin/python3 step2_select_concepts.py")


if __name__ == "__main__":
    main()
