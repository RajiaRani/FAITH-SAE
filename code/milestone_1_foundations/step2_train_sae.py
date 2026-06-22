"""step2_train_sae.py
================================================================================
STEP 2 of 5 — A SPARSE AUTOENCODER (SAE) AND ITS "CONCEPT SWITCHES"
================================================================================
Run me:   /usr/bin/python3 step2_train_sae.py
(Step 2 re-trains its own tiny SAE; you do NOT need to have run step1 first.)

WHAT YOU LEARN HERE
-------------------
What a Sparse Autoencoder is, what "top-k sparsity" means, what a "concept
direction" is, and how to train the toy SAE so its reconstruction loss drops.

-------------------------------------------------------------------------------
TERM 3 — SPARSE AUTOENCODER (SAE)  +  CONCEPT DIRECTIONS / SWITCHES
-------------------------------------------------------------------------------
  Definition: An autoencoder is a network that ENCODES an input into a code, then
              DECODES the code back into (approximately) the original input. A
              *sparse* autoencoder forces the code to be mostly zeros — only a few
              entries may be non-zero. Each code entry behaves like a labelled
              switch for one concept; the decoder column that switch turns on is
              that concept's "direction" in activation space.
  Analogy:    A mixing board. A messy live sound (the activation) is broken into a
              few labelled faders — "bass", "vocals", "drums" (the switches). Most
              faders sit at zero; only a few are up. Push them back up and you
              rebuild the sound (the decode). Each fader's effect on the speakers
              is its "direction".
  Tiny number: activation a = [0.7,-1.2,0.0,0.3]  --encode-->  code z =
              [0, 0, 2.1, 0]  (only switch #3 is on)  --decode-->  a_hat ~= a.
              Switch #3's decoder column, e.g. [0.33,-0.57,0.0,0.14], is its
              "concept direction".

-------------------------------------------------------------------------------
TERM 4 — TOP-K SPARSITY
-------------------------------------------------------------------------------
  Definition: Top-k means: of all the switches, keep only the k largest ON and
              force every other switch to exactly zero.
  Analogy:    A talent show where only the top 2 acts advance; everyone else is
              sent home (set to zero), no matter how close they were.
  Tiny number: raw switch values [0.1, 3.0, 0.2, 2.5] with k=2 -> keep the two
              biggest -> [0, 3.0, 0, 2.5]. Exactly 2 switches survive.
  WHY: Sparsity is what makes the switches INTERPRETABLE. If all 128 switches
       could be on at once, no single one would mean a clean concept. Forcing
       only k=8 on pushes the SAE to give each concept its own dedicated switch
       (this is the Gao et al. 2024 "TopK SAE"). It is also ablation A2 (the k
       sweep) in the design brief.

-------------------------------------------------------------------------------
WHAT "TRAINING" MEANS HERE
-------------------------------------------------------------------------------
Training = nudging the SAE's numbers so that decode(encode(a)) ~= a. We measure
how wrong it is with "reconstruction loss" (mean squared error between a and its
rebuild a_hat). Loss going DOWN = the SAE is learning to represent activations
with just a few switches. The backbone stays frozen the whole time; only the SAE
learns.
"""
from __future__ import annotations

from _common import banner, load_cfg

# Reuse the project's real training loop and SAE — we do not re-implement them.
from src.train import train
from src.utils import count_params


def main() -> None:
    cfg = load_cfg()
    banner("STEP 2 — TRAIN THE TINY TOP-K SPARSE AUTOENCODER")

    print(f"SAE dictionary size (switches) : {cfg['sae_dim']}")
    print(f"top-k (switches allowed ON)    : {cfg['topk_k']}   "
          f"-> {cfg['sae_dim'] - cfg['topk_k']} switches forced to ZERO per patch")

    # --- Train the SAE. src.train.train returns (model, final_loss).
    # Only SAE weights carry gradients; the frozen backbone does not move.
    print(f"\nTraining for {cfg['steps']} steps (only the SAE learns; backbone frozen)...\n")
    model, final_loss = train(cfg, steps=cfg["steps"])

    total = count_params(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel weights total            : {total}")
    print(f"Model weights TRAINABLE (=SAE) : {trainable}   (backbone contributes 0)")
    print(f"Final reconstruction loss      : {final_loss:.4f}   "
          f"(lower = SAE rebuilds activations better)")

    # --- Make the "concept switches" concrete: encode a few activations and show
    # how few switches are actually ON, then show one concept's DIRECTION.
    import torch
    from src.data import synthetic_batch

    x, _ = synthetic_batch(batch=4, n_patches=cfg["n_patches"], dim=cfg["dim"], seed=123)
    with torch.no_grad():
        a = model.activations(x)          # frozen-backbone activations
        z = model.sae.encode(a)           # the sparse code (the switches)

    # Count, for the very first patch, how many switches are non-zero.
    z0 = z[0, 0]
    n_on = int((z0 != 0).sum())
    print(f"\nFor one patch, switches ON     : {n_on}  (should be <= k = {cfg['topk_k']})")
    assert n_on <= cfg["topk_k"], "top-k sparsity violated!"

    # Show concept #0's direction = decoder column 0 (the activation-space vector
    # that switch #0 turns on). This is exactly what we will STEER in step3/4.
    d0 = model.sae.concept_direction(0)
    show = ", ".join(f"{v:+.2f}" for v in d0[:6].tolist())
    print(f"Concept #0 'direction' (first 6 of {cfg['dim']} numbers): [{show}, ...]")

    print("\n[STEP 2 OK]  You trained a TopK SAE; it represents each activation")
    print("with only a few ON switches, and each switch has a concept direction.")
    print("Next: step3 plants a KNOWN concept signal so we have a ground truth.")


if __name__ == "__main__":
    main()
