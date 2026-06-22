"""step1_backbone_activations.py
================================================================================
STEP 1 of 5 — A FROZEN VISION BACKBONE AND ITS "ACTIVATIONS"
================================================================================
Run me:   /usr/bin/python3 step1_backbone_activations.py

WHAT YOU LEARN HERE
-------------------
What an "activation" is, what a "frozen model" is, and how to get a batch of
synthetic activation vectors out of the project's tiny stand-in backbone.

-------------------------------------------------------------------------------
TERM 1 — ACTIVATION (of a vision model)
-------------------------------------------------------------------------------
  Definition: An activation is the list of numbers a neural network computes for
              a piece of input as that input flows through the network. For a
              vision model, each little square tile ("patch") of an image gets
              its own vector of numbers describing what the model "sees" there.
  Analogy:    Think of a panel of light meters on a camera. Point it at a scene
              and each meter shows a reading. The whole panel of readings is the
              "activation" — a fingerprint of what is in front of the lens.
  Tiny number: With dim=4, one patch's activation might be [0.7, -1.2, 0.0, 0.3].
              Four numbers = one patch. An image of 16 patches => 16 such vectors.

-------------------------------------------------------------------------------
TERM 2 — FROZEN MODEL
-------------------------------------------------------------------------------
  Definition: A frozen model is a network whose weights are LOCKED — they never
              change / never learn during our experiment. We only read its
              outputs; we never train it.
  Analogy:    A ruler. You measure things with it; you do not bend the ruler to
              fit the thing you measure. The ruler stays fixed so measurements
              are comparable.
  Tiny number: A frozen layer with weight w=2.0 turns input 3.0 into 6.0 today,
              tomorrow, always — w stays 2.0 forever.
  WHY FROZEN: Interpretability studies a model AS IT IS. If the backbone kept
              changing, we could never say "this concept lives here" — the target
              would move. So the real FAITH-SAE freezes CLIP ViT-B/16; here we
              freeze a tiny random MLP that just produces activation vectors.

-------------------------------------------------------------------------------
WHY SYNTHETIC?
-------------------------------------------------------------------------------
A real backbone (CLIP ViT-B/16) is ~300 MB to download and needs images. To
learn the PIPELINE we replace it with a tiny random-frozen MLP that emits
made-up activation vectors. Same shapes, same code path, zero downloads, runs on
CPU in milliseconds. The real backbone swaps in at Milestone 2.
"""
from __future__ import annotations

from _common import banner, load_cfg

# We import the REAL research classes from src/ (added to sys.path in _common):
#   * FrozenBackbone via make_model -> the frozen MLP stand-in
#   * synthetic_batch -> the toy bank of activations with a planted concept
from src.data import synthetic_batch
from src.model import make_model
from src.utils import count_params, set_seed


def main() -> None:
    cfg = load_cfg()
    set_seed(cfg["seed"])     # fix the dice so the output is identical every run

    banner("STEP 1 — A FROZEN BACKBONE PRODUCES ACTIVATIONS")

    # --- Build the model. make_model returns a FaithSAE = backbone + SAE + steerer.
    # In THIS step we only touch its .backbone (the frozen part).
    model = make_model(cfg)

    # PROOF the backbone is frozen: count how many of its numbers can learn.
    bb_total = count_params(model.backbone)
    bb_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    print(f"\nBackbone weights total      : {bb_total}")
    print(f"Backbone weights TRAINABLE  : {bb_trainable}   <-- 0 means fully FROZEN")
    assert bb_trainable == 0, "backbone should be frozen!"

    # --- Make one batch of synthetic backbone INPUTS.
    # synthetic_batch returns (x, a_target):
    #   x        = raw inputs the backbone reads, shape [batch, n_patches, dim]
    #   a_target = the same with a known concept planted in (we use that in step3)
    x, _ = synthetic_batch(
        batch=8, n_patches=cfg["n_patches"], dim=cfg["dim"], seed=cfg["seed"]
    )
    print(f"\nInput batch shape           : {tuple(x.shape)}  "
          f"= [batch, n_patches, dim]  (8 images x {cfg['n_patches']} patches x {cfg['dim']} numbers)")

    # --- Push inputs through the FROZEN backbone to get ACTIVATIONS.
    a = model.activations(x)      # no gradients flow; weights never move
    print(f"Activation batch shape      : {tuple(a.shape)}  (same shape: 1 vector per patch)")

    # --- Show ONE patch's activation vector so an activation is concrete, not abstract.
    one_patch = a[0, 0]           # image 0, patch 0
    show = ", ".join(f"{v:+.2f}" for v in one_patch[:6].tolist())
    print(f"\nActivation of image 0, patch 0 (first 6 of {cfg['dim']} numbers):")
    print(f"   [{show}, ...]")
    print("\nThat vector of numbers IS the 'activation' — what the frozen model")
    print("'sees' at that patch. Every later step operates on vectors like this.")

    print("\n[STEP 1 OK]  You now have a frozen backbone and a batch of activations.")
    print("Next: step2 learns the Sparse Autoencoder that breaks each activation")
    print("into a few interpretable 'concept switches'.")


if __name__ == "__main__":
    main()
