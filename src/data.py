"""Data: synthetic activations with a planted concept (day-one smoke path).

`synthetic_batch` is the analog of the needle task: it generates Gaussian
"patch activations" and injects ONE known concept direction at a controllable
magnitude, so the eval can verify a steer actually recovers/moves that planted
concept. `image_stream` is the hook you fill in for real CLIP ViT-B/16 features
over ImageNet (and its shifts), at a fixed equal budget across variants.
"""
from __future__ import annotations


def planted_concept(dim: int, seed: int = 0):
    """The fixed ground-truth concept direction injected into activations."""
    import torch
    g = torch.Generator().manual_seed(seed + 7)
    d = torch.randn(dim, generator=g)
    return d / (d.norm() + 1e-8)


def synthetic_batch(batch: int = 16, n_patches: int = 16, dim: int = 64,
                    seed: int = 0, concept_strength: float = 1.0):
    """A toy bank of patch activations with a planted concept.

    Returns (x, a_target):
      x        — backbone INPUTS [B, n_patches, dim] (Gaussian);
      a_target — the same tensor with the planted concept added at a known
                 magnitude per item, used as the SAE reconstruction target proxy
                 and as ground truth for the concept readout.
    """
    import torch
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, n_patches, dim, generator=g)
    d = planted_concept(dim, seed=0)
    # plant the concept on a random subset of items (the "concept present" label)
    amp = (torch.rand(batch, 1, 1, generator=g) * concept_strength)
    a_target = x + amp * d
    return x, a_target


def concept_readout(a, dim: int = 64):
    """Held-out linear probe: projection of activations onto the planted concept.

    Mean over patches of <a, d>. This is the readout the CFS monotonicity check
    correlates against the steering knob (TODO(M2): a learned probe on real CLIP)."""
    import torch
    d = planted_concept(dim, seed=0)
    return (a * d).sum(-1).mean(-1)              # [B]


def image_stream(path: str, n_patches: int):
    """TODO(M2): yield batches of REAL CLIP ViT-B/16 patch activations.

    Keep the activation budget identical across all steering variants so the
    faithfulness comparison stays controlled (roadmap Milestone 1). Wire
    open_clip / a cached ImageNet activation bank (+ ImageNet-R/Sketch/C/ObjectNet
    for the OOD sweep) here.
    """
    raise NotImplementedError("Wire your CLIP+ImageNet activation loader here (TODO M2).")
