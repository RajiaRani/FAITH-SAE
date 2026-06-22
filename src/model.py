"""A minimal frozen-backbone + TopK-SAE harness with a *pluggable steering method*.

The research lives in the steerer: topic scaffolds register variants
(naive_steer, random_steer, clamp_steer, onmanifold_steer, ...) into
STEER_REGISTRY and select them by name in the config. The base ships the naive
off-manifold activation-addition steerer so the repo runs (and has a falsifiable
comparison point) on day one.

Pieces (the "TinyGPT" analog for vision interpretability):
  * Backbone  — a small RANDOM-FROZEN MLP standing in for CLIP ViT-B/16; it just
                produces synthetic patch activations. No grads, no download.
  * TopK SAE  — encoder/decoder dictionary with top-k sparsity (Gao et al. 2024);
                the only trainable part on the smoke path.
  * Steerer   — the pluggable component: edits a chosen SAE concept inside the
                activation space (build_steer(name, cfg)).
Real CLIP ViT-B/16 + ImageNet activations are TODO(M2).
"""
from __future__ import annotations

STEER_REGISTRY: dict = {}


def register_steer(name: str):
    def deco(fn):
        STEER_REGISTRY[name] = fn
        return fn
    return deco


def build_steer(name: str, cfg: dict):
    if name not in STEER_REGISTRY:
        raise KeyError(f"unknown steerer '{name}'. Registered: {sorted(STEER_REGISTRY)}")
    return STEER_REGISTRY[name](cfg)


def _build():  # deferred so `import src.model` works without torch (docs/lint)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FrozenBackbone(nn.Module):
        """Random-frozen MLP -> synthetic 'patch activations'. Stands in for a
        frozen CLIP ViT-B/16 trunk; weights never receive gradients."""

        def __init__(self, cfg):
            super().__init__()
            d = cfg["d_model"]
            self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            for p in self.parameters():
                p.requires_grad_(False)

        @torch.no_grad()
        def forward(self, x):                       # x: [B, n_patches, d]
            return self.net(x)

    class TopKSAE(nn.Module):
        """Tied-ish TopK sparse autoencoder over backbone activations.

        z = TopK(W_enc (a - b_dec) + b_enc);  a_hat = W_dec z + b_dec.
        The decoder columns are the *concept directions* we steer (Gao 2024)."""

        def __init__(self, cfg):
            super().__init__()
            d, h = cfg["d_model"], cfg["sae_dim"]
            self.k = cfg.get("topk_k", 8)
            self.sae_type = cfg.get("sae_type", "topk")
            self.enc = nn.Linear(d, h)
            self.dec = nn.Linear(h, d, bias=False)
            self.b_dec = nn.Parameter(torch.zeros(d))

        def encode(self, a):
            z = F.relu(self.enc(a - self.b_dec))
            if self.sae_type == "topk":               # A1 ablation: topk vs l1
                k = min(self.k, z.shape[-1])
                thresh = z.topk(k, dim=-1).values[..., -1:]
                z = z * (z >= thresh)
            return z

        def decode(self, z):
            return self.dec(z) + self.b_dec

        def forward(self, a):
            z = self.encode(a)
            a_hat = self.decode(z)
            l1 = z.abs().mean()
            loss = F.mse_loss(a_hat, a) + (1e-3 * l1 if self.sae_type == "l1" else 0.0)
            return a_hat, z, loss

        def concept_direction(self, concept: int):
            """Decoder column for `concept` = the activation-space direction `d`."""
            return self.dec.weight[:, concept]

    class FaithSAE(nn.Module):
        """Backbone + SAE + a pluggable steerer selected by cfg['steer']."""

        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.backbone = FrozenBackbone(cfg)
            self.sae = TopKSAE(cfg)
            self.steer = build_steer(cfg.get("steer", "naive_steer"), cfg)

        def activations(self, x):
            return self.backbone(x)

        def forward(self, x, targets=None):
            """Train signal = SAE reconstruction of frozen activations."""
            a = self.activations(x)
            a_hat, z, loss = self.sae(a)
            return a_hat, loss

        def steered_activations(self, x, concept: int, strength: float, basis=None):
            """Apply the selected steerer to concept `concept` at knob `strength`."""
            a = self.activations(x)
            d = self.sae.concept_direction(concept)
            return self.steer(a, d, strength, sae=self.sae, concept=concept, basis=basis)

    return FaithSAE


def make_model(cfg: dict):
    FaithSAE = _build()
    return FaithSAE(cfg)
