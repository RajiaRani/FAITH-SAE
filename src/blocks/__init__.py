"""Steering-method variants — the pluggable component of FAITH-SAE.

Registers the four steerers from the design brief (the *steering method* is the
pluggable component selected by config name, not an attention block):

  * 'naive_steer'     — off-manifold activation addition  a <- a + s*d   (baseline,
                        ships in base; the main competitor the field admits is
                        unreliable, ActAdd-style; Turner 2023).
  * 'random_steer'    — same form, RANDOM direction (null/sanity baseline).
  * 'clamp_steer'     — clamp the SAE feature to a fixed magnitude, no projection.
  * 'onmanifold_steer'— OURS: project the edit onto the top-r real-image subspace,
                        a <- a + s*(P_M d) with P_M = U_r U_r^T (brief eq. §14).

Design knobs (cfg): `steer_strength` (s), `proj_rank` (r). naive_steer is the
r -> inf (P_M = I) special case of onmanifold_steer.

Roadmap RQs: RQ1 (on-manifold vs naive/random/clamp/TCAV CFS, matched strength),
RQ2 (CFS decomposition + the strength x proj_rank design grid), RQ3 (CFS survival
across the OOD shift ladder). Real CLIP+ImageNet basis estimation is TODO(M2).
"""
from __future__ import annotations

from ..model import register_steer


def _impl():
    import torch

    def _as_dir(d):
        """Unit-normalize a direction (keeps strength interpretation clean)."""
        return d / (d.norm() + 1e-8)

    @register_steer("naive_steer")
    def _naive(cfg):
        def steer(a, d, strength, sae=None, concept=None, basis=None):
            # Off-manifold activation addition: a <- a + s*d  (no projection).
            return a + strength * _as_dir(d)
        return steer

    @register_steer("random_steer")
    def _random(cfg):
        seed = cfg.get("seed", 0)

        def steer(a, d, strength, sae=None, concept=None, basis=None):
            # Null baseline: add a fixed random direction of matched norm.
            g = torch.Generator().manual_seed(seed + 13)
            r = torch.randn(d.shape, generator=g)
            return a + strength * _as_dir(r)
        return steer

    @register_steer("clamp_steer")
    def _clamp(cfg):
        def steer(a, d, strength, sae=None, concept=None, basis=None):
            # Clamp the SAE feature's coefficient to a fixed magnitude, then
            # decode back -- no manifold projection, so it can drift off-manifold.
            if sae is None or concept is None:
                return a + strength * _as_dir(d)
            z = sae.encode(a)
            z = z.clone()
            z[..., concept] = strength            # hard clamp the target feature
            a_clamped = sae.decode(z)
            # residual outside the SAE dictionary is preserved
            return a + (a_clamped - sae.decode(sae.encode(a)))
        return steer

    @register_steer("onmanifold_steer")
    def _onmanifold(cfg):
        r = cfg.get("proj_rank", 16)

        def _basis(a, basis):
            # P_M onto top-r real-image subspace. TODO(M2): estimate U_r once from
            # a large bank of REAL CLIP activations; here we PCA the live batch as
            # a stand-in so the smoke path runs forward+backward.
            if basis is not None:
                return basis
            flat = a.reshape(-1, a.shape[-1])
            flat = flat - flat.mean(0, keepdim=True)
            # economy SVD; right singular vectors are the activation principal axes
            _, _, Vh = torch.linalg.svd(flat, full_matrices=False)
            rr = min(r, Vh.shape[0])
            return Vh[:rr].T                      # [d, r] = U_r

        def steer(a, d, strength, sae=None, concept=None, basis=None):
            U = _basis(a, basis)                  # [d, r]
            d = _as_dir(d)
            d_proj = U @ (U.T @ d)                # P_M d  (top-r component)
            return a + strength * d_proj
        return steer


_impl()
