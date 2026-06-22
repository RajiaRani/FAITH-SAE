"""steering_real.py — the four steering-method variants at real scale.

================================================================================
FAITH-SAE (real-scale run) · author: Rajia Rani
                            �
For research and educational purposes only.
================================================================================

THE PLUGGABLE COMPONENT (DESIGN_BRIEF §12)
------------------------------------------
The "method" the paper studies is the STEERING METHOD. Four variants, selected by
name, each edits a batch of REAL CLIP patch activations toward an SAE concept:

  * naive_steer      a' = a + s * d_hat            off-manifold activation addition
                                                   (ActAdd-style; the main competitor)
  * random_steer     a' = a + s * r_hat            same form, RANDOM unit direction
                                                   (null / sanity baseline)
  * clamp_steer      set the SAE feature's code to a target magnitude, decode back
                                                   (no projection -> can go off-manifold)
  * onmanifold_steer a' = a + s * (P_M · d_hat)    OURS: project the edit onto the
                                                   top-r real-image subspace U_r
                                                   (P_M = U_r U_r^T, brief §14)

These match the SEMANTICS of the toy scaffold's `src/blocks/__init__.py` exactly;
the only differences are (1) they operate on dense real activation batches shaped
[n, d] (n patch tokens x d width) rather than [B, patches, d] toy tensors, and
(2) onmanifold_steer uses the FIXED, pre-estimated real-image basis U_r passed in
`ctx` (estimated once by manifold.estimate_manifold_basis), never a per-batch PCA.
At matched strength `s` (RQ1) all four apply the SAME `s` — the only thing that
differs is the DIRECTION each actually pushes along, which is what CFS measures.

SHARED SIGNATURE (honoured exactly)
-----------------------------------
Every steerer is  fn(acts[n,d], direction[d], strength, ctx) -> acts2[n,d]
where `ctx` is a dict carrying optional extras:
    ctx["U_r"]      : torch [d, r] on-manifold basis        (onmanifold_steer)
    ctx["sae"]      : a TopKSAE with encode()/decode()       (clamp_steer)
    ctx["concept"]  : int feature id to clamp                (clamp_steer)
    ctx["seed"]     : int rng seed                           (random_steer)
    ctx["clamp_target"] : magnitude to clamp the feature to  (clamp_steer; default s)
The registry is `STEER` (name -> fn). `build_steer(name, ctx)` returns the fn.

CLI:  /usr/bin/python3 steering_real.py --smoke
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# --- make src/ and sibling real_run importable (same convention as manifold.py) -
_THIS = pathlib.Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_HERE = _THIS.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the project's off-manifold residual + projection (one source of truth).
# manifold.py is a sibling; this import is light (no open_clip / no GPU).
from manifold import offmanifold_residual, project_onmanifold  # noqa: E402


# --------------------------------------------------------------------------- #
# Registry (mirrors src.model.STEER_REGISTRY / register_steer, scaled to [n,d]).#
# --------------------------------------------------------------------------- #
STEER: dict = {}


def register_steer(name: str):
    """Decorator: register a steering fn under `name` in the STEER registry."""
    def deco(fn):
        STEER[name] = fn
        return fn
    return deco


def build_steer(name: str, ctx: dict | None = None):
    """Return the steering fn for `name`. (ctx is accepted for symmetry with the
    toy build_steer; the variants here read everything they need from the per-call
    ctx, so this just validates the name and hands back the fn.)"""
    if name not in STEER:
        raise KeyError(f"unknown steerer '{name}'. Registered: {sorted(STEER)}")
    return STEER[name]


def _as_unit(d):
    """Unit-normalize a direction so `strength` is a clean, comparable knob across
    variants (matched-strength comparison, RQ1). Adds a tiny eps for safety."""
    import torch
    if not torch.is_tensor(d):
        d = torch.as_tensor(d)
    d = d.float()
    return d / (d.norm() + 1e-8)


# --------------------------------------------------------------------------- #
# naive_steer — off-manifold activation addition  a' = a + s * d_hat           #
# --------------------------------------------------------------------------- #
@register_steer("naive_steer")
def naive_steer(acts, direction, strength, ctx=None):
    """Add the WHOLE (unit) concept direction to every token. No projection, so
    the edit largely leaves the real-image manifold -> high off-manifold residual,
    high apparent effect but low real faithfulness. This is the field's standard
    ActAdd-style edit and the main competitor (brief §5)."""
    import torch
    a = acts if torch.is_tensor(acts) else torch.as_tensor(acts)
    a = a.float()
    d = _as_unit(direction).to(a.dtype)
    return a + float(strength) * d                     # broadcasts d[d] over rows [n,d]


# --------------------------------------------------------------------------- #
# random_steer — same form, a FIXED random unit direction (null baseline)       #
# --------------------------------------------------------------------------- #
@register_steer("random_steer")
def random_steer(acts, direction, strength, ctx=None):
    """Null/sanity baseline: add a fixed RANDOM unit direction of matched norm.
    There is no real concept along it, so a faithful metric should see near-zero
    monotonicity -> CFS collapses (the corner of the Pareto plot, fig2)."""
    import torch
    ctx = ctx or {}
    a = acts if torch.is_tensor(acts) else torch.as_tensor(acts)
    a = a.float()
    d_in = direction if torch.is_tensor(direction) else torch.as_tensor(direction)
    seed = int(ctx.get("seed", 0)) + 13
    g = torch.Generator().manual_seed(seed)            # fixed -> reproducible null
    r = torch.randn(d_in.shape[-1], generator=g)
    r = _as_unit(r).to(a.dtype)
    return a + float(strength) * r


# --------------------------------------------------------------------------- #
# clamp_steer — clamp the SAE feature's code to a target magnitude, decode back #
# --------------------------------------------------------------------------- #
@register_steer("clamp_steer")
def clamp_steer(acts, direction, strength, ctx=None):
    """Set the target SAE feature's coefficient to a fixed magnitude and decode,
    REPLACING that feature's reconstructed contribution while preserving the rest
    (residual + other features). No manifold projection -> can drift off-manifold.

    Effective edit per token:  a' = a + (decode(z_clamped) - decode(z))  where z is
    the encoding and z_clamped is z with z[..., concept] = clamp_target. This keeps
    the parts of `a` the SAE cannot reconstruct (the residual) exactly, matching the
    toy clamp_steer semantics. Falls back to naive if no sae/concept is provided.
    """
    import torch
    ctx = ctx or {}
    a = acts if torch.is_tensor(acts) else torch.as_tensor(acts)
    a = a.float()

    sae = ctx.get("sae", None)
    concept = ctx.get("concept", None)
    if sae is None or concept is None:
        # No dictionary available -> degrade to naive (still a valid off-manifold edit).
        return naive_steer(a, direction, strength, ctx)

    # clamp_target defaults to the strength knob so a strength sweep also sweeps
    # the clamp magnitude (the natural "turn the feature up" interpretation).
    target = float(ctx.get("clamp_target", strength))

    with torch.no_grad():
        enc = sae.encode(a)                            # may be z or (z, pre_acts)
        z = enc[0] if isinstance(enc, (tuple, list)) else enc
        recon = sae.decode(z)                          # current reconstruction
        z_clamped = z.clone()
        z_clamped[..., int(concept)] = target          # hard-clamp the target feature
        recon_clamped = sae.decode(z_clamped)
        # add only the CHANGE the clamp made; everything else (residual) is preserved.
        return a + (recon_clamped - recon)


# --------------------------------------------------------------------------- #
# onmanifold_steer — OURS: a' = a + s * (P_M · d_hat),  P_M = U_r U_r^T         #
# --------------------------------------------------------------------------- #
@register_steer("onmanifold_steer")
def onmanifold_steer(acts, direction, strength, ctx=None):
    """Project the (unit) concept direction onto the FIXED top-r real-image
    subspace U_r before adding it, so the edit stays in the region the frozen
    model actually uses on real images -> off-manifold residual ~ 0, edits stay
    realistic and decodable (brief §3, §14).

    U_r is the pre-estimated basis from manifold.estimate_manifold_basis, passed
    as ctx["U_r"] (or ctx["basis"]). It MUST be the fixed real-image basis, not a
    per-batch PCA: projecting onto the live test batch would let the "manifold"
    follow the OOD shift and hide exactly the effect RQ3 measures. If no basis is
    supplied we fall back to naive (the r -> inf / P_M = I degenerate case), which
    keeps the smoke path runnable but is NOT the intended real configuration."""
    import torch
    ctx = ctx or {}
    a = acts if torch.is_tensor(acts) else torch.as_tensor(acts)
    a = a.float()
    d = _as_unit(direction)

    U_r = ctx.get("U_r", ctx.get("basis", None))
    if U_r is None:
        # No fixed basis -> degenerate to naive (P_M = I). Real runs always pass U_r.
        return a + float(strength) * d.to(a.dtype)

    d_proj = project_onmanifold(d, U_r)                # P_M · d_hat  [d]
    return a + float(strength) * d_proj.to(a.dtype)


# --------------------------------------------------------------------------- #
# Convenience: the EFFECTIVE edit (a' - a) a steerer applies, averaged over     #
# tokens, for the off-manifold-residual diagnostic (reused by cfs_eval / plots).#
# --------------------------------------------------------------------------- #
def effective_edit(name, acts, direction, strength, ctx=None):
    """Return the mean effective edit direction (a' - a) over the token batch.
    This is the vector whose off-manifold residual against U_r tells us how far
    method `name` left the manifold (naive/random large, on-manifold ~0)."""
    import torch
    steer = build_steer(name, ctx)
    a = acts if torch.is_tensor(acts) else torch.as_tensor(acts)
    a = a.float()
    a2 = steer(a, direction, strength, ctx)
    return (a2 - a).reshape(-1, a.shape[-1]).mean(0)   # [d]


# offmanifold_residual is re-exported from manifold.py so callers can do
# `from steering_real import offmanifold_residual` per the module-signature spec.
__all__ = [
    "STEER", "register_steer", "build_steer",
    "naive_steer", "random_steer", "clamp_steer", "onmanifold_steer",
    "effective_edit", "offmanifold_residual", "project_onmanifold",
]


# --------------------------------------------------------------------------- #
# Smoke: fabricate a real-SHAPED activation batch + a fixed U_r, run all four    #
# steerers, and verify the ORDERING the paper relies on:                        #
#   off-manifold residual:  onmanifold << naive, clamp; random ~ large.         #
# No open_clip / no GPU.                                                         #
# --------------------------------------------------------------------------- #
class _ToySAE:
    """A minimal stand-in TopK SAE with encode/decode for the clamp smoke path.
    (The real clamp_steer receives sae_real.TopKSAE; the interface is the same.)"""
    def __init__(self, d, h, k=8, seed=0):
        import torch
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(h, d, generator=g)
        self.W_enc = W / (W.norm(dim=1, keepdim=True) + 1e-8)   # [h, d]
        self.W_dec = self.W_enc.T.contiguous()                  # tied [d, h]
        self.k = k

    def encode(self, a):
        import torch
        z = torch.relu(a @ self.W_enc.T)                        # [n, h]
        k = min(self.k, z.shape[-1])
        thresh = z.topk(k, dim=-1).values[..., -1:]
        return z * (z >= thresh)

    def decode(self, z):
        return z @ self.W_dec.T                                  # [n, d]


def _smoke() -> int:
    import numpy as np
    import torch

    from manifold import estimate_manifold_basis

    torch.manual_seed(0)
    np.random.seed(0)

    # Real-SHAPED batch: n patch tokens x d = ViT-L/14 width. Plant a low-rank
    # sheet (the manifold) so we can build a meaningful fixed U_r, then estimate it.
    d, n, true_r = 1024, 2048, 24
    Q, _ = torch.linalg.qr(torch.randn(d, true_r))
    coords = torch.randn(n, true_r) * torch.linspace(3.0, 1.0, true_r)
    acts = (coords @ Q.T) + 0.02 * torch.randn(n, d)            # on-sheet + wobble [n, d]

    U_r = estimate_manifold_basis(acts, 64)                     # fixed real-image basis
    # A concept direction that LIVES on the sheet (a realistic SAE concept).
    direction = Q[:, 0].clone()
    ctx = {"U_r": U_r, "seed": 0, "sae": _ToySAE(d, 128, k=8), "concept": 3}
    s = 4.0

    print(f"[smoke] acts {tuple(acts.shape)}  U_r {tuple(U_r.shape)}  strength={s}")
    print(f"  {'variant':<18} {'out shape':>14} {'off-manifold residual':>24}")
    print("  " + "-" * 60)

    residuals = {}
    for name in ("naive_steer", "random_steer", "clamp_steer", "onmanifold_steer"):
        steer = build_steer(name, ctx)
        out = steer(acts, direction, s, ctx)
        assert out.shape == acts.shape, f"{name}: shape {tuple(out.shape)} != {tuple(acts.shape)}"
        edit = effective_edit(name, acts, direction, s, ctx)
        res = offmanifold_residual(edit, U_r)
        residuals[name] = res
        print(f"  {name:<18} {str(tuple(out.shape)):>14} {res:>24.4f}")

    # The headline ordering the method depends on: on-manifold edit stays on the
    # sheet (residual ~ 0); naive/clamp/random leave it (residual large).
    on = residuals["onmanifold_steer"]
    nv = residuals["naive_steer"]
    rnd = residuals["random_steer"]
    print(f"\n[smoke] residual check: onmanifold {on:.4f} < naive {nv:.4f}  "
          f"-> {'PASS' if on < nv else 'FAIL'}")
    assert on < nv, "onmanifold residual must be below naive"
    assert on < 0.1, "onmanifold edit should be (almost) fully on-manifold"
    assert rnd > 0.5, "random direction should be mostly off-manifold"

    # naive == onmanifold when U_r is full rank (P_M = I): the r->inf degenerate case.
    U_full = estimate_manifold_basis(acts, d, center=False)
    out_naive = naive_steer(acts, direction, s, ctx)
    out_onm_full = onmanifold_steer(acts, direction, s, {"U_r": U_full})
    degen_err = float((out_naive - out_onm_full).abs().max())
    print(f"[smoke] full-rank degeneracy ||naive - onmanifold(P_M=I)|| = {degen_err:.2e}  (~0)")
    assert degen_err < 1e-2

    # Matched-strength sanity: every steerer applies the same scalar s; only the
    # direction differs (the controlled comparison RQ1 requires).
    print(f"[smoke] registered steerers: {sorted(STEER)}")
    print("[smoke] steering_real.py PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="FAITH-SAE real-scale steering variants.")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU path on real-SHAPED tensors (no open_clip/GPU)")
    args = ap.parse_args()
    # This module has no heavy real driver of its own (it is a library used by
    # cfs_eval / ood_sweep); --smoke is the only runnable entry, default to it.
    return _smoke()


if __name__ == "__main__":
    raise SystemExit(main())
