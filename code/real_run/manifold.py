"""manifold.py — estimate and cache the on-manifold subspace U_r.

================================================================================
FAITH-SAE (real-scale run) · author: Rajia Rani
                            �
For research and educational purposes only.
================================================================================

WHAT THIS MODULE IS FOR (the "P_M" half of the headline method)
---------------------------------------------------------------
The paper's method (DESIGN_BRIEF §3, §14) edits an activation toward an SAE
concept but CONSTRAINS that edit to the *on-manifold subspace* — the directions
the frozen vision model actually uses on real images:

        a' = a + s * (P_M · Delta),      P_M = U_r U_r^T

`U_r` is a [d, r] matrix whose r orthonormal columns are the top-r principal
directions of a large bank of REAL CLIP patch activations. We estimate it ONCE
(an SVD of a centered activation bank), cache it to disk, and every steerer /
every concept / every OOD shift level reuses the SAME fixed basis. Estimating
U_r per-batch would let the "manifold" drift with the test distribution and
defeat the purpose — the whole point is to project onto the subspace the model
learned on its TRAINING distribution.

WHY SVD (not a fitted sklearn PCA object)
-----------------------------------------
At real scale the bank is millions of d=1024 vectors. We only ever need the top-r
RIGHT singular vectors of the centered bank X (X = U S V^T  =>  columns of V are
the principal activation axes, identical to PCA components up to sign). torch's
`linalg.svd` on a centered float32 bank gives exactly that, runs on GPU, and
avoids carrying a heavyweight estimator object around. The math matches the toy
scaffold's `src/.../step2_estimate_subspace.py` (which used sklearn PCA on a tiny
bank); here we re-implement the heavy part for scale but keep identical SEMANTICS.

The off-manifold residual diagnostic and the projection math are REUSED from the
project's `src/utils.py` so there is exactly one source of truth for the math.

PUBLIC API (honoured exactly by the rest of the pipeline)
---------------------------------------------------------
    estimate_manifold_basis(bank[torch n,d], r) -> U_r[d, r]   (torch.float32)
    save_basis(U_r, path)        / load_basis(path) -> U_r
    project_onmanifold(delta, U_r) -> U_r @ (U_r.T @ delta)    (= P_M · delta)
    offmanifold_residual(edit, U_r) -> fraction of `edit` outside span(U_r)

CLI:  /usr/bin/python3 manifold.py --smoke   (tiny CPU path, no open_clip / no GPU)
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# --- make `src/` (the project's real math) importable, plus sibling real_run --
# parents[2] of this file = the 25_..._FAITH_SAE project root, which contains src/.
_THIS = pathlib.Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_HERE = _THIS.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the project's analytic off-manifold residual (one source of truth for the
# math). Guarded so the module still imports if torch is momentarily unavailable
# during pure-lint passes; the real path always has torch.
try:
    from src.utils import onmanifold_projection_residual as _src_residual
except Exception:  # pragma: no cover - only hit in a torch-less lint pass
    _src_residual = None


# --------------------------------------------------------------------------- #
# Estimation: U_r = top-r right singular vectors of the CENTERED bank.         #
# --------------------------------------------------------------------------- #
def estimate_manifold_basis(bank, r: int, center: bool = True):
    """Estimate the on-manifold basis U_r [d, r] from a real-activation bank.

    Args:
        bank:   torch.Tensor [n, d] of real CLIP patch activations (a sample of
                the model's residual stream on real images; built by
                data_real.load_activation_bank). May be float16 on disk; we
                upcast to float32 for a numerically stable SVD.
        r:      number of principal directions to keep (the projection rank; the
                core A3 knob). Clamped to <= d and <= n.
        center: subtract the bank mean first (PCA convention). The mean is the
                bulk activation offset; we want the directions of VARIATION, not
                the offset, so we center by default.

    Returns:
        U_r: torch.FloatTensor [d, r], ORTHONORMAL columns = the top-r principal
             activation directions. P_M = U_r U_r^T then projects onto the
             on-manifold subspace.

    Note: when r == d (full rank) span(U_r) = R^d and P_M = I, so on-manifold
    steering degenerates EXACTLY into naive steering (brief §14: naive is the
    r -> inf / P_M = I special case). That identity is a useful sanity anchor.
    """
    import torch

    if not torch.is_tensor(bank):
        bank = torch.as_tensor(bank)
    X = bank.float()                                   # upcast (float16 -> float32)
    if X.dim() != 2:
        X = X.reshape(-1, X.shape[-1])                 # tolerate [.., d] inputs
    n, d = X.shape
    r = int(max(1, min(int(r), d, n)))                 # valid rank band

    if center:
        X = X - X.mean(dim=0, keepdim=True)            # center: keep variation only

    # Economy SVD: X = U S V^T. The columns of V (rows of Vh) are the principal
    # activation axes, ordered by singular value (== variance). We keep the top r.
    # full_matrices=False keeps it [n,k]/[k,k]/[k,d] — cheap even for large n.
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)   # Vh: [min(n,d), d]
    U_r = Vh[:r].T.contiguous()                           # [d, r], columns = PCs
    return U_r


# --------------------------------------------------------------------------- #
# Projection helpers (P_M · delta and its off-manifold residual).             #
# --------------------------------------------------------------------------- #
def project_onmanifold(delta, U_r):
    """Project an edit onto the on-manifold subspace: P_M · delta = U_r (U_r^T delta).

    Works for a single direction [d] or a batch of edits [n, d] (projects every
    row). We never materialise the [d, d] matrix P_M = U_r U_r^T — applying U_r
    twice is far cheaper when d is large (1024-1280) and r is moderate (<=512).
    """
    import torch

    if not torch.is_tensor(delta):
        delta = torch.as_tensor(delta)
    if not torch.is_tensor(U_r):
        U_r = torch.as_tensor(U_r)
    delta = delta.float()
    U = U_r.float()                                    # [d, r]
    if delta.dim() == 1:                               # single direction [d]
        return U @ (U.T @ delta)                       # [d]
    # batch [n, d]: coords = delta @ U -> [n, r]; rebuild -> [n, d]
    return (delta @ U) @ U.T


def offmanifold_residual(edit, U_r) -> float:
    """Fraction of `edit` that lies OFF the subspace span(U_r):

        ||edit - P_M·edit|| / ||edit||      (0 = fully on-manifold, 1 = fully off)

    This is the manifold-faithfulness diagnostic that separates onmanifold_steer
    (residual ~ 0) from naive/random/clamp (residual large). We delegate to the
    project's `src.utils.onmanifold_projection_residual` so the definition is
    shared with the toy scaffold; if a batch [n, d] is passed we average the
    per-row residual (the steering modules pass one representative edit vector).
    """
    import torch

    if not torch.is_tensor(edit):
        edit = torch.as_tensor(edit)
    if not torch.is_tensor(U_r):
        U_r = torch.as_tensor(U_r)
    e = edit.float()
    U = U_r.float()

    if e.dim() == 1:
        if _src_residual is not None:
            return float(_src_residual(e, U))
        # fallback (kept identical to src.utils): ||e - P_M e|| / ||e||
        nrm = e.norm()
        if nrm <= 1e-8:
            return 0.0
        proj = U @ (U.T @ e)
        return float((e - proj).norm() / nrm)

    # batch: residual per row, then mean (ignoring ~zero-norm rows).
    proj = (e @ U) @ U.T                               # [n, d]
    num = (e - proj).norm(dim=1)
    den = e.norm(dim=1)
    mask = den > 1e-8
    if mask.sum() == 0:
        return 0.0
    return float((num[mask] / den[mask]).mean())


# --------------------------------------------------------------------------- #
# Persistence: cache U_r to .npy (the paths.manifold_basis target).           #
# --------------------------------------------------------------------------- #
def save_basis(U_r, path: str) -> str:
    """Save U_r [d, r] to `path` as a float32 .npy (the cfg.paths.manifold_basis
    target). float32 keeps the projection numerically exact while staying small
    (1024 x 512 x 4 bytes ~ 2 MB)."""
    import numpy as np
    import torch

    arr = U_r.detach().cpu().float().numpy() if torch.is_tensor(U_r) else np.asarray(U_r, dtype=np.float32)
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(p), arr.astype(np.float32))
    return str(p)


def load_basis(path: str):
    """Load a cached U_r [d, r] back as a torch.FloatTensor."""
    import numpy as np
    import torch

    arr = np.load(str(path))
    return torch.from_numpy(arr).float()


# --------------------------------------------------------------------------- #
# Real-run driver: estimate U_r from a cached real-activation bank and cache.  #
# --------------------------------------------------------------------------- #
def estimate_and_cache(cfg: dict, cache_dir: str | None = None):
    """REAL PATH: pull a bank of real CLIP activations (data_real.load_activation_bank),
    estimate U_r at rank cfg.steering.proj_rank_r, and cache it to
    cfg.paths.manifold_basis. Imported here (not at module top) so the file still
    imports on a machine without data_real.py / open_clip.

    Returns (U_r, out_path).
    """
    import torch

    steering = cfg.get("steering", {})
    paths = cfg.get("paths", {})
    r = int(steering.get("proj_rank_r", 512))
    n_tokens = int(steering.get("bank_tokens", 2_000_000))
    cache_dir = cache_dir or paths.get("cache_dir", "./cache")
    out_path = paths.get("manifold_basis", "./outputs/U_r.npy")

    # data_real provides the cached real-activation bank loader. Guarded import:
    # absent on the build machine (no open_clip / no GPU), present on the GPU box.
    from data_real import load_activation_bank  # type: ignore

    # Bank = a uniform sample of real ImageNet-train patch activations (the
    # distribution the frozen model was trained on => the manifold we project to).
    bank = load_activation_bank(cache_dir, "imagenet_train", n_tokens, seed=0)
    if not torch.is_tensor(bank):
        bank = torch.as_tensor(bank)
    U_r = estimate_manifold_basis(bank, r)
    save_basis(U_r, out_path)
    return U_r, out_path


# --------------------------------------------------------------------------- #
# Smoke: fabricate a real-SHAPED bank with a planted low-rank sheet + off-sheet #
# wobble, recover it, verify the projection identities. No open_clip / no GPU.  #
# --------------------------------------------------------------------------- #
def _smoke() -> int:
    import numpy as np
    import torch

    torch.manual_seed(0)
    np.random.seed(0)

    # Real-SHAPED activation bank: d matches CLIP ViT-L/14 width; a realistic
    # token count for a smoke run. We PLANT a true 16-D sheet then add small
    # off-sheet Gaussian wobble, so a correct SVD should recover the sheet.
    d, n, true_r = 1024, 4096, 16
    Q, _ = torch.linalg.qr(torch.randn(d, true_r))     # orthonormal true sheet [d, true_r]
    coords = torch.randn(n, true_r) * torch.linspace(4.0, 1.0, true_r)  # decaying spread
    bank = coords @ Q.T                                 # points ON the sheet [n, d]
    bank = bank + 0.02 * torch.randn(n, d)              # small off-sheet wobble
    bank = bank + 3.0 * torch.randn(d)                  # a bulk mean offset (centered away)
    bank = bank.to(torch.float16)                       # match the on-disk float16 cache

    r = 32
    U_r = estimate_manifold_basis(bank, r)
    assert U_r.shape == (d, r), f"U_r shape {tuple(U_r.shape)} != ({d}, {r})"
    print(f"[smoke] estimated U_r shape = {tuple(U_r.shape)}  (d={d}, r={r})")

    # Orthonormality: U_r^T U_r should be I_r.
    gram = U_r.T @ U_r
    ortho_err = float((gram - torch.eye(r)).abs().max())
    print(f"[smoke] orthonormality  max|U_r^T U_r - I| = {ortho_err:.2e}  (~0 expected)")
    assert ortho_err < 1e-3

    # Projection idempotence on a representative vector: P_M(P_M v) == P_M v.
    v = torch.randn(d)
    pv = project_onmanifold(v, U_r)
    ppv = project_onmanifold(pv, U_r)
    idem_err = float((ppv - pv).norm())
    print(f"[smoke] idempotence  ||P_M(P_M v) - P_M v|| = {idem_err:.2e}  (~0 expected)")
    assert idem_err < 1e-3

    # The planted sheet should be (almost) fully captured by the top-r subspace:
    # projecting a true sheet direction onto U_r keeps ~all its length.
    captured = []
    for j in range(true_r):
        b = Q[:, j]
        captured.append(float(project_onmanifold(b, U_r).norm() / (b.norm() + 1e-8)))
    cap = float(np.mean(captured))
    print(f"[smoke] planted-sheet capture = {cap*100:.1f}%  (r>=true_r => ~100%)")
    assert cap > 0.95

    # A sheet direction is on-manifold (residual ~0); a random direction is mostly
    # off (residual large) — exactly the diagnostic onmanifold_steer relies on.
    res_on = offmanifold_residual(Q[:, 0], U_r)
    res_rand = offmanifold_residual(torch.randn(d), U_r)
    print(f"[smoke] off-manifold residual  sheet-dir={res_on:.3f}  random-dir={res_rand:.3f}")
    assert res_on < 0.1 < res_rand

    # Batch projection path + save/load round-trip.
    batch = torch.randn(64, d)
    pb = project_onmanifold(batch, U_r)
    assert pb.shape == batch.shape
    res_batch = offmanifold_residual(batch, U_r)
    print(f"[smoke] batch residual (random rows) = {res_batch:.3f}")

    import tempfile
    tmp = pathlib.Path(tempfile.mkdtemp()) / "U_r.npy"
    save_basis(U_r, str(tmp))
    U_r2 = load_basis(str(tmp))
    rt_err = float((U_r - U_r2).abs().max())
    print(f"[smoke] save/load round-trip max err = {rt_err:.2e}")
    assert rt_err < 1e-6

    # Full-rank sanity: r == d => P_M == I => projection is a no-op (naive case).
    U_full = estimate_manifold_basis(bank, d, center=False)
    res_full = offmanifold_residual(torch.randn(d), U_full)
    print(f"[smoke] full-rank (r=d) residual of random dir = {res_full:.2e}  "
          f"(~0 => P_M=I, on-manifold degenerates to naive)")
    assert res_full < 1e-3

    print("[smoke] manifold.py PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate/cache the on-manifold basis U_r.")
    ap.add_argument("--config", default=None, help="path to a real_run YAML config")
    ap.add_argument("--cache_dir", default=None, help="activation cache dir (overrides cfg)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU path on real-SHAPED synthetic tensors (no open_clip/GPU)")
    args = ap.parse_args()

    if args.smoke or args.config is None:
        return _smoke()

    # REAL PATH: load the config and estimate U_r from the cached real bank.
    from src.utils import load_config
    cfg = load_config(args.config)
    U_r, out = estimate_and_cache(cfg, cache_dir=args.cache_dir)
    print(f"[manifold] cached U_r {tuple(U_r.shape)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
