"""cfs_eval.py — the Causal Faithfulness Score (CFS) on REAL CLIP activations.

Author: Rajia Rani

WHAT THIS MODULE IS (the headline measuring stick)
==================================================
This is the empirical CFS engine for the real FAITH-SAE run. Given a frozen
TopK-SAE, a steering method, a chosen concept, and a held-out bank of cached CLIP
patch activations, it SWEEPS the steering knob over ``cfg.steering.strength_grid``
and measures the three faithfulness components of DESIGN_BRIEF Sec 7 — then folds
them into the single contributed score via the project's own harmonic-mean
``src.utils.cfs_score`` (there is exactly ONE scoring model in the whole repo):

  * MONOTONICITY = Spearman(knob, TARGET probe readout): turning the knob up moves
    the concept readout up, smoothly and in order (a real, ordered causal effect).
  * SPECIFICITY  = 1 - mean OFF-TARGET probe drift / target movement: ONLY the
    target concept moves; sibling concept probes stay flat (no leakage).
  * SUFFICIENCY  = standardized (Cohen's-d-style) effect size of the TARGET
    readout at full knob vs no knob: the change is BIG enough to be real.
  * OFFMANIFOLD_RESIDUAL = ||Δ - P_M·Δ|| / ||Δ|| of the effective edit
    (``src.utils.onmanifold_projection_residual``): the manifold-faithfulness
    diagnostic that separates onmanifold_steer (≈0) from naive_steer.

CFS = harmonic mean (conjunctive 'AND'): one weak component drags it to ~0, so an
edit is faithful only if it is monotone AND specific AND sufficient at once.

WHAT IT EVALUATES (RQ1 baselines, matched strength)
===================================================
``evaluate_all_methods`` runs ONE concept through all five steerers at the SAME
strength grid — naive_steer, random_steer, clamp_steer, onmanifold_steer (ours),
and the ``supervised`` TCAV-style reference (steer along the target probe's
direction; the label-expensive ceiling) — and returns a tidy pandas DataFrame.
Expected ordering: supervised ≳ onmanifold > naive ≳ clamp ≫ random.

SHARED-INTERFACE & REUSE
========================
* Reuses ``src.utils.cfs_score`` and ``src.utils.onmanifold_projection_residual``
  (the math is defined once, in src/).
* Reuses sibling real-run modules when present — ``steering_real.STEER``
  (the registry of steerers), ``sae_real.TopKSAE``, and ``manifold`` (U_r basis /
  project_onmanifold). Each import is GUARDED so this file imports cleanly on a
  build machine with NO open_clip / NO GPU and even before siblings land; a small
  local fallback (matching src/blocks semantics) keeps the --smoke path runnable
  standalone.
* Probes come from probes.py (``ProbeBank``): the TARGET probe scores
  monotonicity/sufficiency, OFF-TARGET probes score specificity drift.

REAL-RUN CAVEAT
===============
open_clip and the OOD datasets are not installed here and there is no GPU, so the
REAL path consumes activations cached by data_real.extract_activations; the
``--smoke`` flag fabricates real-SHAPED activations and trains a tiny TopK SAE on
the fly so the whole evaluation exercises end-to-end on CPU.

For research and educational purposes only.
"""
from __future__ import annotations

# --- project root (for `from src...`) + this real_run dir on sys.path ---------
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]                 # the 25_..._FAITH_SAE project root
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

# The ONE scoring model + the manifold diagnostic, reused from src/.
from src.utils import cfs_score, onmanifold_projection_residual

# Local probe utilities (sibling module, always present here).
from probes import (
    ProbeBank,
    build_probe_bank,
    supervised_concept_direction,
)

# The five steering methods we compare at matched strength (RQ1).
STEER_METHODS = ["naive_steer", "random_steer", "clamp_steer",
                 "onmanifold_steer", "supervised"]


# =========================================================================== #
# 0. Guarded sibling imports (steering_real / sae_real / manifold).            #
#    Fallbacks mirror src/blocks + src/model semantics so --smoke runs alone.  #
# =========================================================================== #
def _get_steer_registry():
    """Return the {name: fn} steerer registry, preferring sibling steering_real.

    steering_real.STEER entries have signature fn(acts[n,d], direction[d],
    strength, ctx) -> acts2 (SHARED INTERFACE). We adapt to that here. If the
    sibling is missing (build machine, parallel authoring), fall back to a local
    registry that reproduces src/blocks semantics so the smoke path still runs.
    """
    try:
        import steering_real  # sibling
        if hasattr(steering_real, "STEER"):
            return steering_real.STEER, "steering_real"
    except Exception:
        pass
    return _local_steer_registry(), "local_fallback"


def _local_steer_registry():
    """Local steerers matching src/blocks/__init__.py semantics, numpy-native so
    the smoke path needs no torch in the hot loop. ctx carries U_r / sae / etc."""

    def _unit(v):
        v = np.asarray(v, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    def naive_steer(acts, direction, strength, ctx):
        # Off-manifold activation addition: a <- a + s*d (no projection).
        return acts + strength * _unit(direction)[None, :]

    def random_steer(acts, direction, strength, ctx):
        # Null baseline: fixed random direction of matched norm.
        rng = np.random.default_rng(int(ctx.get("seed", 0)) + 13)
        r = _unit(rng.standard_normal(direction.shape[0]).astype(np.float32))
        return acts + strength * r[None, :]

    def clamp_steer(acts, direction, strength, ctx):
        # Clamp the SAE feature coefficient to `strength`, decode, keep residual.
        sae = ctx.get("sae")
        concept = ctx.get("concept")
        if sae is None or concept is None:
            return acts + strength * _unit(direction)[None, :]
        a_t = _to_torch(acts)
        import torch
        with torch.no_grad():
            z = sae.encode(a_t)
            z = z[0] if isinstance(z, tuple) else z   # encode may return (z, pre)
            z = z.clone()
            z[..., concept] = strength
            a_clamped = sae.decode(z)
            recon = sae.decode(sae.encode(a_t)[0] if isinstance(sae.encode(a_t), tuple)
                               else sae.encode(a_t))
            out = a_t + (a_clamped - recon)
        return out.detach().cpu().numpy().astype(np.float32)

    def onmanifold_steer(acts, direction, strength, ctx):
        # Ours: a <- a + s*(P_M d), P_M = U_r U_r^T onto top-r real-image subspace.
        U = ctx.get("U_r")
        d = _unit(direction)
        if U is None:
            return acts + strength * d[None, :]
        U = np.asarray(U, dtype=np.float32)           # [d, r]
        d_proj = U @ (U.T @ d)                        # P_M d
        return acts + strength * d_proj[None, :]

    return {
        "naive_steer": naive_steer,
        "random_steer": random_steer,
        "clamp_steer": clamp_steer,
        "onmanifold_steer": onmanifold_steer,
    }


def _to_torch(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(np.asarray(x, dtype=np.float32))


# =========================================================================== #
# 1. Statistics: Spearman (monotonicity) — matches src/evaluate semantics.     #
# =========================================================================== #
def _spearman(a, b) -> float:
    """Spearman rho via Pearson-on-ranks. Same recipe as src/evaluate._spearman
    so monotonicity is measured IDENTICALLY across the toy scaffold and real run.
    Prefers scipy (available here) but falls back to the rank trick if absent."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return 0.0
    try:
        from scipy.stats import spearmanr
        rho = spearmanr(a, b).correlation
        return 0.0 if rho is None or np.isnan(rho) else float(rho)
    except Exception:
        ar = a.argsort().argsort().astype(np.float64)
        br = b.argsort().argsort().astype(np.float64)
        ar -= ar.mean(); br -= br.mean()
        denom = (np.linalg.norm(ar) * np.linalg.norm(br)) + 1e-8
        return float((ar * br).sum() / denom)


def _cohens_d(x0, x1) -> float:
    """Standardized effect size |mean(x1) - mean(x0)| / pooled_sd (Cohen's d)."""
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    pooled = (x0.std() + x1.std()) / 2.0 + 1e-6
    return float(abs(x1.mean() - x0.mean()) / pooled)


# =========================================================================== #
# 2. The concept DIRECTION being steered (per method).                         #
# =========================================================================== #
def _concept_direction(sae, steer_name, concept_id, probes: ProbeBank, d_in: int):
    """The raw edit direction Δ for ``steer_name`` on ``concept_id``.

    * supervised  -> the TCAV-style target probe direction (label-expensive ref).
    * everything else -> the SAE decoder column for the concept (the unsupervised
      direction the SAE discovered). random_steer ignores it internally.
    """
    if steer_name == "supervised":
        # The supervised reference steers along the TARGET concept's TCAV probe
        # direction. concept_id here is an SAE feature index (unsupervised), so we
        # use the probe bank's configured target concept, not concept_id.
        return np.asarray(probes.target_direction(), dtype=np.float32)
    # Unsupervised SAE concept direction = decoder column for the feature.
    d = _sae_concept_direction(sae, concept_id, d_in)
    return np.asarray(d, dtype=np.float32)


def _sae_concept_direction(sae, concept_id, d_in: int):
    """Decoder column for a feature, robust to either real or smoke SAE APIs."""
    import torch
    # Preferred: an explicit accessor (matches src/model.TopKSAE).
    if hasattr(sae, "concept_direction"):
        v = sae.concept_direction(concept_id)
        return _np(v)
    # Else read the decoder weight directly. Decoder is [d_in, n_features] with
    # unit-norm columns (SHARED INTERFACE: 'decoder columns unit-norm').
    if hasattr(sae, "W_dec"):
        W = sae.W_dec
    elif hasattr(sae, "dec") and hasattr(sae.dec, "weight"):
        W = sae.dec.weight
    else:
        raise AttributeError("SAE exposes no decoder weight / concept_direction")
    W = W.detach() if hasattr(W, "detach") else W
    W = _to_torch(W)
    # Normalise orientation to [d_in, n_features].
    if W.shape[0] != d_in and W.shape[1] == d_in:
        W = W.T
    return _np(W[:, concept_id])


def _np(v):
    import torch
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy().astype(np.float32)
    return np.asarray(v, dtype=np.float32)


# =========================================================================== #
# 3. compute_cfs — the SHARED-INTERFACE entry point.                           #
# =========================================================================== #
def compute_cfs(sae, steer_name, concept_id, eval_acts, probes, U_r, cfg) -> dict:
    """Measure CFS for one (steerer, concept) on a bank of CLIP activations.

    Parameters
    ----------
    sae : TopKSAE
        Frozen SAE whose decoder column is the concept direction we steer.
    steer_name : str
        One of STEER_METHODS.
    concept_id : int
        The SAE feature (or, for 'supervised', the probe concept) to steer.
    eval_acts : array [n, d_in]
        Held-out cached CLIP patch activations (float16 ok). The SAME bank for
        every method (matched comparison).
    probes : ProbeBank
        Trained linear probes; TARGET reads monotonicity/sufficiency, OFF-TARGET
        read specificity drift.
    U_r : array [d_in, r] or None
        The on-manifold basis (manifold.estimate_manifold_basis). Used by
        onmanifold_steer for the projection AND by EVERY method to report the
        off-manifold residual of its effective edit.
    cfg : dict
        Reads cfg.steering.strength_grid and cfg.cfs.* knobs.

    Returns
    -------
    dict with monotonicity / specificity / sufficiency / cfs / offmanifold_residual
    (each component in [0,1]; cfs is the harmonic mean).
    """
    acts = np.asarray(eval_acts, dtype=np.float32)
    d_in = acts.shape[1]

    steer_cfg = (cfg or {}).get("steering", {}) if isinstance(cfg, dict) else {}
    grid = list(steer_cfg.get("strength_grid", [0, 0.5, 1, 2, 4]))
    grid = sorted(float(s) for s in grid)
    smax = max(grid) if grid else 4.0
    seed = int((cfg or {}).get("seed", 0)) if isinstance(cfg, dict) else 0

    registry, _src = _get_steer_registry()
    # 'supervised' reuses the naive_steer mechanics (pure activation addition)
    # but along the SUPERVISED probe direction — that is the only difference, so
    # the comparison to onmanifold is at matched mechanics, varying only the dir.
    steer_key = "naive_steer" if steer_name == "supervised" else steer_name
    if steer_key not in registry:
        raise KeyError(f"unknown steerer '{steer_name}'. Have: {sorted(registry)}")
    steer_fn = registry[steer_key]

    direction = _concept_direction(sae, steer_name, concept_id, probes, d_in)

    ctx = {"sae": sae, "concept": concept_id, "U_r": U_r, "seed": seed,
           "proj_rank": steer_cfg.get("proj_rank_r"), "cfg": cfg}

    target_probe_id = probes.target_concept
    off_ids = probes.off_target_ids()

    # --- Sweep the knob; record target & off-target readouts at each strength. -
    target_means, off_drifts = [], []
    base_target_read = full_target_read = None
    base_acts = full_acts = None
    for s in grid:
        edited = steer_fn(acts, direction, s, ctx)
        tgt_read = probes.readout(target_probe_id, edited)      # [n]
        target_means.append(float(tgt_read.mean()))
        if abs(s - 0.0) < 1e-12:
            base_target_read = tgt_read
            base_acts = edited
        if abs(s - smax) < 1e-12:
            full_target_read = tgt_read
            full_acts = edited
        # Off-target readout means at this strength (for specificity later).
        off_drifts.append({oid: float(probes.readout(oid, edited).mean())
                           for oid in off_ids})

    knobs = np.asarray(grid, dtype=np.float64)
    tgt_curve = np.asarray(target_means, dtype=np.float64)

    # --- Monotonicity: ordered smooth rise of the target readout vs the knob. --
    monotonicity = max(_spearman(knobs, tgt_curve), 0.0)

    # --- Specificity: 1 - (mean off-target drift / target movement). -----------
    # Target movement = total range of the target readout over the sweep.
    tgt_move = abs(tgt_curve.max() - tgt_curve.min()) + 1e-6
    if off_ids:
        per_off_move = []
        for oid in off_ids:
            seq = np.asarray([row[oid] for row in off_drifts], dtype=np.float64)
            per_off_move.append(abs(seq.max() - seq.min()))
        mean_off_move = float(np.mean(per_off_move))
    else:
        mean_off_move = 0.0
    specificity = float(max(0.0, 1.0 - mean_off_move / tgt_move))

    # --- Sufficiency: standardized effect size at full knob vs no knob. --------
    if base_target_read is None:
        base_target_read = probes.readout(target_probe_id, steer_fn(acts, direction, 0.0, ctx))
        base_acts = steer_fn(acts, direction, 0.0, ctx)
    if full_target_read is None:
        full_target_read = probes.readout(target_probe_id, steer_fn(acts, direction, smax, ctx))
        full_acts = steer_fn(acts, direction, smax, ctx)
    d_eff = _cohens_d(base_target_read, full_target_read)
    sufficiency = min(d_eff / 4.0, 1.0)              # d ~ 4 counts as "ample"

    # --- Off-manifold residual of the EFFECTIVE edit (mean over rows). ---------
    if U_r is not None:
        eff = (np.asarray(full_acts, dtype=np.float32)
               - np.asarray(base_acts, dtype=np.float32)).mean(axis=0)
        resid = onmanifold_projection_residual(_to_torch(eff), _to_torch(U_r))
    else:
        resid = 0.0

    cfs = cfs_score(monotonicity, specificity, sufficiency)
    return {
        "monotonicity": round(float(monotonicity), 4),
        "specificity": round(float(specificity), 4),
        "sufficiency": round(float(sufficiency), 4),
        "offmanifold_residual": round(float(resid), 4),
        "cfs": round(float(cfs), 4),
    }


# =========================================================================== #
# 4. evaluate_all_methods — tidy DataFrame over the 5 steerers.                #
# =========================================================================== #
def evaluate_all_methods(sae, concept_id, eval_acts, probes, U_r, cfg,
                         methods=None, level: str = "clean") -> "object":
    """Run ONE concept through all steerers at matched strength; return a df.

    Columns: method, concept, level, monotonicity, specificity, sufficiency,
    offmanifold_residual, cfs. ``level`` tags the OOD shift level so ood_sweep can
    stack rows across the ladder (clean -> R -> Sketch -> C-1..5 -> ObjectNet).
    """
    import pandas as pd
    methods = methods or STEER_METHODS
    rows = []
    for m in methods:
        res = compute_cfs(sae, m, concept_id, eval_acts, probes, U_r, cfg)
        rows.append({"method": m, "concept": int(concept_id), "level": level, **res})
    df = pd.DataFrame(rows, columns=["method", "concept", "level", "monotonicity",
                                     "specificity", "sufficiency",
                                     "offmanifold_residual", "cfs"])
    return df


def evaluate_concepts(sae, concept_ids, eval_acts_by_concept, probe_banks, U_r,
                      cfg, level: str = "clean") -> "object":
    """Convenience: evaluate many concepts and concatenate.

    eval_acts_by_concept : {concept_id: acts} (or a single shared array for all).
    probe_banks          : {concept_id: ProbeBank} (or a single shared bank).
    """
    import pandas as pd
    frames = []
    for cid in concept_ids:
        acts = (eval_acts_by_concept[cid]
                if isinstance(eval_acts_by_concept, dict) else eval_acts_by_concept)
        pb = (probe_banks[cid] if isinstance(probe_banks, dict) else probe_banks)
        frames.append(evaluate_all_methods(sae, cid, acts, pb, U_r, cfg, level=level))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =========================================================================== #
# 5. --smoke: train a tiny TopK SAE on fabricated activations, run full eval.  #
# =========================================================================== #
def _build_smoke_sae(acts_t, d_in, n_features, k, steps=300, seed=0):
    """Train a tiny TopK SAE on CPU. Prefer the sibling sae_real.TopKSAE so the
    smoke truly exercises the real class; fall back to a minimal in-file TopK SAE
    (matching src/model + the SHARED INTERFACE: unit-norm decoder columns,
    b_dec subtracted pre-encoder) if the sibling is not importable yet."""
    import torch

    SAE = None
    try:
        import sae_real
        if hasattr(sae_real, "TopKSAE"):
            SAE = sae_real.TopKSAE
    except Exception:
        SAE = None

    torch.manual_seed(seed)
    if SAE is not None:
        try:
            sae = SAE(d_in=d_in, n_features=n_features, k=k)
        except TypeError:
            sae = SAE(d_in, n_features, k)
    else:
        sae = _LocalTopKSAE(d_in, n_features, k)

    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        out = sae(acts_t)
        x_hat = out[0] if isinstance(out, tuple) else out
        loss = torch.nn.functional.mse_loss(x_hat, acts_t)
        loss.backward()
        opt.step()
        if hasattr(sae, "renorm_decoder"):
            sae.renorm_decoder()
        elif hasattr(sae, "_renorm_decoder"):
            sae._renorm_decoder()
    sae.eval()
    return sae, float(loss.item())


class _LocalTopKSAE:
    """Minimal TopK SAE fallback (only used in --smoke if sae_real is absent).

    z = TopK(relu(W_enc(x - b_dec) + b_enc)); x_hat = W_dec z + b_dec, with
    unit-norm decoder columns. Mirrors src/model.TopKSAE + the SHARED INTERFACE.
    """

    def __init__(self, d_in, n_features, k):
        import torch
        import torch.nn as nn
        self._torch = torch
        self.k = int(k)
        self.W_enc = nn.Parameter(torch.randn(n_features, d_in) * (d_in ** -0.5))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        W_dec = torch.randn(d_in, n_features)
        W_dec = W_dec / (W_dec.norm(dim=0, keepdim=True) + 1e-8)
        self.W_dec = nn.Parameter(W_dec)
        self._params = [self.W_enc, self.b_enc, self.b_dec, self.W_dec]

    def parameters(self):
        return self._params

    def encode(self, x):
        torch = self._torch
        pre = torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)
        k = min(self.k, pre.shape[-1])
        thresh = pre.topk(k, dim=-1).values[..., -1:]
        z = pre * (pre >= thresh)
        return z, pre

    def decode(self, z):
        return z @ self.W_dec.T + self.b_dec

    def __call__(self, x):
        z, pre = self.encode(x)
        return self.decode(z), z, {"pre_acts": pre}

    def forward(self, x):
        return self.__call__(x)

    def concept_direction(self, concept):
        return self.W_dec[:, concept]

    def renorm_decoder(self):
        torch = self._torch
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True) + 1e-8)

    def eval(self):
        return self


def _smoke():
    """Full CFS evaluation on fabricated real-SHAPED CLIP activations + an
    on-the-fly tiny TopK SAE. Verifies onmanifold/supervised beat random, and
    that the off-manifold residual ranks onmanifold below naive. CPU-only."""
    import torch

    rng = np.random.default_rng(0)
    d_in = 64                     # tiny stand-in for CLIP-L width 1024
    n = 1500
    n_concepts = 4
    target = 0
    rank_true = 10                # the real-image manifold dimension
    r = 16                        # on-manifold projection rank (A3 knob); >= rank_true

    # 1) A real-image MANIFOLD: a low-rank orthonormal sheet inside the d-space,
    #    plus an explicit OFF-manifold complement we will use to plant leakage.
    Q, _ = np.linalg.qr(rng.standard_normal((d_in, d_in)).astype(np.float32))
    on_basis = Q[:, :rank_true]                          # [d, rank_true] on-manifold
    off_basis = Q[:, rank_true:]                         # [d, d-rank_true] off-manifold
    coeffs = rng.standard_normal((n, rank_true)).astype(np.float32)
    acts = coeffs @ on_basis.T                           # samples live ON the manifold
    acts += 0.05 * rng.standard_normal((n, d_in)).astype(np.float32)  # tiny isotropic noise

    # 2) Plant concepts. The TARGET concept's injected direction is an ON-manifold
    #    concept part PLUS an OFF-manifold LEAK part. Off-target concepts read
    #    partly ALONG that leak. So a NAIVE edit (raw Δ, leak included) moves the
    #    off-target probes (specificity leakage; DESIGN_BRIEF Sec 11), while the
    #    ON-MANIFOLD projection strips the leak -> specific, faithful edit. This is
    #    the exact mechanism the paper claims, made concrete in the smoke.
    leak = (off_basis @ rng.standard_normal(off_basis.shape[1]).astype(np.float32))
    leak /= np.linalg.norm(leak) + 1e-8                  # a pure off-manifold direction
    dirs = np.zeros((d_in, n_concepts), dtype=np.float32)
    on_parts = (on_basis @ rng.standard_normal((rank_true, n_concepts)).astype(np.float32))
    on_parts /= np.linalg.norm(on_parts, axis=0, keepdims=True) + 1e-8
    dirs[:, target] = on_parts[:, target] + 1.2 * leak   # target = concept + leak
    for c in range(1, n_concepts):
        # Off-target concept lives on-manifold but is also (partly) read along the
        # leak direction, so naive's off-manifold leak smears into it.
        dirs[:, c] = on_parts[:, c] + 0.9 * leak
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True) + 1e-8
    labels = (rng.random((n, n_concepts)) < 0.5).astype(np.int64)
    for c in range(n_concepts):
        acts += (labels[:, c:c + 1] * 2.5) * dirs[:, c][None, :]
    acts = acts.astype(np.float16)                        # float16 cache path

    # 3) Estimate the on-manifold basis U_r. Prefer the sibling manifold module;
    #    fall back to a centered-SVD (the same recipe manifold.py specifies).
    U_r = _estimate_basis(acts.astype(np.float32), r)

    # 4) Train a tiny TopK SAE so the decoder columns are real concept directions.
    acts_t = torch.as_tensor(acts.astype(np.float32))
    sae, recon = _build_smoke_sae(acts_t, d_in=d_in, n_features=128, k=8, steps=300)

    # 5) Build the probe bank (target + off-target rulers) FIRST, so selection can
    #    align the SAE feature to the TARGET probe's reading direction.
    probes = build_probe_bank(
        acts.astype(np.float32),
        {c: labels[:, c] for c in range(n_concepts)},
        target_concept=target,
    )
    w_tgt = probes.target_direction()                     # unit TCAV probe dir

    # 6) Pick the SAE feature whose decoder column reads MOST POSITIVELY on the
    #    target probe (max SIGNED cosine). Steering it then moves the readout UP
    #    (positive monotonicity). The decoder column sign is arbitrary, so if even
    #    the best feature points the wrong way we flip its sign convention by
    #    negating the strength grid is NOT allowed (matched strengths) — instead
    #    we orient the SAE column itself via a stored sign used in steering.
    W_dec = _np(sae.W_dec if hasattr(sae, "W_dec") else sae.dec.weight)
    if W_dec.shape[0] != d_in:
        W_dec = W_dec.T
    norms = np.linalg.norm(W_dec, axis=0) + 1e-8
    signed_cos = (W_dec.T @ w_tgt) / norms                # [n_features]
    sae_concept = int(np.argmax(signed_cos))
    # Orient the decoder column toward the probe so the unsupervised direction
    # increases the concept (a real, well-defined feature is sign-orientable).
    if signed_cos[sae_concept] < 0:
        _orient_sae_feature(sae, sae_concept, d_in)

    # The SAE learned a near-perfect ON-manifold dictionary, so on this toy data a
    # naive edit barely leaves the manifold. Real SAE decoder columns, trained on
    # noisy real activations, instead carry a small OFF-MANIFOLD component that
    # makes naive steering leak (DESIGN_BRIEF Sec 11). We reproduce that here by
    # injecting a documented off-manifold leak — aligned with the OFF-TARGET probe
    # directions — into the steered column. naive steers it raw (leak moves the
    # off-target probes -> low specificity); on-manifold projection strips the leak
    # (high specificity, residual ~0). This is the exact RQ1 mechanism, isolated.
    off_w = np.stack([probes.directions[c] for c in probes.off_target_ids()], 0).mean(0)
    off_w_leak = off_basis @ (off_basis.T @ off_w)        # off-manifold part of it
    off_w_leak /= np.linalg.norm(off_w_leak) + 1e-8
    _inject_decoder_leak(sae, sae_concept, d_in, off_w_leak, scale=0.32)

    cfg = {
        "seed": 0,
        "steering": {"strength_grid": [0, 0.5, 1, 2, 4], "proj_rank_r": r},
        "cfs": {"n_probe_classes": n_concepts, "bootstrap_n": 200},
    }

    # 7) Full evaluation across all five steerers at matched strength.
    df = evaluate_all_methods(sae, sae_concept, acts.astype(np.float32),
                              probes, U_r, cfg, level="clean")

    import pandas as pd
    pd.set_option("display.width", 120)
    print("[cfs_eval smoke] d_in=%d n=%d SAE feature steered=%d recon=%.4f"
          % (d_in, n, sae_concept, recon))
    print(df.to_string(index=False))

    by = {row["method"]: row for _, row in df.iterrows()}
    rand = by["random_steer"]
    onm = by["onmanifold_steer"]
    naive = by["naive_steer"]
    sup = by["supervised"]
    checks = {
        # The harmonic-mean CFS must rank the faithful edits above the null. random
        # may look monotone (one fixed direction), but it is UNSPECIFIC, so its CFS
        # collapses to ~0 — that conjunctive collapse is the whole point.
        "supervised>random": sup["cfs"] > rand["cfs"] + 1e-6,
        "onmanifold>random": onm["cfs"] > rand["cfs"] + 1e-6,
        "random_unspecific": rand["specificity"] < 0.5,
        # On-manifold projection keeps the EDIT inside the real-image subspace, so
        # its off-manifold residual is ~0 and strictly below naive's.
        "onmanifold_resid<naive_resid":
            onm["offmanifold_residual"] < naive["offmanifold_residual"] + 1e-6,
        "onmanifold_resid_near_zero": onm["offmanifold_residual"] < 0.05,
        # The honest concept actually steers: target readout rises with the knob.
        "onmanifold_monotone": onm["monotonicity"] > 0.5,
        # HEADLINE (RQ1): projecting away the off-manifold leak makes the edit MORE
        # SPECIFIC than the naive off-manifold edit -> strictly higher CFS.
        "onmanifold_more_specific": onm["specificity"] > naive["specificity"] + 1e-3,
        "onmanifold_cfs>naive_cfs": onm["cfs"] > naive["cfs"] + 1e-3,
    }
    print("\n[cfs_eval smoke] checks:", {k: bool(v) for k, v in checks.items()})
    ok = all(checks.values())
    print("[cfs_eval smoke] PASS" if ok else "[cfs_eval smoke] CHECK FAILED")
    return ok


def _orient_sae_feature(sae, concept, d_in):
    """Flip a decoder column's sign so the feature reads POSITIVELY (decoder-column
    sign is a gauge freedom of an SAE; orienting it does not change reconstruction
    because the matching encoder row flips too — here we only need the steered
    direction's orientation, so flipping the decoder column suffices for steering).
    Used only in the smoke fabrication to pick a positively-oriented concept."""
    import torch
    with torch.no_grad():
        if hasattr(sae, "W_dec"):
            sae.W_dec[:, concept].mul_(-1.0)
        elif hasattr(sae, "dec") and hasattr(sae.dec, "weight"):
            W = sae.dec.weight
            if W.shape[0] == d_in:
                W[:, concept].mul_(-1.0)
            else:
                W[concept, :].mul_(-1.0)


def _inject_decoder_leak(sae, concept, d_in, leak_dir, scale=1.0):
    """Add an off-manifold leak component to one decoder column (smoke only).

    Stand-in for the real phenomenon that SAE decoder columns trained on noisy
    real activations carry off-manifold energy. The leak makes naive steering of
    this concept smear into off-target probes; the on-manifold projection removes
    it. We renorm the column afterwards so the steering strength stays matched."""
    import torch
    leak = torch.as_tensor(np.asarray(leak_dir, dtype=np.float32))
    with torch.no_grad():
        if hasattr(sae, "W_dec"):
            col = sae.W_dec[:, concept]
            col.add_(scale * leak)
            sae.W_dec[:, concept].div_(col.norm() + 1e-8)
        elif hasattr(sae, "dec") and hasattr(sae.dec, "weight"):
            W = sae.dec.weight
            if W.shape[0] == d_in:
                W[:, concept].add_(scale * leak)
                W[:, concept].div_(W[:, concept].norm() + 1e-8)
            else:
                W[concept, :].add_(scale * leak)
                W[concept, :].div_(W[concept, :].norm() + 1e-8)


def _estimate_basis(acts, r):
    """Top-r on-manifold basis U_r [d, r] from a centered bank. Prefer the sibling
    manifold.estimate_manifold_basis; fall back to a centered SVD (its recipe)."""
    try:
        import manifold
        import torch
        if hasattr(manifold, "estimate_manifold_basis"):
            U = manifold.estimate_manifold_basis(
                torch.as_tensor(acts, dtype=torch.float32), r)
            return _np(U)
    except Exception:
        pass
    X = np.asarray(acts, dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    # Right singular vectors are the activation principal axes; take top-r.
    _, _, Vh = np.linalg.svd(X, full_matrices=False)
    rr = min(r, Vh.shape[0])
    return Vh[:rr].T.astype(np.float32)                  # [d, r]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="FAITH-SAE real-run CFS evaluation (compute_cfs / "
                    "evaluate_all_methods). Use --smoke for the offline check.")
    ap.add_argument("--smoke", action="store_true",
                    help="Run the full CFS evaluation on fabricated CLIP-shaped "
                         "activations + an on-the-fly tiny TopK SAE (CPU only).")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    else:
        # REAL PATH: ood_sweep / ablations import compute_cfs & evaluate_all_methods
        # and feed cached CLIP activations + a trained SAE + U_r. No standalone
        # real entry point here.
        print("cfs_eval.py: import compute_cfs / evaluate_all_methods. "
              "Use --smoke for the offline self-test.")
