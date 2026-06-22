"""_common.py — shared setup every step script in milestone 7 imports.

WHY THIS FILE EXISTS
--------------------
Each ablation step (step1_a1_*.py ... step5_a5_*.py) needs to do the SAME boring
things before it can start teaching:
  1. Make Python able to `import` the project's real research code in `src/`.
  2. Load the knobs from `config.yaml`.
  3. Build a small LABELLED synthetic activation bank + train probe "rulers" + the
     on-manifold subspace U_r + an empirical CFS measurement (the SAME measuring
     rig every ablation reuses, so the ONLY thing that differs between ablation
     runs is the one knob being turned).
Rather than copy-paste that into every step, we write it ONCE here and every step
does `from _common import ...`. (Analogy: a kitchen's prep station — you set out
the same knives and cutting board once, not per dish.)

KEY IDEA: adding the project ROOT to `sys.path`
------------------------------------------------
`sys.path` is the list of folders Python searches when you write `import src`.
The project root (three levels up from this file:
  .../25_Rajia_Rani_FAITH_SAE/code/milestone_7_ablations/_common.py
   parents[0] = milestone_7_ablations
   parents[1] = code
   parents[2] = 25_..._FAITH_SAE   <-- the project ROOT, which contains src/)
is NOT on that list by default, so `import src` would fail. We insert it, then
`from src import ...` works. This is exactly how the milestone REUSES the real
research code (the STEER_REGISTRY, cfs_score, onmanifold_projection_residual)
instead of re-implementing it.

WHAT IS SHARED HERE FOR THE ABLATIONS
-------------------------------------
An "ablation" turns ONE knob and holds everything else fixed (see the README).
To measure how the Causal Faithfulness Score (CFS) responds to that one knob, we
need a fixed measuring rig: a labelled activation bank + one linear-probe "ruler"
per concept + the on-manifold subspace U_r + an empirical CFS measurement. We put
the rig-builders (`build_labelled_bank`, `train_probes`, `estimate_U_r`,
`train_sae_decoder`, `measure_cfs`) here so each ablation step calls the EXACT
same measurement and only its own knob changes. Nothing here is a lookup table:
every CFS number is COMPUTED from the data (Spearman, sklearn probes, Cohen's-d
effect size).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

# --- 1. Put the project root on sys.path so `from src import ...` works ------
ROOT = pathlib.Path(__file__).resolve().parents[2]   # the 25_..._FAITH_SAE folder
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = pathlib.Path(__file__).resolve().parent       # milestone_7_ablations/


def load_cfg(path: str = "config.yaml") -> dict:
    """Read config.yaml into a plain Python dict.

    We reuse the project's own loader (`src.utils.load_config`) so this milestone
    parses config EXACTLY the way the real pipeline does. A dict is just a set of
    name->value pairs, e.g. cfg["steer_strength"] -> 4.0.
    """
    from src.utils import load_config
    p = HERE / path
    return load_config(str(p))


def banner(title: str) -> None:
    """Print a clear section header so the console output is easy to follow."""
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def outpath(name: str) -> str:
    """Absolute path inside this milestone's outputs/ folder."""
    return str(HERE / "outputs" / name)


# =========================================================================== #
# THE SHARED MEASURING RIG — every ablation reuses these so the only thing     #
# that changes between runs is the single knob being ablated.                  #
# =========================================================================== #

def _random_orthonormal(dim: int, rank: int, seed: int) -> np.ndarray:
    """[dim, rank] matrix with orthonormal columns (a basis for a `rank`-D sheet).

    Orthonormal = each column has length 1 and any two are perpendicular. We draw
    a random matrix and orthonormalize it with QR (the standard "make these arrows
    perpendicular and length-1" tool).
    """
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((dim, rank))
    Q, _ = np.linalg.qr(M)
    return Q[:, :rank].astype(np.float32)


def _on_sheet_concept_dirs(B_sheet: np.ndarray, n_concepts: int,
                           seed: int) -> np.ndarray:
    """[n_concepts, dim]: unit concept directions that lie ON the sheet.

    We place each concept inside the column span of B_sheet (so the concept lives
    where real data lives) and orthonormalize the chosen within-sheet coordinates
    so the concepts do not mechanically overlap WITHIN the sheet. Any specificity
    leakage we measure later is then the STEERER pushing OFF the sheet, not a
    rigged in-sheet direction overlap.
    """
    dim, true_r = B_sheet.shape
    rng = np.random.default_rng(seed + 5)
    C = rng.standard_normal((true_r, n_concepts))     # coordinates within the sheet
    Qc, _ = np.linalg.qr(C)                            # perpendicular within-sheet
    dirs = (B_sheet @ Qc[:, :n_concepts]).T           # [n_concepts, dim], on-sheet
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
    return dirs.astype(np.float32)


def build_a3_bank(cfg: dict):
    """A bank purpose-built so the A3 projection-rank knob shows its real geometry.

    A3 studies WHAT the projection rank r does to a steer. To see both failure
    modes from a real computation, the target concept and the activations are
    constructed so that:
      * the TARGET concept's variation is spread ACROSS the whole sheet (not packed
        into PC1) -> a small r truncates it -> the effect DIES (over-constrained);
      * the activations also carry an OFF-sheet leak component that off-target
        probes can read -> a large r (P_M -> I) re-admits that leak -> off-target
        probes move -> specificity LEAKS (under-constrained, == naive).
    So CFS rises from a starved low-r value, PEAKS near the true sheet rank, then
    declines toward the naive level as r -> dim. Returns
    (acts, labels, dirs, edit_dir, U_ref) where edit_dir is the raw steering
    direction (concept part + off-sheet leak) and U_ref is a FIXED reference sheet
    (the true `true_manifold_rank` PCA basis) used only for the residual diagnostic.
    """
    dim = int(cfg["dim"])
    n_c = int(cfg["n_concepts"])
    N = int(cfg["bank_size"])
    true_r = int(cfg["true_manifold_rank"])
    strength = float(cfg["concept_strength"])
    noise = float(cfg["noise_off_manifold"])
    seed = int(cfg["seed"])

    rng = np.random.default_rng(seed + 23)
    B_sheet = _random_orthonormal(dim, true_r, seed=seed)        # on-sheet directions
    # ONE leftover OFF-sheet direction (perpendicular to the whole sheet).
    full = _random_orthonormal(dim, dim, seed=seed + 1)          # full orthonormal frame
    P_sheet = B_sheet @ B_sheet.T
    off_dir = full[:, -1] - P_sheet @ full[:, -1]
    off_dir = (off_dir / (np.linalg.norm(off_dir) + 1e-8)).astype(np.float32)

    # TARGET concept direction = an EQUAL mix of all sheet directions (spread across
    # the sheet) so low r truncates it; plus a small off-sheet leak.
    tgt_on = (B_sheet @ np.ones(true_r, dtype=np.float32))
    tgt_on = tgt_on / (np.linalg.norm(tgt_on) + 1e-8)
    leak = 0.6                                                   # off-sheet leak weight
    edit_dir = (tgt_on + leak * off_dir)
    edit_dir = (edit_dir / (np.linalg.norm(edit_dir) + 1e-8)).astype(np.float32)

    # OFF-target concepts: pure on-sheet directions (perpendicular within the sheet).
    Cc = rng.standard_normal((true_r, n_c - 1))
    Qc, _ = np.linalg.qr(Cc)
    off_concepts = (B_sheet @ Qc[:, :n_c - 1]).T
    off_concepts = off_concepts / (np.linalg.norm(off_concepts, axis=1, keepdims=True) + 1e-8)
    dirs = np.vstack([tgt_on[None, :], off_concepts]).astype(np.float32)  # [n_c, dim]

    labels = (rng.random((N, n_c)) < 0.5).astype(np.float32)
    on_coords = noise * rng.standard_normal((N, true_r)).astype(np.float32)
    acts = on_coords @ B_sheet.T
    acts += (labels @ dirs) * strength
    # an OFF-sheet signal that one off-target probe can latch onto, so re-admitting
    # the off-sheet direction at high r causes a MEASURABLE specificity leak.
    acts += (labels[:, 1:2] * strength) * off_dir[None, :]
    acts += 0.1 * noise * rng.standard_normal((N, dim)).astype(np.float32)

    U_ref = estimate_U_r(acts, true_r)                          # fixed reference sheet
    return (acts.astype(np.float32), labels, dirs.astype(np.float32),
            edit_dir, U_ref)


def build_labelled_bank(cfg: dict, seed_extra: int = 0, manifold_rank_override=None):
    """Build (acts [N, dim], labels [N, n_concepts], dirs [n_concepts, dim], B_sheet).

    The bank lives on a known low-rank SHEET: the background variation and ALL
    concept directions sit on a `true_manifold_rank`-D sheet inside the dim-D
    space, with only a tiny OFF-sheet wobble. That sheet is exactly what
    on-manifold steering projects onto; the off-sheet region is where a naive edit
    drifts and causes off-target probes to misread (specificity leakage). Concept
    0 is the steering TARGET (planted CLEANLY so it is reliably steerable); the
    rest are OFF-TARGET watchers planted at DECREASING cleanliness so they span a
    real range of interpretability — that range is what A4's threshold filters.

    `seed_extra` lets a step request an independent draw (e.g. A5 uses two banks
    for the two backbone-layer choices) without colliding with the default bank.
    `manifold_rank_override` lets A5's "early vs late layer" stand-in change the
    true sheet rank (an earlier layer = a thicker, less concept-aligned sheet).
    """
    dim = int(cfg["dim"])
    n_c = int(cfg["n_concepts"])
    N = int(cfg["bank_size"])
    true_r = int(manifold_rank_override if manifold_rank_override is not None
                 else cfg["true_manifold_rank"])
    strength = float(cfg["concept_strength"])
    noise = float(cfg["noise_off_manifold"])
    seed = int(cfg["seed"]) + int(seed_extra)

    B_sheet = _random_orthonormal(dim, true_r, seed=seed)        # [dim, true_r]
    dirs = _on_sheet_concept_dirs(B_sheet, n_c, seed)            # [n_concepts, dim]

    rng = np.random.default_rng(seed + 11)
    labels = (rng.random((N, n_c)) < 0.5).astype(np.float32)     # fair coin per concept

    # Per-concept CLEANLINESS via POLYSEMANTICITY (not loudness): every concept is
    # injected at the SAME strength (so effect sizes are comparable), but later
    # concepts are progressively CONTAMINATED — each is mixed with a shared nuisance
    # direction by a growing fraction. A clean concept (concept 0) has a crisp,
    # well-defined direction (accurate probe, faithful steer); a polysemantic
    # concept (high index) blurs into the nuisance direction (less accurate probe
    # AND less faithful steer at once). This is the realistic reason a feature is
    # "well-defined" or not, and it is what A4's threshold filters on.
    poly = np.linspace(0.0, 0.9, n_c).astype(np.float32)         # contamination per concept
    nuisance = _random_orthonormal(dim, 1, seed=seed + 31)[:, 0]  # shared junk dir (on/off sheet)
    eff_dirs = dirs.copy()
    for c in range(n_c):
        mixed = (1.0 - poly[c]) * dirs[c] + poly[c] * nuisance
        eff_dirs[c] = mixed / (np.linalg.norm(mixed) + 1e-8)
    on_coords = noise * rng.standard_normal((N, true_r)).astype(np.float32)
    acts = on_coords @ B_sheet.T                                # [N, dim], on-sheet
    acts += (labels * strength) @ eff_dirs                      # inject present concepts
    acts += 0.15 * noise * rng.standard_normal((N, dim)).astype(np.float32)  # off-sheet wobble
    # report the EFFECTIVE (possibly contaminated) directions so probes/steerers
    # see the same concept geometry the activations actually contain.
    dirs = eff_dirs
    return acts.astype(np.float32), labels, dirs, B_sheet


def train_probes(acts: np.ndarray, labels: np.ndarray, seed: int):
    """Train one LogisticRegression "ruler" per concept; return (W, b, accs).

    A probe is a tiny linear model that reads ONE concept off an activation. W
    [n_concepts, dim] = each row is the learned weight vector (the concept's
    ruler-direction). b [n_concepts] = each bias. accs = held-out accuracy per
    concept (how trustworthy each ruler is). sklearn's LogisticRegression does
    the fit; the held-out split keeps the accuracy honest (not memorized).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    n_c = labels.shape[1]
    dim = acts.shape[1]
    W = np.zeros((n_c, dim), dtype=np.float32)
    b = np.zeros((n_c,), dtype=np.float32)
    accs = []
    for c in range(n_c):
        y = labels[:, c].astype(int)
        Xtr, Xte, ytr, yte = train_test_split(
            acts, y, test_size=0.2, random_state=seed + c, stratify=y)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, ytr)
        W[c] = clf.coef_[0].astype(np.float32)
        b[c] = float(clf.intercept_[0])
        accs.append(float(clf.score(Xte, yte)))
    return W, b, accs


def estimate_U_r(acts: np.ndarray, r: int) -> np.ndarray:
    """PCA the bank; return U_r [dim, r] (top-r principal directions as columns).

    PCA finds the directions the activation cloud spreads out most; the top-r of
    them are our estimate of the on-manifold sheet. We transpose components_ so
    the directions are COLUMNS (the project's convention: U_r is [dim, r]).
    Milestone 4 teaches PCA / U_r / P_M from zero; here we just reuse the idea.
    """
    from sklearn.decomposition import PCA
    dim = acts.shape[1]
    r = max(1, min(int(r), dim))
    pca = PCA(n_components=r, svd_solver="full")
    pca.fit(acts)
    U_r = pca.components_[:r].T.copy()              # [dim, r]
    return U_r.astype(np.float32)


def train_sae_decoder(cfg_overrides: dict, acts: np.ndarray, base_cfg: dict):
    """Train the project's REAL SAE on the bank; return (decoder, recon_mse, model).

    We REUSE src.model.make_model (the FaithSAE harness) — we do NOT re-implement
    the SAE. `cfg_overrides` lets an ablation change ONE SAE knob (e.g. sae_type or
    topk_k) while every other setting stays at the shared baseline. Only the SAE
    trains; the backbone is frozen. The returned decoder columns are the SAE's
    discovered concept directions that clamp_steer / onmanifold_steer act on.

    --- THE A1 L1-SAE VARIANT (minimal, clearly commented) ----------------------
    src.model.TopKSAE already carries BOTH sparsity recipes (it switches on
    cfg["sae_type"]):
      * sae_type == "topk": after the ReLU encoder it KEEPS only the k biggest
        feature values per item and zeroes the rest (a hard top-k mask).
      * sae_type == "l1":   it SKIPS the top-k mask and instead ADDS an
        L1 penalty  l1_coeff * mean(|z|)  to the reconstruction loss, which
        PUSHES most feature values toward zero (soft sparsity) during training.
    So the A1 ablation is literally "set sae_type to 'topk' vs 'l1' and retrain".
    To make the L1 penalty's STRENGTH a real, tunable knob (the src default hard-
    codes 1e-3), we monkey-patch the SAE's forward to read cfg["l1_coeff"] when
    sae_type == "l1". This is the whole local L1 variant — a few lines, no new
    architecture — so the reader can see exactly what "L1 SAE" means here.
    """
    import torch
    import torch.nn.functional as F

    from src.model import make_model
    from src.utils import set_seed

    cfg = dict(base_cfg)
    cfg.update(cfg_overrides)
    cfg.setdefault("d_model", int(base_cfg["dim"]))
    set_seed(int(cfg["seed"]))
    model = make_model(cfg)

    # --- A1 local L1 variant: make the L1 penalty weight a real config knob -----
    if cfg.get("sae_type") == "l1":
        l1_coeff = float(cfg.get("l1_coeff", 1e-3))

        def l1_forward(a, _sae=model.sae, _coeff=l1_coeff):
            # encode WITHOUT a top-k mask (sae_type=='l1' already skips it), then
            # penalise the average feature magnitude so most features go to ~0.
            z = _sae.encode(a)
            a_hat = _sae.decode(z)
            loss = F.mse_loss(a_hat, a) + _coeff * z.abs().mean()
            return a_hat, z, loss

        model.sae.forward = l1_forward     # the entire local L1-SAE override

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-3)

    A = torch.from_numpy(acts).float()
    bs = 256
    steps = int(cfg["steps"])
    last = None
    for _ in range(steps):
        idx = torch.randint(0, A.shape[0], (bs,))
        a = A[idx]
        a_hat, _, loss = model.sae(a)
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.detach())
    with torch.no_grad():
        dec = model.sae.dec.weight.detach().cpu().numpy()       # [dim, sae_dim]
        # final reconstruction MSE over the whole bank (a clean diagnostic).
        a_hat_all, _, _ = model.sae(A)
        recon_mse = float(((a_hat_all - A) ** 2).mean().item())
    return dec.astype(np.float32), recon_mse, model


def _readout(acts: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Linear-probe score per item: w . a + b (the ruler's raw, smooth reading)."""
    return acts @ w + b


def measure_cfs(variant: str, cfg: dict, acts, dirs, dec, W, b, U_r, model,
                target_concept: int = 0, edit_dir_override=None, resid_basis=None):
    """MEASURE (monotonicity, specificity, sufficiency) -> CFS for one steerer.

    Every value is computed FROM THE DATA — nothing is read from a lookup table:
      * Monotonicity = Spearman(knob, target readout), negatives clipped to 0.
      * Specificity  = 1 - (mean off-target probe drift / target move), in [0,1].
      * Sufficiency  = Cohen's-d effect size at full knob, mapped to [0,1].
      * CFS          = harmonic mean of the three (src.utils.cfs_score).
    The knob sweep, probe rulers, and steerers are identical across ablations; the
    only thing that changes is whichever knob the calling ablation overrode (which
    is already baked into `dec` / `U_r` / `model` before we get here).

    Returns a dict with the three components, cfs, and the measured off-manifold
    residual of the effective edit (a diagnostic). All of cfs in [0,1] by
    construction (cfs_score clips each component to [0,1] and takes a harmonic
    mean of non-negative numbers).
    """
    import torch
    from scipy.stats import spearmanr

    from src.model import build_steer
    from src.utils import cfs_score, onmanifold_projection_residual

    tgt = int(target_concept)
    n_c = int(cfg["n_concepts"])
    s_max = float(cfg["steer_strength"])
    n_steps = int(cfg["n_knob_steps"])
    d_ample = float(cfg["cohen_d_ample"])

    acts_t = torch.from_numpy(acts).float()                # [N, dim]
    U_r_t = torch.from_numpy(U_r).float()                  # [dim, r]
    dec_t = torch.from_numpy(dec).float()                  # [dim, sae_dim]
    tgt_dir = torch.from_numpy(dirs[tgt]).float()          # planted target dir

    # Which SAE decoder column best matches the planted target concept?
    cos = (dec_t.T @ tgt_dir) / (dec_t.norm(dim=0) + 1e-8) / (tgt_dir.norm() + 1e-8)
    concept = int(cos.abs().argmax())                      # the matching SAE feature
    sign = float(torch.sign(cos[concept]))                 # align edit with +target
    edit_dir = dec_t[:, concept] * sign                    # raw SAE edit direction
    if edit_dir_override is not None:
        # A3 supplies the raw edit direction directly (concept part + off-sheet
        # leak) so the projection-rank knob's geometry is exactly the one we teach.
        edit_dir = torch.from_numpy(np.asarray(edit_dir_override)).float()
    supervised_dir = torch.from_numpy(W[tgt]).float()      # TCAV-style direction

    w_tgt, b_tgt = W[tgt], float(b[tgt])
    off_idx = [c for c in range(n_c) if c != tgt]

    def _steer(a_t, s):
        if variant == "supervised_steer":
            d = supervised_dir / (supervised_dir.norm() + 1e-8)
            return a_t + s * d
        steer = build_steer(variant, model.cfg)
        return steer(a_t, edit_dir, s, sae=model.sae, concept=concept, basis=U_r_t)

    knobs = np.linspace(0.0, s_max, n_steps)
    tgt_means, off_means = [], {c: [] for c in off_idx}
    tgt_at_0 = tgt_at_max = None
    with torch.no_grad():
        for j, s in enumerate(knobs):
            a_s = _steer(acts_t, float(s)).cpu().numpy()
            tgt_means.append(float(_readout(a_s, w_tgt, b_tgt).mean()))
            for c in off_idx:
                off_means[c].append(float(_readout(a_s, W[c], float(b[c])).mean()))
            if j == 0:
                tgt_at_0 = _readout(a_s, w_tgt, b_tgt)
            if j == n_steps - 1:
                tgt_at_max = _readout(a_s, w_tgt, b_tgt)

    tgt_means = np.asarray(tgt_means)

    # (1) MONOTONICITY = Spearman(knob, target readout), negatives -> 0.
    rho, _ = spearmanr(knobs, tgt_means)
    if np.isnan(rho):
        rho = 0.0
    monotonicity = float(max(rho, 0.0))

    # (2) SPECIFICITY = 1 - normalized off-target drift.
    tgt_move = abs(tgt_means.max() - tgt_means.min()) + 1e-6
    drifts = [abs(np.asarray(off_means[c]).max() - np.asarray(off_means[c]).min())
              for c in off_idx]
    mean_drift = float(np.mean(drifts))
    specificity = float(np.clip(1.0 - mean_drift / tgt_move, 0.0, 1.0))

    # (3) SUFFICIENCY = Cohen's-d effect size at full knob, mapped to [0,1].
    r0 = np.asarray(tgt_at_0); r1 = np.asarray(tgt_at_max)
    pooled = (r0.std() + r1.std()) / 2.0 + 1e-6
    cohen_d = float(abs(r1.mean() - r0.mean()) / pooled)
    sufficiency = float(min(cohen_d / d_ample, 1.0))

    cfs = float(cfs_score(monotonicity, specificity, sufficiency))

    # Diagnostic: off-manifold residual of the EFFECTIVE edit (a' - a) at full knob.
    # By default we measure against the steerer's own basis U_r; A3 passes a FIXED
    # reference sheet (resid_basis) so naive's residual is constant and on-manifold's
    # residual reflects how far the projected edit sits from the TRUE sheet.
    resid_t = (torch.from_numpy(np.asarray(resid_basis)).float()
               if resid_basis is not None else U_r_t)
    with torch.no_grad():
        a0 = acts_t
        a1 = _steer(acts_t, s_max)
        eff = (a1 - a0).reshape(-1, a0.shape[-1]).mean(0)
        off_resid = float(onmanifold_projection_residual(eff, resid_t))

    return {
        "variant": variant,
        "monotonicity": round(monotonicity, 4),
        "specificity": round(specificity, 4),
        "sufficiency": round(sufficiency, 4),
        "cfs": round(cfs, 4),
        "offmanifold_residual": round(off_resid, 4),
        "cohen_d": round(cohen_d, 3),
        "sae_feature": concept,
    }


def append_rows(rows: list, csv_name: str = "ablations.csv") -> None:
    """Write/append ablation rows to outputs/ablations.csv (one schema for all 5).

    Schema (the contract's required columns + a couple of shared components):
      ablation_id, knob_value, variant, cfs, diagnostic, diagnostic_name,
      monotonicity, specificity, sufficiency, offmanifold_residual
    `diagnostic` is the ablation-RELEVANT extra number (e.g. reconstruction MSE for
    A1/A2, off-manifold residual for A3, reliable-concept fraction for A4); its
    label travels in `diagnostic_name` so one CSV holds all five ablations.
    Each step calls this once with its own rows; run_all.py clears the file first.
    """
    import csv
    fields = ["ablation_id", "knob_value", "variant", "cfs", "diagnostic",
              "diagnostic_name", "monotonicity", "specificity", "sufficiency",
              "offmanifold_residual"]
    path = outpath(csv_name)
    new = not pathlib.Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def fresh_csv(csv_name: str = "ablations.csv") -> None:
    """Delete outputs/ablations.csv so a full run starts clean (run_all calls this)."""
    p = pathlib.Path(outpath(csv_name))
    if p.exists():
        p.unlink()
