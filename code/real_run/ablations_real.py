#!/usr/bin/env python3
# ===========================================================================
#  ablations_real.py  —  FAITH-SAE REAL RUN  ·  ablations A1..A5
#  --------------------------------------------------------------------------
#  Each ablation turns ONE design knob across a grid, HOLDS EVERYTHING ELSE
#  FIXED, retrains / re-projects / re-selects only what that knob touches (at a
#  REDUCED token budget so the full sweep fits in a few GPU-hours), and re-
#  measures the Causal Faithfulness Score (CFS). The five knobs are exactly the
#  ones in DESIGN_BRIEF §10:
#
#    A1  SAE type            — TopK vs L1 (vanilla) SAE.
#    A2  TopK k              — the sparsity level.
#    A3  projection rank r   — the CORE knob; locate the CFS knee (over- vs
#                              under-projection). r -> d_in is naive steering.
#    A4  selection threshold — the "well-defined concept" filter strictness and
#                              the resulting reliable-concept fraction (~10-15%).
#    A5  layer / token       — backbone layer & patch-vs-CLS token choice.
#
#  The output table outputs/ablations.csv (one row per (ablation, knob value))
#  is what the ablation figures plot. On the real path each ablation retrains or
#  re-derives only the artefact its knob controls; on --smoke we run ONE value
#  per ablation on fabricated real-SHAPED activations so the whole grid imports
#  and executes on CPU with no open_clip / no downloads.
#
#  author: Rajia Rani
#  For research and educational purposes only.
# ===========================================================================
#
#  WHY EACH ABLATION RETRAINS SOMETHING DIFFERENT
#  ----------------------------------------------
#    * A1/A2 change the SAE itself (type / k) -> the SAE must be RETRAINED on the
#      reduced-budget activation stream before its concepts can be scored.
#    * A3 changes only the on-manifold basis U_r -> re-RUN the SVD at each rank
#      on the SAME activation bank; the SAE is unchanged (cheapest ablation).
#    * A4 changes only the concept-selection threshold -> re-SELECT concepts from
#      the SAME trained SAE; nothing is retrained.
#    * A5 changes which layer / token type the activations come from -> the cache
#      itself differs, so the SAE is RETRAINED on the alternative activations.
#  Centralising "the SAE + the bank + the basis + the concepts" and only varying
#  the one knob is what makes each row a clean controlled comparison.
#
#  REAL-RUN CAVEAT
#  ---------------
#  The real path needs open_clip + ImageNet cached activations (per layer/token
#  for A5) + a working train_sae / sae_real / manifold / concept_select. None of
#  those exist on this build machine and there is no GPU. So the default real
#  path imports the sibling modules lazily and raises a clear, actionable error
#  if a prerequisite is missing; `--smoke` runs the whole A1..A5 grid (one value
#  each) on fabricated activations on CPU.

from __future__ import annotations

import argparse
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils import cfs_score  # noqa: E402  (single source of truth, brief §13)

# Reuse the OOD sweep's smoke fabric + fallback scorer so the two modules score
# CFS IDENTICALLY (a shared scorer is itself a controlled variable). These are
# pure-CPU helpers with no open_clip dependency, so importing them is always
# safe; the import is guarded only to keep ablations_real importable even if
# ood_sweep is mid-edit during a parallel build.
try:
    from ood_sweep import (_build_clean_probe_bank, _fallback_compute_cfs,
                           _planted_basis, _smoke_bank, build_ladder,
                           load_real_config)
    _HAVE_SWEEP = True
except Exception:  # pragma: no cover - only during a broken parallel build
    _HAVE_SWEEP = False
    _build_clean_probe_bank = None

    def load_real_config(path: str) -> dict:
        from src.utils import load_config
        return load_config(path)


# --------------------------------------------------------------------------- #
# Smoke fabric (delegates to ood_sweep's helpers when available, else a tiny  #
# local copy so this file is independently runnable).                         #
# --------------------------------------------------------------------------- #
def _sheet(d_in, true_rank, seed=0):
    if _HAVE_SWEEP:
        return _planted_basis(d_in, true_rank, seed=seed)
    import torch
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d_in, d_in, generator=g))
    return Q[:, :true_rank].contiguous()


def _bank(d_in, n_tokens, sheet, shift_index=0, severity=None, seed=0):
    if _HAVE_SWEEP:
        return _smoke_bank(d_in, n_tokens, sheet, shift_index, severity, seed=seed)
    import torch
    g = torch.Generator().manual_seed(seed + 7919 * (shift_index + 1))
    coords = torch.randn(n_tokens, sheet.shape[1], generator=g)
    return (coords @ sheet.T + 0.02 * torch.randn(n_tokens, d_in, generator=g)).float()


def _subspace(bank, rank):
    """Top-`rank` PCA directions of a bank -> [d, rank] orthonormal torch basis.
    Mirrors manifold.estimate_manifold_basis (centred SVD) so A3's smoke basis is
    derived the same way the real basis is."""
    import torch
    X = torch.as_tensor(bank, dtype=torch.float32)
    X = X - X.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    rr = min(int(rank), Vh.shape[0])
    return Vh[:rr].T.contiguous()


# --------------------------------------------------------------------------- #
# Shared scoring on the SMOKE path: plant concepts ON the sheet and score each #
# variant with the same fallback probe the OOD sweep uses.                     #
# --------------------------------------------------------------------------- #
def _smoke_score(cfg, sheet, U_r, n_concepts, variant="onmanifold_steer",
                 eval_bank=None, snr=1.0, seed=0):
    """Average CFS over `n_concepts` planted concepts for one steering variant on
    a fabricated bank. `U_r` is the on-manifold basis the on-manifold steerer
    projects onto (A3 varies its rank). `snr` lets A5 model that some layers /
    tokens carry the concept more cleanly. Returns the mean component dict."""
    import torch
    d_in = sheet.shape[0]
    true_rank = sheet.shape[1]
    if eval_bank is None:
        eval_bank = _bank(d_in, min(4096, int(cfg.get("steering", {})
                          .get("bank_tokens", 4096))), sheet, seed=seed)

    agg = {"monotonicity": [], "specificity": [], "sufficiency": [],
           "cfs": [], "offmanifold_residual": []}
    for j in range(n_concepts):
        cd = sheet[:, j % true_rank].clone() * snr        # snr scales the signal
        od = sheet[:, (j + 1) % true_rank].clone()
        U_for = U_r if variant == "onmanifold_steer" else None
        if _HAVE_SWEEP:
            m = _fallback_compute_cfs(variant, eval_bank, U_for, cfg, cd, od,
                                      seed=seed + j)
        else:                                             # minimal inline fallback
            m = _inline_cfs(variant, eval_bank, U_for, cfg, cd, od)
        for key in agg:
            if key in m:
                agg[key].append(float(m[key]))
    return {key: (sum(v) / len(v) if v else float("nan"))
            for key, v in agg.items()}


def _inline_cfs(variant, eval_acts, U_r, cfg, concept_dir, off_dir):
    """Tiny self-contained CFS probe (only used if ood_sweep failed to import)."""
    import torch
    grid = [float(s) for s in cfg.get("steering", {}).get(
        "strength_grid", [0, 0.5, 1, 2, 4])]

    def _u(v):
        return v / (v.norm() + 1e-8)

    raw = _u(concept_dir)
    if variant == "onmanifold_steer" and U_r is not None:
        edit = _u(U_r @ (U_r.T @ raw))
    else:
        edit = raw
    a0 = eval_acts.float()
    tr = torch.tensor([float(((a0 + s * edit) * _u(concept_dir)).sum(-1).mean())
                       for s in grid])
    ofr = torch.tensor([float(((a0 + s * edit) * _u(off_dir)).sum(-1).mean())
                        for s in grid])
    k = torch.tensor(grid)
    ar = k.argsort().argsort().float(); br = tr.argsort().argsort().float()
    ar = ar - ar.mean(); br = br - br.mean()
    mono = max(float((ar * br).sum() / (ar.norm() * br.norm() + 1e-8)), 0.0)
    spec = float(max(0.0, 1.0 - (ofr.max() - ofr.min()).abs()
                     / ((tr.max() - tr.min()).abs() + 1e-6)))
    suff = min(float((tr.max() - tr.min()).abs() / (tr.std() + 1e-6)) / 4.0, 1.0)
    return {"monotonicity": round(mono, 4), "specificity": round(spec, 4),
            "sufficiency": round(suff, 4),
            "cfs": round(cfs_score(mono, spec, suff), 4),
            "offmanifold_residual": 0.0 if variant == "onmanifold_steer" else 1.0}


# --------------------------------------------------------------------------- #
# Real-path SAE (re)training helper at a REDUCED token budget.                #
# --------------------------------------------------------------------------- #
def _train_sae_reduced(cfg, cache_dir, overrides: dict):
    """Retrain the SAE for an ablation that changes the SAE itself (A1/A2/A5),
    at a reduced token budget so the whole grid stays cheap. Delegates to
    train_sae.train_sae after patching the knob (sae_type / k / layer / token).

    Raises a clear error on the build machine (no open_clip / no cache). Returns
    a trained TopKSAE (sae_real.TopKSAE)."""
    import copy
    rc = copy.deepcopy(cfg)
    rc.setdefault("sae", {})
    # Reduced budget for ablations (a fraction of the headline run's budget).
    abl = cfg.get("ablations", {}) or {}
    rc["sae"]["token_budget"] = int(abl.get("token_budget",
                                  cfg["sae"].get("token_budget", 30_000_000) // 10))
    for k, v in overrides.items():
        # overrides may target cfg['sae'][...] or cfg['backbone'][...]
        if k in ("sae_type", "k", "expansion"):
            rc["sae"][k] = v
        else:
            rc.setdefault("backbone", {})[k] = v
    try:
        import train_sae
    except Exception as e:
        raise RuntimeError(
            "ablations_real real path needs train_sae.py + a populated activation "
            f"cache: {e}. Use --smoke for the offline CPU path.") from e
    return train_sae.train_sae(rc, cache_dir)


def _load_real_bank(cfg, cache_dir, dataset="clean"):
    """Clean activation bank for basis estimation / scoring on the real path."""
    try:
        import data_real
    except Exception as e:
        raise RuntimeError(
            f"ablations_real real path needs data_real.py + cache: {e}.") from e
    n = int(cfg.get("steering", {}).get("bank_tokens", 2_000_000))
    return data_real.load_activation_bank(cache_dir, dataset, n,
                                          seed=int(cfg.get("seed", 0)))


def _score_real(cfg, sae, variant, concept_ids, eval_acts, probes, U_r,
                probe_concept_ids=None):
    """Mean CFS over concepts on the real path via cfs_eval.compute_cfs.

    ``probes`` is a frozen probes.ProbeBank (built once by _real_probes), NOT the
    module. cfs_eval reads probes.target_concept for the target readout, so we
    re-point the bank's target at a class probe per SAE concept (round-robin over
    probe_concept_ids) — the probes are never refit, only the target/off-target
    split rotates, keeping every ablation row a controlled comparison.
    """
    try:
        import cfs_eval
    except Exception as e:
        raise RuntimeError(
            f"ablations_real real path needs cfs_eval.compute_cfs: {e}.") from e
    agg = {"monotonicity": [], "specificity": [], "sufficiency": [],
           "cfs": [], "offmanifold_residual": []}
    for i, cid in enumerate(concept_ids):
        if probe_concept_ids:
            probes.target_concept = probe_concept_ids[i % len(probe_concept_ids)]
        m = cfs_eval.compute_cfs(sae, variant, cid, eval_acts, probes, U_r, cfg)
        for key in agg:
            if key in m:
                agg[key].append(float(m[key]))
    return {key: (sum(v) / len(v) if v else float("nan"))
            for key, v in agg.items()}


def _real_probes(cfg, cache_dir):
    """Build the frozen clean-data ProbeBank once for the whole ablation grid.

    Delegates to ood_sweep._build_clean_probe_bank (the SAME probe construction
    the OOD sweep uses, so the two modules score CFS identically). Returns
    (probe_bank, probe_concept_ids); raises a clear error if the sweep helper is
    unavailable (a broken parallel build)."""
    if _build_clean_probe_bank is None:
        raise RuntimeError(
            "ablations_real real path needs ood_sweep._build_clean_probe_bank "
            "(import failed). Ensure ood_sweep.py is present.")
    return _build_clean_probe_bank(cfg, cache_dir)


# --------------------------------------------------------------------------- #
# Grid resolution: full grids on the real path, ONE value each on --smoke.    #
# --------------------------------------------------------------------------- #
def _grids(cfg: dict, smoke: bool) -> dict:
    """Return the per-ablation knob grids. On --smoke each grid is truncated to a
    single value so the whole A1..A5 sweep runs in seconds on CPU; on the real
    path the full grids from cfg.ablations (with sensible defaults) are used."""
    abl = cfg.get("ablations", {}) or {}
    d_in = int(cfg["sae"]["d_in"])
    base_r = int(cfg.get("steering", {}).get("proj_rank_r", 512))
    full = {
        "A1": list(abl.get("a1_sae_type", ["topk", "l1"])),
        "A2": list(abl.get("a2_k", [16, 32, 64])),
        # A3 spans under-projection -> over-projection (r -> d_in is naive).
        "A3": list(abl.get("a3_proj_rank", [base_r // 8, base_r // 2, base_r,
                                           min(2 * base_r, d_in)])),
        "A4": list(abl.get("a4_select_thresh", [0.3, 0.5, 0.7, 0.9])),
        # A5: (layer, token_type) pairs.
        "A5": list(abl.get("a5_layer_token",
                   [["22", "patch"], ["22", "cls"], ["18", "patch"]])),
    }
    if smoke:
        # One representative value per ablation. For categorical knobs (A1/A5)
        # take the first; for the numeric sweeps (A2/A3/A4) take a MIDDLE value
        # so the single smoke point is representative rather than a degenerate
        # extreme (e.g. A3's smallest rank over-constrains and reads near-zero).
        def _mid(v):
            return [v[len(v) // 2]] if v else v
        return {"A1": full["A1"][:1], "A2": _mid(full["A2"]),
                "A3": _mid(full["A3"]), "A4": _mid(full["A4"]),
                "A5": full["A5"][:1]}
    return full


# --------------------------------------------------------------------------- #
# The driver.                                                                 #
# --------------------------------------------------------------------------- #
def run_ablations(cfg: dict, cache_dir: str, *, smoke: bool = False):
    """Run ablations A1..A5 and return a pandas DataFrame (also written to
    cfg.paths.out_dir/ablations.csv).

    Schema mirrors the milestone teaching CSV so the analysis/figure code can
    read either: ablation_id, ablation_name, knob, knob_value, variant, the
    three CFS components, cfs, and a per-ablation diagnostic (off-manifold
    residual for A3, reliable-concept fraction for A4, recon-ish marker else).
    """
    import pandas as pd

    out_dir = pathlib.Path(cfg.get("paths", {}).get("out_dir",
                                                    str(_HERE / "outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablations.csv"

    grids = _grids(cfg, smoke)
    d_in = int(cfg["sae"]["d_in"])
    base_r = int(cfg.get("steering", {}).get("proj_rank_r", 512))
    n_concepts = max(2, int(cfg.get("cfs", {}).get("n_probe_classes", 6)))

    print("=" * 72)
    print("FAITH-SAE — ABLATIONS A1..A5 (one knob each, all else fixed)")
    print("=" * 72)
    print(f"  mode: {'SMOKE (synthetic, CPU)' if smoke else 'REAL'}")
    print(f"  grids: { {k: grids[k] for k in grids} }")

    # --- Shared smoke fabric (the 'all else fixed' baseline) ----------------
    if smoke:
        true_rank = min(base_r, max(2, d_in // 2))
        sheet = _sheet(d_in, true_rank, seed=int(cfg.get("seed", 0)))
        n_concepts = min(n_concepts, true_rank)
        clean_bank = _bank(d_in, min(4096, int(cfg.get("steering", {})
                           .get("bank_tokens", 4096))), sheet,
                           seed=int(cfg.get("seed", 0)))
    else:
        sheet = clean_bank = None

    # --- Real path: build the frozen clean-data ProbeBank ONCE (reused by every
    # ablation row; the probes are an 'all else fixed' control, never refit). ---
    probe_bank = probe_concept_ids = None
    if not smoke:
        probe_bank, probe_concept_ids = _real_probes(cfg, cache_dir)

    rows: list = []

    # ===================================================================== A1
    # SAE TYPE: TopK vs L1. Changing the SAE type RETRAINS the SAE; the steerer
    # is fixed to on-manifold (the method whose faithfulness we report).
    for sae_type in grids["A1"]:
        if smoke:
            # L1's denser code drifts more off-manifold -> a small specificity
            # penalty vs TopK; we model that by a slightly lower SNR for l1.
            snr = 1.0 if sae_type == "topk" else 0.85
            U_r = _subspace(clean_bank, base_r)
            m = _smoke_score(cfg, sheet, U_r, n_concepts,
                             variant="onmanifold_steer", eval_bank=clean_bank,
                             snr=snr, seed=int(cfg.get("seed", 0)))
            diag = ("dense_code_l1" if sae_type == "l1" else "topk_sparse")
            diag_val = snr
        else:
            sae = _train_sae_reduced(cfg, cache_dir, {"sae_type": sae_type})
            bank = _load_real_bank(cfg, cache_dir)
            import manifold
            U_r = manifold.estimate_manifold_basis(bank, base_r)
            import concept_select
            cids = concept_select.select_concepts(sae, bank, None, cfg)[:n_concepts]
            m = _score_real(cfg, sae, "onmanifold_steer", cids, bank,
                            probe_bank, U_r, probe_concept_ids)
            diag, diag_val = "sae_type", sae_type
        rows.append(_row("A1", "sae_type", "sae_type", sae_type,
                         "onmanifold_steer", m, diag, diag_val))

    # ===================================================================== A2
    # TopK k (sparsity). Changing k RETRAINS the TopK SAE. Too-small k starves
    # concepts (effect dies); too-large k -> dense/polysemantic (specificity
    # leaks). We expect a CFS hump over k.
    for k in grids["A2"]:
        if smoke:
            # Model k's sweet spot: CFS peaks near a mid k via an snr bump and a
            # specificity penalty at extremes. Here we just vary snr with a hump.
            ks = grids["A2"]
            mid = ks[len(ks) // 2]
            snr = 1.0 - 0.12 * abs(float(k) - float(mid)) / (float(mid) + 1e-6)
            U_r = _subspace(clean_bank, base_r)
            m = _smoke_score(cfg, sheet, U_r, n_concepts, eval_bank=clean_bank,
                             snr=max(0.6, snr), seed=int(cfg.get("seed", 0)) + int(k))
            diag, diag_val = "topk_k", k
        else:
            sae = _train_sae_reduced(cfg, cache_dir, {"k": int(k)})
            bank = _load_real_bank(cfg, cache_dir)
            import manifold
            U_r = manifold.estimate_manifold_basis(bank, base_r)
            import concept_select
            cids = concept_select.select_concepts(sae, bank, None, cfg)[:n_concepts]
            m = _score_real(cfg, sae, "onmanifold_steer", cids, bank,
                            probe_bank, U_r, probe_concept_ids)
            diag, diag_val = "topk_k", k
        rows.append(_row("A2", "topk_k", "k", k, "onmanifold_steer", m,
                         diag, diag_val))

    # ===================================================================== A3
    # PROJECTION RANK r (the CORE knob). Only U_r changes: re-run the SVD at each
    # rank on the SAME bank; the SAE is unchanged. r too small over-constrains
    # (effect dies); r -> d_in is naive steering (off-manifold residual rises).
    for r in grids["A3"]:
        r = int(r)
        if smoke:
            U_r = _subspace(clean_bank, r)
            m = _smoke_score(cfg, sheet, U_r, n_concepts, eval_bank=clean_bank,
                             seed=int(cfg.get("seed", 0)))
        else:
            bank = _load_real_bank(cfg, cache_dir)
            import manifold
            U_r = manifold.estimate_manifold_basis(bank, r)
            import sae_real, concept_select
            sae = sae_real.load_sae(cfg.get("paths", {}).get(
                "sae_ckpt", "./outputs/sae.safetensors"))
            cids = concept_select.select_concepts(sae, bank, None, cfg)[:n_concepts]
            m = _score_real(cfg, sae, "onmanifold_steer", cids, bank,
                            probe_bank, U_r, probe_concept_ids)
        rows.append(_row("A3", "manifold_projection_rank_r", "proj_rank", r,
                         "onmanifold_steer", m, "offmanifold_residual",
                         m.get("offmanifold_residual", "")))

    # ===================================================================== A4
    # SELECTION THRESHOLD. Only the concept-selection filter changes: re-select
    # from the SAME SAE. Stricter threshold -> fewer but cleaner concepts -> the
    # ~10-15% reliable tail. We log the kept fraction as the diagnostic.
    for thresh in grids["A4"]:
        thresh = float(thresh)
        if smoke:
            # Stricter threshold keeps fewer concepts but they score higher; we
            # model 'kept fraction' shrinking with threshold and a mild CFS lift.
            kept_frac = round(max(0.05, 1.0 - thresh), 4)
            n_keep = max(1, int(round(n_concepts * kept_frac)))
            snr = 0.9 + 0.1 * thresh                       # cleaner survivors
            U_r = _subspace(clean_bank, base_r)
            m = _smoke_score(cfg, sheet, U_r, n_keep, eval_bank=clean_bank,
                             snr=min(1.0, snr), seed=int(cfg.get("seed", 0)))
            diag, diag_val = "reliable_fraction", kept_frac
        else:
            import sae_real, concept_select, manifold
            sae = sae_real.load_sae(cfg.get("paths", {}).get(
                "sae_ckpt", "./outputs/sae.safetensors"))
            bank = _load_real_bank(cfg, cache_dir)
            # Patch only the selection threshold; everything else held fixed.
            rc = {**cfg, "concept_select_thresh": thresh}
            cids = concept_select.select_concepts(sae, bank, None, rc)
            kept_frac = round(len(cids) / max(1, getattr(sae, "n_features", len(cids))), 4)
            cids = cids[:n_concepts]
            U_r = manifold.estimate_manifold_basis(bank, base_r)
            m = _score_real(cfg, sae, "onmanifold_steer", cids, bank,
                            probe_bank, U_r, probe_concept_ids)
            diag, diag_val = "reliable_fraction", kept_frac
        rows.append(_row("A4", "concept_selection_threshold", "select_thresh",
                         thresh, "onmanifold_steer", m, diag, diag_val))

    # ===================================================================== A5
    # LAYER / TOKEN. The activation cache itself differs (layer + patch vs CLS),
    # so the SAE is RETRAINED on the alternative activations. CLS tokens carry a
    # coarser, more global signal -> usually lower per-concept CFS than patch.
    for layer, token in grids["A5"]:
        if smoke:
            # Model token quality: patch tokens read the concept more cleanly
            # than CLS; deeper layers slightly cleaner here.
            snr = (1.0 if token == "patch" else 0.8) * (1.0 if str(layer) == "22" else 0.93)
            U_r = _subspace(clean_bank, base_r)
            m = _smoke_score(cfg, sheet, U_r, n_concepts, eval_bank=clean_bank,
                             snr=snr, seed=int(cfg.get("seed", 0)))
            diag, diag_val = "layer_token", f"{layer}/{token}"
        else:
            sae = _train_sae_reduced(cfg, cache_dir,
                                     {"layer": int(layer), "token_type": token})
            bank = _load_real_bank(cfg, cache_dir)
            import manifold, concept_select
            U_r = manifold.estimate_manifold_basis(bank, base_r)
            cids = concept_select.select_concepts(sae, bank, None, cfg)[:n_concepts]
            # A5 changes the activation SPACE (layer/token), so the default-layer
            # probe bank no longer aligns; rebuild probes on THIS cache so the
            # rulers match the activations being scored.
            a5_cache = cfg.get("ablations", {}).get("a5_cache_dir", cache_dir)
            a5_bank, a5_pcids = _real_probes(cfg, a5_cache)
            m = _score_real(cfg, sae, "onmanifold_steer", cids, bank,
                            a5_bank, U_r, a5_pcids)
            diag, diag_val = "layer_token", f"{layer}/{token}"
        rows.append(_row("A5", "layer_and_token_choice", "layer_token",
                         f"{layer}/{token}", "onmanifold_steer", m,
                         diag, diag_val))

    # --- Assemble, report the per-ablation best, write the CSV --------------
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    for aid in ["A1", "A2", "A3", "A4", "A5"]:
        sub = df[df["ablation_id"] == aid]
        if len(sub):
            best = sub.loc[sub["cfs"].astype(float).idxmax()]
            print(f"  {aid} {best['ablation_name']:<28} best CFS="
                  f"{float(best['cfs']):.4f} at {best['knob']}={best['knob_value']}")
    print(f"\n  wrote {csv_path}  ({len(df)} rows)")
    return df


def _row(ablation_id, ablation_name, knob, knob_value, variant, m, diag,
         diag_val) -> dict:
    """Assemble one ablations.csv row in the shared schema."""
    return {
        "ablation_id": ablation_id,
        "ablation_name": ablation_name,
        "knob": knob,
        "knob_value": knob_value,
        "variant": variant,
        "monotonicity": m.get("monotonicity", ""),
        "specificity": m.get("specificity", ""),
        "sufficiency": m.get("sufficiency", ""),
        "cfs": m.get("cfs", ""),
        "diagnostic": diag,
        "diagnostic_value": diag_val,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _default_smoke_cfg() -> dict:
    """Tiny self-contained config for the CPU ablation smoke (one value per
    ablation). Real-SHAPED (d_in=64) but small; no YAML / open_clip / downloads."""
    return {
        "seed": 0,
        "sae": {"d_in": 64, "expansion": 8, "k": 32, "token_budget": 1_000_000},
        "steering": {"strength_grid": [0, 0.5, 1, 2, 4], "proj_rank_r": 16,
                     "bank_tokens": 4096},
        "cfs": {"n_probe_classes": 6, "bootstrap_n": 200},
        "ablations": {"token_budget": 100_000},
        "paths": {"out_dir": str(_HERE / "outputs")},
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="FAITH-SAE ablations A1..A5. Real path is the default; "
                    "--smoke runs one knob value per ablation on synthetic-but-"
                    "real-shaped activations on CPU (no open_clip, no downloads).")
    ap.add_argument("--config", type=str, default=None,
                    help="path to the real-run YAML config")
    ap.add_argument("--cache_dir", type=str, default=None,
                    help="activation cache dir (data_real shards). "
                         "Defaults to cfg.paths.cache_dir or ./cache")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny offline CPU ablation grid on fabricated activations")
    args = ap.parse_args()

    if args.smoke:
        cfg = _default_smoke_cfg()
        if args.config:
            cfg = load_real_config(args.config)
        cache_dir = args.cache_dir or str(_HERE / "cache")
        run_ablations(cfg, cache_dir, smoke=True)
        return

    if not args.config:
        ap.error("real path requires --config (or pass --smoke for the CPU path)")
    cfg = load_real_config(args.config)
    cache_dir = (args.cache_dir or cfg.get("paths", {}).get("cache_dir")
                 or str(_HERE / "cache"))
    run_ablations(cfg, cache_dir, smoke=False)


if __name__ == "__main__":
    main()
