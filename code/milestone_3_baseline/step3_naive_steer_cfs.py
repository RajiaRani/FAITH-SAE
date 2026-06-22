#!/usr/bin/env python3
# ===========================================================================
#  step3_naive_steer_cfs.py  —  Milestone 3 (Baseline), Part B
#  Run the BASELINE steerer (naive_steer) on each selected concept and measure
#  its Causal Faithfulness Score (CFS). This is the number milestone 4's
#  on-manifold method must BEAT.
#  FAITH-SAE  ·  author: Rajia Rani  ·  educational use only
# ===========================================================================
#
#  ============ TERMS, FROM ZERO ============
#
#  STEERING
#    Deliberately EDITING a network's internal activation to turn a concept up or
#    down, then seeing what changes. Like adjusting the bass knob on a stereo:
#    you push one control and listen for the effect.
#
#  STEERING STRENGTH  s
#    HOW HARD you push. We move the activation by `s` units along the concept's
#    direction d (a unit vector). s = 0 -> no edit; bigger s -> bigger push.
#    Example: a = [1, 0], d = [0, 1], s = 3  ->  a' = a + s*d = [1, 3].
#    Every steering method in this study uses the SAME s (matched strength), so
#    differences come from the METHOD, not from pushing one harder.
#
#  BASELINE  (and why naive is the WEAK one)
#    A "baseline" is the simplest sensible thing you compare against — the bar to
#    clear. Here the baseline is `naive_steer`: a' = a + s*d, just add the
#    direction. It is "OFF-MANIFOLD": it ignores where REAL activations actually
#    live. The set of activations a frozen model truly produces forms a thin,
#    curved region (the "manifold") inside the big 64-D space — like the surface
#    of a balloon inside a room. Adding a raw direction usually shoves the point
#    OFF that surface, into a place the model never sees in real life. The readout
#    still moves (so it LOOKS like an effect), but the edit is unrealistic and
#    leaks into other concepts. That is exactly why naive steering scores a
#    MEDIOCRE CFS — and why milestone 4 projects the edit back onto the manifold.
#    (naive_steer is the r -> infinity / no-projection special case of ours.)
#
#  CAUSAL FAITHFULNESS SCORE  (CFS)  — the headline number, in [0, 1]
#    Asks: is this edit a REAL, clean causal lever? It is the harmonic mean (an
#    "AND": all three must be high) of three parts:
#      * MONOTONICITY — turn the knob up, does the target readout go up SMOOTHLY
#        and in order? Measured by Spearman rank correlation between knob and
#        readout. 1.0 = perfectly ordered; ~0 = no consistent response.
#      * SPECIFICITY  — does ONLY the target move? We check an OFF-TARGET concept
#        readout: if it barely drifts, specificity is high; if steering one
#        concept drags an unrelated one along, specificity is low. (1 - drift.)
#      * SUFFICIENCY  — is the effect BIG enough to matter? A standardized effect
#        size (Cohen's-d style) of the readout at full knob vs no knob.
#    CFS = harmonic_mean(monotonicity, specificity, sufficiency). A near-zero in
#    ANY one drags the whole score down. Example: (0.9, 0.1, 0.9) -> CFS ~ 0.23,
#    because the 0.1 specificity poisons it. naive steering typically has decent
#    monotonicity but POOR specificity -> a mediocre CFS in the ~0.3-0.6 band.
#
#  ============ WHAT THIS SCRIPT DOES ============
#    1. Load the trained SAE + the concepts selected in step 2.
#    2. Build the BASELINE steerer `naive_steer` from the shared STEER_REGISTRY.
#    3. For each selected concept: sweep s from 0..steer_strength, measure
#       monotonicity / specificity / sufficiency, combine with the project's
#       cfs_score(). Also record the off-manifold residual diagnostic.
#    4. Write outputs/baseline_cfs.csv (one row per concept + a mean row).
#  ========================================================================

from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Reuse the project's steerer registry + the shared CFS helpers (do NOT reinvent).
from src.model import _build, build_steer  # noqa: E402
from src.utils import (  # noqa: E402
    cfs_score, load_config, onmanifold_projection_residual, set_seed,
)
from step1_train_sae import make_activation_bank  # noqa: E402


def _spearman(a, b):
    """Spearman rank correlation (Pearson on ranks); no scipy needed.
    Measures MONOTONIC (ordered) agreement: does b rise whenever a rises?"""
    import torch

    ar = a.argsort().argsort().float()
    br = b.argsort().argsort().float()
    ar = ar - ar.mean()
    br = br - br.mean()
    return float((ar * br).sum() / ((ar.norm() * br.norm()) + 1e-8))


def load_sae(cfg: dict):
    import torch

    ckpt = torch.load(os.path.join(HERE, cfg["sae_ckpt"]),
                      map_location="cpu", weights_only=False)
    FaithSAE = _build()
    model = FaithSAE(cfg)
    model.sae.load_state_dict(ckpt["state_dict"])
    return model


def read_selected(cfg: dict):
    path = os.path.join(HERE, cfg["concepts_csv"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No selected concepts at {path}. Run step2_select_concepts.py first.")
    with open(path, encoding="utf-8") as f:
        return [int(r["feature_id"]) for r in csv.DictReader(f)]


def measure_cfs_for_concept(model, steer, bank, basis, concept, off_concepts, cfg):
    """Sweep the knob on ONE concept with the BASELINE steerer; return the three
    CFS components + CFS + the off-manifold residual diagnostic.

    `off_concepts` is a LIST of unrelated feature ids whose readouts should stay
    flat. Specificity is judged on the WORST off-target (a faithful edit must not
    move ANY of them). naive off-manifold steering leaks into several of these
    because its raw edit drifts off the real-data subspace -> mediocre score."""
    import torch

    dim = cfg["dim"]
    smax = cfg["steer_strength"]
    n_steps = 6
    flat = bank.reshape(-1, dim)[:512]                # a fixed slice of activations

    with torch.no_grad():
        # Target readout direction = the concept we steer (unit decoder column).
        d_tgt = model.sae.concept_direction(concept)
        d_tgt = d_tgt / (d_tgt.norm() + 1e-8)
        # Off-target readout directions = several OTHER concepts that should NOT
        # move. We watch the worst one (max leakage) — true specificity demands
        # that steering the target leaves every unrelated probe flat.
        d_offs = []
        for oc in off_concepts:
            if oc == concept:
                continue
            do = model.sae.concept_direction(oc)
            d_offs.append(do / (do.norm() + 1e-8))

        def readout(a, d):
            return (a * d).sum(-1)                    # projection onto direction d

        # Sweep the knob s = 0 .. smax and record target + each off-target readout.
        knobs, tgt_read = [], []
        off_reads = [[] for _ in d_offs]
        for j in range(n_steps):
            s = smax * j / (n_steps - 1)
            # The BASELINE steerer: a' = a + s*d_tgt (off-manifold add).
            a_s = steer(flat, model.sae.concept_direction(concept), s,
                        sae=model.sae, concept=concept, basis=basis)
            knobs.append(s)
            tgt_read.append(float(readout(a_s, d_tgt).mean()))
            for i, do in enumerate(d_offs):
                off_reads[i].append(float(readout(a_s, do).mean()))

        knobs = torch.tensor(knobs)
        tr = torch.tensor(tgt_read)

        # MONOTONICITY: smooth ordered response of target readout to the knob.
        monotonicity = max(_spearman(knobs, tr), 0.0)
        # SPECIFICITY: 1 - (WORST off-target movement / target movement). Using
        # the worst off-target surfaces the leakage naive steering causes.
        tgt_move = (tr.max() - tr.min()).abs() + 1e-6
        worst_off_move = 0.0
        for col in off_reads:
            col = torch.tensor(col)
            worst_off_move = max(worst_off_move,
                                 float((col.max() - col.min()).abs()))
        specificity = float(max(0.0, 1.0 - worst_off_move / tgt_move))
        # SUFFICIENCY: standardized effect size at full knob vs no knob.
        base = steer(flat, model.sae.concept_direction(concept), 0.0,
                     sae=model.sae, concept=concept, basis=basis)
        full = steer(flat, model.sae.concept_direction(concept), smax,
                     sae=model.sae, concept=concept, basis=basis)
        r0 = readout(base, d_tgt)
        r1 = readout(full, d_tgt)
        pooled = (r0.std() + r1.std()) / 2 + 1e-6
        d_eff = float((r1.mean() - r0.mean()).abs() / pooled)
        sufficiency = min(d_eff / 4.0, 1.0)           # d~4 is "ample" -> 1.0

        # OFF-MANIFOLD RESIDUAL: how much of the edit lands OFF the real-data
        # subspace. naive_steer does no projection, so this is HIGH (~1.0) — the
        # diagnostic that explains its mediocre faithfulness (and what M4 fixes).
        # naive_steer adds the SAME s*d to every row, so the per-vector edit is a
        # single length-`dim` vector; take row 0 as its representative.
        edit = (full - base)[0]                        # the activation edit, [dim]
        residual = onmanifold_projection_residual(edit, basis)

    cfs = cfs_score(monotonicity, specificity, sufficiency)
    return {
        "monotonicity": round(monotonicity, 4),
        "specificity": round(specificity, 4),
        "sufficiency": round(sufficiency, 4),
        "offmanifold_residual": round(residual, 4),
        "cfs": round(cfs, 4),
    }


def estimate_manifold_basis(bank, cfg):
    """Top-r real-image subspace U_r (for the residual diagnostic only — naive
    steering itself does NOT project). PCA of the activation bank via SVD: the
    leading right singular vectors are the directions activations actually use."""
    import torch

    flat = bank.reshape(-1, cfg["dim"])
    flat = flat - flat.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(flat, full_matrices=False)
    r = min(cfg.get("proj_rank", 16), Vh.shape[0])
    return Vh[:r].T                                   # [dim, r] = U_r


def run(cfg: dict):
    set_seed(cfg["seed"])
    model = load_sae(cfg)
    bank, _ = make_activation_bank(cfg)
    basis = estimate_manifold_basis(bank, cfg)
    concepts = read_selected(cfg)

    # Build the BASELINE steerer by name from the shared registry (the contract's
    # naive_steer). One line — this is the whole point of the pluggable design.
    steer = build_steer(cfg.get("steer", "naive_steer"), cfg)
    print(f"[steer] baseline method = '{cfg.get('steer', 'naive_steer')}'  "
          f"(off-manifold activation addition, a <- a + s*d)")
    print(f"[steer] matched strength s = {cfg['steer_strength']}, "
          f"steering {len(concepts)} selected concepts")

    # Off-target probe panel: the OTHER selected concepts plus a spread of random
    # SAE features. A faithful edit must leave ALL of these flat; naive
    # off-manifold steering leaks into some -> the mediocre-specificity story.
    import torch
    g = torch.Generator().manual_seed(cfg["seed"] + 5)
    rand_offs = torch.randint(0, cfg["sae_dim"], (12,), generator=g).tolist()
    off_panel = sorted(set(concepts) | set(rand_offs))

    rows = []
    for c in concepts:
        m = measure_cfs_for_concept(model, steer, bank, basis, c, off_panel, cfg)
        rows.append({"variant": cfg.get("steer", "naive_steer"),
                     "feature_id": c, **m})
        print(f"    concept {c:3d}:  mono={m['monotonicity']:.2f}  "
              f"spec={m['specificity']:.2f}  suff={m['sufficiency']:.2f}  "
              f"offman={m['offmanifold_residual']:.2f}  ->  CFS={m['cfs']:.3f}")

    # Append a MEAN row — the single baseline number M4 must beat.
    def avg(key):
        return round(sum(r[key] for r in rows) / len(rows), 4)

    mean_row = {"variant": "MEAN", "feature_id": -1,
                "monotonicity": avg("monotonicity"),
                "specificity": avg("specificity"),
                "sufficiency": avg("sufficiency"),
                "offmanifold_residual": avg("offmanifold_residual"),
                "cfs": avg("cfs")}
    rows.append(mean_row)

    out = os.path.join(HERE, cfg["baseline_csv"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n[save] baseline CFS table -> {out}")
    print(f"[BASELINE] mean naive_steer CFS = {mean_row['cfs']:.3f}  "
          f"(off-manifold residual = {mean_row['offmanifold_residual']:.2f}). "
          f"This is the bar milestone_4 (on-manifold) must beat.")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Baseline naive_steer + CFS.")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg)
    print("\nDONE step 3. See outputs/baseline_cfs.csv. "
          "Next milestone: code/milestone_4_method (on-manifold steering).")


if __name__ == "__main__":
    main()
