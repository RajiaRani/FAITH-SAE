"""step5_a5_layer_token.py — ABLATION A5: backbone layer & patch-vs-CLS token.

==============================================================================
WHAT THIS ABLATION DOES (in one sentence)
==============================================================================
It turns exactly ONE knob at a time — WHICH backbone layer (early vs late) and
WHICH token (patch vs CLS) the SAE reads — across the four combinations, holds
every other dial fixed, and MEASURES how CFS responds to each choice.

==============================================================================
TEACH-FROM-ZERO: the A5 knob — layer & token
==============================================================================
THE BACKBONE (one paragraph)
  A Vision Transformer (ViT) processes an image in a STACK of layers. It first
  cuts the image into a grid of small square PATCHES, turns each into a vector
  (a "token"), and also keeps one extra special token called CLS that pools a
  whole-image summary. As the image flows UP the stack, the tokens' activations
  change: early layers carry low-level texture/edges; late layers carry abstract,
  concept-level meaning ("dog", "wheel").

WHICH LAYER (the first part of the knob)
  We can attach the SAE to an EARLY layer or a LATE layer.
  * EARLY layer: activations are lower-level and less concept-aligned — the
    concept "sheet" is THICKER and noisier, concept directions are less clean.
  * LATE layer: activations are concept-aligned — a CLEANER, lower-dimensional
    sheet where a concept gets a crisp direction, so steering tends to be more
    faithful.
  Analogy: reading a draft vs a final copy. The early-layer draft has the idea but
  smudged; the late-layer final copy states it cleanly.
  Tiny number (our synthetic stand-in): "early" = a THICKER planted sheet
  (true rank 16, blurrier concepts); "late" = a CLEAN planted sheet
  (true rank 8, crisp concepts). Late should score higher CFS.

WHICH TOKEN: PATCH vs CLS (the second part of the knob)
  * PATCH tokens: one activation per image patch (here 16 patches). The SAE sees
    many fine-grained, per-region notes — more signal, localized concepts.
  * CLS token: a SINGLE pooled vector per image (the whole-image summary). Fewer,
    coarser notes — global concepts only, and a noisier readout because everything
    is squeezed into one vector.
  Analogy: photographing each room of a house (patch) vs one blurry photo of the
  whole house from the street (CLS). The per-room shots resolve more detail.
  Tiny number (our synthetic stand-in): "patch" reads the per-item activation
  directly (full signal); "cls" POOLS extra background variation into the readout
  (a noisier ruler), so CLS tends to score a notch lower CFS than patch.

WHY A5 MATTERS
  Where you attach the SAE is a DESIGN choice made before any steering. A5 checks
  whether that choice changes faithfulness — and the expected ordering
  (late+patch best, early+cls worst) tells you where to put the SAE.

THE DIAGNOSTIC WE REPORT ALONGSIDE CFS
  Reconstruction MSE of the SAE on that layer/token's bank (lower = the SAE fit
  those activations better). It tends to be lower (better) on the cleaner late
  layer — a second signal that late is the friendlier place to attach the SAE.

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step5_a5_layer_token.py
Appends rows to outputs/ablations.csv (ablation_id=A5) — one row per
(layer-token, steerer). knob_value reads like "late|patch".
"""
from __future__ import annotations

import numpy as np

from _common import (banner, build_labelled_bank, estimate_U_r, load_cfg,
                     measure_cfs, train_probes, train_sae_decoder)


def _bank_for(cfg: dict, layer: str, token: str):
    """Build the labelled bank for a (layer, token) choice — the A5 stand-ins.

    LAYER stand-in: an "early" layer = a THICKER, blurrier concept sheet (we plant
    a higher true_manifold_rank, so concepts are less cleanly separated); a "late"
    layer = the clean baseline sheet. TOKEN stand-in: "patch" reads the activation
    directly; "cls" adds extra pooled background variation to every activation (a
    single squeezed whole-image vector is a noisier ruler). Everything else (dim,
    concept_strength, SAE budget, knob sweep) is the fixed baseline.
    """
    # different seed per layer so the two "layers" are genuinely different sheets
    seed_extra = 0 if layer == "late" else 100
    true_r = int(cfg["true_manifold_rank"]) if layer == "late" \
        else 2 * int(cfg["true_manifold_rank"])     # early = thicker sheet
    acts, labels, dirs, _ = build_labelled_bank(
        cfg, seed_extra=seed_extra, manifold_rank_override=true_r)
    if token == "cls":
        # CLS stand-in: squeeze in extra pooled background variation -> noisier readout.
        rng = np.random.default_rng(int(cfg["seed"]) + seed_extra + 777)
        acts = acts + 0.6 * float(cfg["noise_off_manifold"]) * \
            rng.standard_normal(acts.shape).astype(np.float32)
    return acts.astype(np.float32), labels, dirs


def run(cfg: dict) -> list:
    rows = []
    print(f"  {'layer|token':<14} {'variant':<18} {'recon_MSE':>10} "
          f"{'mono':>6} {'spec':>6} {'suff':>6} {'CFS':>7}")
    print("  " + "-" * 72)
    for choice in cfg["a5_layer_tokens"]:
        layer, token = choice["layer"], choice["token"]
        tag = f"{layer}|{token}"
        # === THE ONE KNOB WE TURN: the (layer, token) attachment point ===========
        acts, labels, dirs = _bank_for(cfg, layer, token)
        W, b, _ = train_probes(acts, labels, seed=int(cfg["seed"]))
        U_r = estimate_U_r(acts, int(cfg["manifold_rank"]))
        dec, recon_mse, model = train_sae_decoder({}, acts, cfg)
        for variant in cfg["ablation_variants"]:
            m = measure_cfs(variant, cfg, acts, dirs, dec, W, b, U_r, model,
                            target_concept=int(cfg["target_concept"]))
            print(f"  {tag:<14} {variant:<18} {recon_mse:>10.4f} "
                  f"{m['monotonicity']:>6.3f} {m['specificity']:>6.3f} "
                  f"{m['sufficiency']:>6.3f} {m['cfs']:>7.4f}")
            rows.append({
                "ablation_id": "A5",
                "knob_value": tag,
                "variant": variant,
                "cfs": m["cfs"],
                "diagnostic": round(recon_mse, 4),
                "diagnostic_name": "recon_mse",
                "monotonicity": m["monotonicity"],
                "specificity": m["specificity"],
                "sufficiency": m["sufficiency"],
                "offmanifold_residual": m["offmanifold_residual"],
            })
    return rows


def main() -> list:
    cfg = load_cfg()
    banner("ABLATION A5 — backbone layer (early/late) & token (patch/CLS)")
    rows = run(cfg)
    onm = [r for r in rows if r["variant"] == "onmanifold_steer"]
    best = max(onm, key=lambda r: r["cfs"])
    worst = min(onm, key=lambda r: r["cfs"])
    print(f"\n  A5 takeaway: best attachment = {best['knob_value']} "
          f"(CFS {best['cfs']:.3f}); worst = {worst['knob_value']} "
          f"(CFS {worst['cfs']:.3f}).")
    print("    cleaner LATE layer + per-PATCH tokens => more faithful steering.")
    return rows


# REAL RUN (M7): read REAL CLIP ViT-B/16 activations from an early vs late block,
# and from the patch-token grid vs the CLS token; train the SAE on each and report
# CFS per (layer, token). The synthetic thicker-sheet / noisier-CLS stand-ins are
# replaced by the real layer/token activations; the measuring rig is unchanged.
if __name__ == "__main__":
    main()
