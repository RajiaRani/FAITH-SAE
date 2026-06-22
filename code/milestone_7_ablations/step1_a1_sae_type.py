"""step1_a1_sae_type.py — ABLATION A1: SAE type (TopK vs L1).

==============================================================================
WHAT THIS ABLATION DOES (in one sentence)
==============================================================================
It turns exactly ONE knob — the SAE's sparsity recipe, `sae_type` ∈ {topk, l1} —
holds every other dial fixed, retrains the SAE for each setting, and MEASURES how
the Causal Faithfulness Score (CFS) and the reconstruction quality respond.

==============================================================================
WHAT IS AN ABLATION? (teach-from-zero)
==============================================================================
ABLATION
  Turn ONE knob, freeze everything else, and measure what changes. It is the
  scientific "controlled experiment": if you change two things at once and the
  result moves, you can't tell WHICH caused it. Change exactly one and any change
  in the outcome must be due to that one knob — so the ablation finds the CAUSE.
  Analogy: a recipe. To learn what the salt does, you cook the dish twice — once
  with salt, once without — keeping the oven, time, and every other ingredient
  identical. The taste difference is the salt's effect, full stop.
  Tiny number: baseline CFS = 0.80 with sae_type=topk. Switch ONLY sae_type to l1
  -> CFS = 0.74. The 0.06 drop is attributable to the SAE type, nothing else.

CONFOUND (why "hold all else fixed" matters)
  A confound is a SECOND thing that changed at the same time and could explain the
  result instead. If, while switching topk->l1, you ALSO doubled the bank size,
  a CFS change could be the bank, not the SAE type — a confound. The fixed
  baseline below removes confounds: only `sae_type` differs between the two runs.

==============================================================================
TEACH-FROM-ZERO: the A1 knob — TopK vs L1 sparsity
==============================================================================
SPARSE AUTOENCODER (SAE)
  A small network that re-expresses one activation as a SHORT list of "concept
  switches" (features), only a FEW of which are ON for any given input, then
  rebuilds the activation from them. "Sparse" = few switches on at once. The
  decoder columns are the concept DIRECTIONS we steer.

THE TWO WAYS TO ENFORCE SPARSITY (the A1 knob)
  * TopK SAE: after the encoder, KEEP only the k largest feature values per item
    and zero the rest — a HARD cap of exactly k switches on (Gao et al. 2024).
    Tiny number: k=8 means at most 8 of the 128 features are ever ON.
  * L1 SAE: do NOT cap the count; instead ADD a penalty l1_coeff * mean(|features|)
    to the training loss, which gently pushes most feature values toward zero — a
    SOFT sparsity (the classic "vanilla" SAE; Cunningham et al. 2023). The number
    of ON features is not fixed; it is whatever the penalty settles on.
    Analogy: TopK is "you may bring exactly 8 items through customs"; L1 is "you
    pay a tax per item, so you naturally bring few".

WHY IT MIGHT CHANGE CFS
  TopK's hard cap tends to give cleaner, higher-magnitude features (a sharper
  concept direction), which can steer more faithfully. L1's soft penalty can blur
  feature magnitudes and leave a longer reconstruction tail. A1 measures whether
  that intuition shows up as a CFS / reconstruction-MSE difference.

THE DIAGNOSTIC WE REPORT ALONGSIDE CFS
  Reconstruction MSE = how well the SAE rebuilds the original activation from its
  features (mean squared error; lower = the SAE captured the activation better).
  A1's relevant diagnostic, because the SAE TYPE most directly trades sparsity
  against reconstruction.

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step1_a1_sae_type.py
Appends rows to outputs/ablations.csv (ablation_id=A1) — one row per
(sae_type, steerer): ablation_id, knob_value, variant, cfs, diagnostic(=recon MSE).
"""
from __future__ import annotations

from _common import (banner, build_labelled_bank, estimate_U_r, load_cfg,
                     measure_cfs, train_probes, train_sae_decoder)


def run(cfg: dict) -> list:
    """Sweep sae_type and measure CFS + reconstruction MSE for each steerer."""
    rows = []
    # The SHARED measuring rig (built ONCE; identical for every knob value).
    acts, labels, dirs, _ = build_labelled_bank(cfg)
    W, b, accs = train_probes(acts, labels, seed=int(cfg["seed"]))
    U_r = estimate_U_r(acts, int(cfg["manifold_rank"]))
    print(f"  probe held-out accuracies per concept = "
          f"{[round(a, 3) for a in accs]} (target=concept {cfg['target_concept']})")
    print(f"  U_r shape = {U_r.shape} (the fixed on-manifold sheet, r={cfg['manifold_rank']})\n")

    print(f"  {'sae_type':<8} {'variant':<18} {'recon_MSE':>10} "
          f"{'mono':>6} {'spec':>6} {'suff':>6} {'CFS':>7}")
    print("  " + "-" * 70)
    for sae_type in cfg["a1_sae_types"]:
        # === THE ONE KNOB WE TURN: sae_type (everything else stays baseline) ===
        # Retrain the SAE with this sparsity recipe; the L1 variant lives in
        # _common.train_sae_decoder (a few clearly-commented lines).
        dec, recon_mse, model = train_sae_decoder(
            {"sae_type": sae_type}, acts, cfg)
        for variant in cfg["ablation_variants"]:
            m = measure_cfs(variant, cfg, acts, dirs, dec, W, b, U_r, model,
                            target_concept=int(cfg["target_concept"]))
            print(f"  {sae_type:<8} {variant:<18} {recon_mse:>10.4f} "
                  f"{m['monotonicity']:>6.3f} {m['specificity']:>6.3f} "
                  f"{m['sufficiency']:>6.3f} {m['cfs']:>7.4f}")
            rows.append({
                "ablation_id": "A1",
                "knob_value": sae_type,
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
    banner("ABLATION A1 — SAE type: TopK vs L1 (clean magnitudes & CFS)")
    rows = run(cfg)
    # Teaching takeaway: compare on-manifold CFS for topk vs l1.
    onm = [r for r in rows if r["variant"] == "onmanifold_steer"]
    by_type = {r["knob_value"]: r["cfs"] for r in onm}
    print(f"\n  A1 takeaway (on-manifold CFS): {by_type}")
    print("    -> the SAE TYPE shifts faithfulness; this is the ONLY changed knob.")
    return rows


# REAL RUN (M7): swap the synthetic bank for REAL CLIP ViT-B/16 patch activations
# over ImageNet-val, and fit BOTH SAE families (TopK and a true vanilla-L1 SAE) on
# them; report clean-CFS and reconstruction-MSE per family. The L1 override in
# _common.train_sae_decoder becomes a full L1-SAE training config. Everything
# downstream (probes, U_r, measure_cfs) is identical.
if __name__ == "__main__":
    main()
