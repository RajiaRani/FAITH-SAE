"""step2_a2_topk_k.py — ABLATION A2: TopK sparsity level k.

==============================================================================
WHAT THIS ABLATION DOES (in one sentence)
==============================================================================
It turns exactly ONE knob — the TopK sparsity level `k` (how many SAE features
may be ON at once) — across a sweep, holds every other dial fixed, retrains the
SAE per k, and MEASURES how CFS and reconstruction quality respond.

==============================================================================
TEACH-FROM-ZERO: the A2 knob — the value of k
==============================================================================
k (THE SPARSITY LEVEL)
  In a TopK SAE, after the encoder we KEEP only the k biggest feature values per
  input and zero the rest. So k = the maximum number of "concept switches" allowed
  ON at the same time. Small k = very sparse (few switches); large k = dense (many
  switches).
  Analogy: a packing limit. k=1 = "pack exactly ONE item"; k=32 = "pack up to 32".
  With only 1 item you must choose the single most important thing (very
  selective); with 32 you bring a cluttered bag.
  Tiny number: dictionary size 128. k=1 keeps 1/128 features on; k=32 keeps up to
  32/128 on. The SAE's job (rebuild the activation) is easier with more features
  but each feature is less forced to be a clean single concept.

OVER- vs UNDER-SPARSITY (the two ways k can be wrong)
  * UNDER-sparse (k TOO BIG): too many features on -> features share the work, each
    one smears across several concepts (polysemantic), the steering direction is
    muddier -> specificity drops -> CFS can sag. Reconstruction is GOOD (lots of
    capacity) but interpretability/faithfulness suffers.
  * OVER-sparse (k TOO SMALL): too few features on -> the SAE can't represent the
    activation well, the target concept may not even get its own clean feature ->
    the steering direction is starved -> the effect weakens. Reconstruction is BAD.
  * "SWEET SPOT": a middle k where each concept gets a clean dedicated feature AND
    reconstruction is still decent -> CFS peaks. A2 LOCATES that sweet spot.
  Tiny number: CFS by k might read 0.55 (k=1), 0.78 (k=4), 0.81 (k=8), 0.79 (k=16),
  0.72 (k=32) -> a gentle hump peaking near k=8.

HOW TO READ "CFS vs knob" TO FIND A SWEET SPOT
  Plot CFS on the y-axis against the knob on the x-axis. A SWEET SPOT is the peak
  of a hump (best faithfulness); a KNEE is an elbow where the curve bends sharply
  (returns stop improving). You pick the knob value at the peak/knee — the smallest
  setting that buys you the most faithfulness.

THE DIAGNOSTIC WE REPORT ALONGSIDE CFS
  Reconstruction MSE again (lower = the SAE rebuilt activations better). Watching
  CFS and recon-MSE TOGETHER across k shows the trade-off directly: recon-MSE keeps
  falling as k rises, but CFS humps and then declines — the sweet spot is where
  faithfulness is best, not where reconstruction is best.

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step2_a2_topk_k.py
Appends rows to outputs/ablations.csv (ablation_id=A2) — one row per (k, steerer).
"""
from __future__ import annotations

from _common import (banner, build_labelled_bank, estimate_U_r, load_cfg,
                     measure_cfs, train_probes, train_sae_decoder)


def run(cfg: dict) -> list:
    rows = []
    acts, labels, dirs, _ = build_labelled_bank(cfg)
    W, b, accs = train_probes(acts, labels, seed=int(cfg["seed"]))
    U_r = estimate_U_r(acts, int(cfg["manifold_rank"]))
    print(f"  probe held-out accuracies = {[round(a, 3) for a in accs]}")
    print(f"  U_r shape = {U_r.shape} (fixed on-manifold sheet)\n")

    print(f"  {'k':>4} {'variant':<18} {'recon_MSE':>10} "
          f"{'mono':>6} {'spec':>6} {'suff':>6} {'CFS':>7}")
    print("  " + "-" * 66)
    for k in cfg["a2_topk_ks"]:
        # === THE ONE KNOB WE TURN: topk_k (sae_type stays 'topk', all else fixed) ==
        dec, recon_mse, model = train_sae_decoder(
            {"sae_type": "topk", "topk_k": int(k)}, acts, cfg)
        for variant in cfg["ablation_variants"]:
            m = measure_cfs(variant, cfg, acts, dirs, dec, W, b, U_r, model,
                            target_concept=int(cfg["target_concept"]))
            print(f"  {k:>4} {variant:<18} {recon_mse:>10.4f} "
                  f"{m['monotonicity']:>6.3f} {m['specificity']:>6.3f} "
                  f"{m['sufficiency']:>6.3f} {m['cfs']:>7.4f}")
            rows.append({
                "ablation_id": "A2",
                "knob_value": int(k),
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
    banner("ABLATION A2 — TopK k (sparsity): the interpretability/faithfulness trade")
    rows = run(cfg)
    onm = [r for r in rows if r["variant"] == "onmanifold_steer"]
    best = max(onm, key=lambda r: r["cfs"])
    print(f"\n  A2 takeaway: on-manifold CFS peaks at k = {best['knob_value']} "
          f"(CFS = {best['cfs']:.3f}) -> the sparsity SWEET SPOT.")
    print("    too small k = effect starved; too large k = polysemantic blur.")
    return rows


# REAL RUN (M7): sweep k on a TopK SAE trained over REAL CLIP activations; the
# diagnostic stays reconstruction MSE (and you can add the interpretability score
# used in A4). Same measuring rig; only the activations change.
if __name__ == "__main__":
    main()
