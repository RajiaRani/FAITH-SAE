"""step4_a4_select_threshold.py — ABLATION A4: concept-selection interpretability threshold.

==============================================================================
WHAT THIS ABLATION DOES (in one sentence)
==============================================================================
It turns exactly ONE knob — how STRICT the "is this a well-defined concept?" filter
is (an interpretability threshold) — across a sweep, holds every other dial fixed,
and MEASURES the RELIABLE-concept fraction that survives each threshold and the
mean CFS of the survivors.

==============================================================================
TEACH-FROM-ZERO: the A4 knob — a selection threshold
==============================================================================
THE PROBLEM: not every SAE feature is a clean concept
  An SAE discovers THOUSANDS of features. Many are POLYSEMANTIC (one feature mixes
  several concepts) or junk (no clean meaning). The field's finding: only roughly
  10-15% of raw SAE features are "well-defined" enough to steer reliably. So before
  steering, you SELECT the good ones with a filter.

INTERPRETABILITY SCORE (per concept, in [0,1])
  A number saying how cleanly a concept corresponds to ONE thing (monosemantic) vs
  blurs into others (polysemantic). Here we use a concrete, computable proxy:
  DISTINCTNESS = 1 - (max |cosine| overlap of this concept's direction with any
  OTHER concept's direction). A clean, well-defined concept points its own way
  (low overlap -> distinctness near 1); a polysemantic one shares its direction
  with neighbours (high overlap -> distinctness near 0). Every number is computed
  from the concept directions the probes and steerers actually use. (Real SAE
  pipelines use auto-interp / activation-coherence scores; this direction-overlap
  proxy plays the same "how monosemantic is this feature?" role offline.)
  Tiny number: a concept that overlaps another at cosine 0.80 -> distinctness 0.20
  (polysemantic, filtered out by a strict threshold); overlap 0.03 -> 0.97 (clean).

SELECTION THRESHOLD (the A4 knob)
  A cutoff τ: KEEP only concepts whose interpretability score ≥ τ; DROP the rest.
  Analogy: a bouncer's height line. Raise the line (bigger τ) and fewer people get
  in, but everyone inside is taller (more reliable). Lower it and a crowd gets in,
  some of them unreliable.
  * τ = 0.0  -> keep EVERYTHING (even junk features) -> the kept set's mean CFS is
    dragged down by the unreliable ones.
  * τ moderate -> keep only the clean concepts -> the kept set's mean CFS RISES,
    but the kept FRACTION falls toward the field's ~10-15% reliable tail.
  * τ too high -> you may keep almost nothing (fraction -> 0); the survivors are
    pristine but you've thrown away usable concepts.
  Tiny number: at τ=0.0 keep 100% of concepts, mean CFS 0.60; at τ=0.6 keep 25%,
  mean CFS 0.84. Stricter filter -> fewer but more faithful concepts.

THE TRADE-OFF A4 MAKES VISIBLE
  As τ rises: reliable-concept FRACTION falls (fewer survive) while mean CFS of the
  survivors RISES (the keepers are cleaner). A4 plots both so you can pick a τ that
  keeps a usable fraction at an acceptable faithfulness — the "reliable-concept"
  operating point.

THE DIAGNOSTIC WE REPORT ALONGSIDE CFS
  Reliable-concept FRACTION = (#concepts kept) / (#concepts total) at each τ.
  This is A4's headline diagnostic: it is exactly the field's "~10-15% reliable
  tail" curve, measured here on planted concepts of varying cleanliness.

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step4_a4_select_threshold.py
Appends rows to outputs/ablations.csv (ablation_id=A4) — one row per (threshold,
steerer): cfs = mean CFS of SURVIVING concepts, diagnostic = kept fraction.
"""
from __future__ import annotations

import numpy as np

from _common import (banner, build_labelled_bank, estimate_U_r, load_cfg,
                     measure_cfs, train_probes, train_sae_decoder)


def _interp_scores(dirs) -> np.ndarray:
    """Per-concept interpretability = DISTINCTNESS in [0,1] (a monosemanticity score).

    distinctness[c] = 1 - max_{k != c} |cosine(dir_c, dir_k)|.
    A monosemantic concept points its own way (low overlap -> ~1); a polysemantic
    one shares its direction with a neighbour (high overlap -> ~0). Computed
    straight from the concept directions; no lookup table.
    """
    D = np.asarray(dirs, dtype=float)
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    G = np.abs(D @ D.T)                              # |cosine| overlaps
    np.fill_diagonal(G, 0.0)
    return np.clip(1.0 - G.max(axis=1), 0.0, 1.0)


def run(cfg: dict) -> list:
    rows = []
    acts, labels, dirs, _ = build_labelled_bank(cfg)
    W, b, accs = train_probes(acts, labels, seed=int(cfg["seed"]))
    U_r = estimate_U_r(acts, int(cfg["manifold_rank"]))
    dec, recon_mse, model = train_sae_decoder({}, acts, cfg)
    n_c = int(cfg["n_concepts"])

    # Interpretability score per concept (the "how monosemantic is this?" number).
    scores = _interp_scores(dirs)
    print(f"  per-concept probe accuracy   = {[round(a, 3) for a in accs]}")
    print(f"  per-concept interpretability = {[round(float(s), 3) for s in scores]} "
          f"(distinctness; higher = cleaner concept)")

    # MEASURE CFS once per concept (treat each planted concept as the steer target).
    # A4 then FILTERS this fixed per-concept CFS list by the threshold knob.
    per_concept_cfs = {}
    for variant in cfg["ablation_variants"]:
        per_concept_cfs[variant] = []
        for c in range(n_c):
            m = measure_cfs(variant, cfg, acts, dirs, dec, W, b, U_r, model,
                            target_concept=c)
            per_concept_cfs[variant].append(m["cfs"])
    print(f"  per-concept CFS (on-manifold) = "
          f"{[round(x, 3) for x in per_concept_cfs['onmanifold_steer']]}\n")

    print(f"  {'tau':>5} {'variant':<18} {'kept_frac':>9} {'mean_CFS':>9}")
    print("  " + "-" * 46)
    for tau in cfg["a4_select_thresholds"]:
        # === THE ONE KNOB WE TURN: the selection threshold tau ===================
        kept = [c for c in range(n_c) if scores[c] >= float(tau)]
        kept_frac = len(kept) / max(n_c, 1)
        for variant in cfg["ablation_variants"]:
            cfs_list = per_concept_cfs[variant]
            mean_cfs = float(np.mean([cfs_list[c] for c in kept])) if kept else 0.0
            print(f"  {tau:>5.2f} {variant:<18} {kept_frac:>9.3f} {mean_cfs:>9.4f}")
            rows.append({
                "ablation_id": "A4",
                "knob_value": float(tau),
                "variant": variant,
                "cfs": round(mean_cfs, 4),
                "diagnostic": round(kept_frac, 4),
                "diagnostic_name": "reliable_concept_fraction",
                # the three components are concept-set means; left blank here
                # (A4's unit is the SELECTED SET, not a single concept).
                "monotonicity": "",
                "specificity": "",
                "sufficiency": "",
                "offmanifold_residual": "",
            })
    return rows


def main() -> list:
    cfg = load_cfg()
    banner("ABLATION A4 — selection threshold: the reliable-concept fraction (~10-15% tail)")
    rows = run(cfg)
    onm = [r for r in rows if r["variant"] == "onmanifold_steer"]
    lo = min(onm, key=lambda r: r["knob_value"])
    hi = max(onm, key=lambda r: r["knob_value"])
    print(f"\n  A4 takeaway: raising tau {lo['knob_value']:.1f}->{hi['knob_value']:.1f} "
          f"moves kept-fraction {lo['diagnostic']:.2f}->{hi['diagnostic']:.2f} and "
          f"mean CFS {lo['cfs']:.3f}->{hi['cfs']:.3f}.")
    print("    stricter filter => fewer but more faithful concepts (the reliable tail).")
    return rows


# REAL RUN (M7): replace the probe-accuracy proxy with the SAE auto-interp /
# activation-coherence score per feature over real CLIP activations; sweep tau,
# report the reliable-feature fraction (expect the ~10-15% tail) and the mean CFS
# of the surviving features. Same filter-then-average logic.
if __name__ == "__main__":
    main()
