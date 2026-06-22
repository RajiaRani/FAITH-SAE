"""step3_measure_cfs.py — MEASURE the three CFS components per steering method.

==============================================================================
WHAT THIS STEP DOES (in one sentence)
==============================================================================
For each of the five steering methods, it turns the steering knob through a
ladder of values, reads the concepts off the steered activations with the trained
probes, and MEASURES the three faithfulness components — Monotonicity (Spearman),
Specificity (off-target probe drift), Sufficiency (Cohen's-d effect size) — then
combines them into the Causal Faithfulness Score (CFS) with the project's
harmonic-mean cfs_score. NOTHING here is read from a lookup table: every number
is computed from the data. Measuring them IS the point of this milestone.

==============================================================================
TEACH-FROM-ZERO: the three components, each from scratch
==============================================================================

THE STEERING KNOB s
  "Steering" = reaching inside the frozen model and nudging an activation along a
  concept direction to make the concept stronger. The KNOB s is how hard we nudge.
  We sweep s = 0, 0.8, 1.6, 2.4, 3.2, 4.0 (n_knob_steps values up to s_max) and
  read the concept at each setting. SAME knob ladder for every method (matched
  strength) so the comparison is fair.

(1) MONOTONICITY = "does turning the knob UP move the readout UP, smoothly?"
  We want: knob up -> target readout up, in order, without zig-zags. We score the
  ORDER agreement with the SPEARMAN RANK CORRELATION.

  SPEARMAN FROM ZERO (a 3-point worked example)
    Spearman correlation = Pearson correlation computed on the RANKS (1st, 2nd,
    3rd...) of the values, not the raw values. It asks only "do they rise
    together in the same order?", ignoring by how much. It ranges -1..+1:
      +1 = perfectly same order (every step up in the knob is a step up in the
           readout), 0 = no order relation, -1 = perfectly reversed order.
    Example. Knobs = [0, 1, 2] (ranks 1,2,3).
      * readout [0.1, 0.5, 0.9] -> ranks [1,2,3]  -> Spearman = +1.0 (monotone up).
      * readout [0.1, 0.9, 0.5] -> ranks [1,3,2]  -> Spearman = +0.5 (one swap).
      * readout [0.9, 0.5, 0.1] -> ranks [3,2,1]  -> Spearman = -1.0 (monotone DOWN).
    We use scipy.stats.spearmanr and clip negatives to 0 (a steer that moves the
    concept the WRONG way is not faithful, scored 0).

(2) SPECIFICITY = "did ONLY the target move, while off-target concepts stayed flat?"
  We hold up each OFF-TARGET probe (the rulers from step2) at the lowest and the
  highest knob and measure how far each off-target read DRIFTED. We normalize that
  drift by how far the TARGET read moved (so it is a fraction), average over the
  off-target concepts, and report  specificity = 1 - (mean off-target drift /
  target move), clipped to [0,1].
    * 0 off-target drift -> specificity 1.0 (perfectly specific: only target moved).
    * off-target drift as big as the target's move -> specificity 0.0 (smears
      everywhere; an entangled / off-manifold edit).
  Tiny number: target read moved by 6.0; off-target reads drifted by 0.3 and 0.9
  -> mean drift 0.6 -> 0.6/6.0 = 0.10 -> specificity = 1 - 0.10 = 0.90.

(3) SUFFICIENCY = "was the effect BIG ENOUGH to matter?", as a standardized size
  COHEN'S d FROM ZERO
    Cohen's d = the gap between two group means measured in STANDARD-DEVIATION
    units:  d = (mean_after - mean_before) / pooled_standard_deviation.
    It answers "how many standard deviations did the readout move?" — a unit-free
    effect size, so it is comparable across concepts and backbones.
    Analogy: saying two towns differ in height by "1.5 std-devs" is more telling
    than "3 cm" — it accounts for how spread-out heights already are.
    Example: readout at knob 0 has mean 0.0, std 1.0; at full knob mean 4.0, std
    1.0 -> pooled std 1.0 -> d = (4.0 - 0.0)/1.0 = 4.0 (a HUGE, ample effect).
  We map d to [0,1] by  sufficiency = min(d / cohen_d_ample, 1.0)  (d ~ 4 std-devs
  apart counts as fully sufficient = 1.0).

CFS = HARMONIC MEAN of the three  (DESIGN_BRIEF §13/§14), reused from src.utils
  The harmonic mean is CONJUNCTIVE: if ANY one component is near zero, CFS is near
  zero — faithfulness needs ALL THREE at once (monotone AND specific AND
  sufficient). A plain average would let one strong axis hide a failure in
  another; the harmonic mean refuses to.
    Tiny number: HM(0.9,0.9,0.9)=0.90, but HM(0.9,0.9,0.05)=0.13 — one weak axis
    tanks the whole score. We call the project's cfs_score() so the EDA notebook
    and every milestone use exactly one scoring rule.

THE FIVE METHODS (names fixed by DESIGN_BRIEF §12 + the TCAV reference)
  * supervised_steer  — TCAV-style: steer along the TARGET PROBE's weight vector
    (a label-trained concept direction). The strong, label-expensive reference a
    good unsupervised SAE direction should approach. (TCAV in one line: Testing
    with Concept Activation Vectors = use a linear concept direction learned from
    labelled examples to probe/steer a concept; Kim et al. 2017.)
  * onmanifold_steer  — OURS: project the SAE edit onto the top-r real-image
    subspace U_r (a <- a + s*(P_M*d)). Realistic, decodable, specific.
  * clamp_steer       — clamp the SAE feature to magnitude s, no projection.
  * naive_steer       — off-manifold activation addition a <- a + s*d (the M3
    competitor the field admits is unreliable).
  * random_steer      — add a fixed RANDOM direction (null/sanity baseline).

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step3_measure_cfs.py
Reads outputs/concept_dirs.npy, probe_acts.npy, probe_labels.npy,
      sae_decoder.npy, probe_weights.npy, probe_bias.npy, U_r.npy  (steps 1-2).
Writes outputs/cfs_breakdown.csv  (variant, monotonicity, specificity,
      sufficiency, cfs).
"""
from __future__ import annotations

import numpy as np

from _common import banner, load_cfg, outpath


# --------------------------------------------------------------------------- #
# Readout: hold a probe-ruler up to a batch of activations -> one number each. #
# --------------------------------------------------------------------------- #
def probe_readout(acts: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Linear-probe score per item: w . a + b  (the ruler's raw reading).

    We use the raw linear score (not the squashed probability) because it is a
    smooth, unbounded reading — exactly what we want to watch rise as the knob
    turns and to take an effect size of. acts [B, dim] -> [B] scores.
    """
    return acts @ w + b


# --------------------------------------------------------------------------- #
# Apply one steering method to a batch of activations at knob value s.         #
# --------------------------------------------------------------------------- #
def steer_batch(variant, acts_t, edit_dir, strength, model, concept, U_r_t,
                supervised_dir):
    """Return the STEERED activations [B, dim] for one method at knob `strength`.

    Four variants come straight from the project's STEER_REGISTRY (we do NOT
    re-implement them). `supervised_steer` is the TCAV-style reference: it uses
    the naive activation-addition MECHANISM but along the supervised (target
    probe) direction instead of the SAE direction.
    """
    import torch

    from src.model import build_steer

    if variant == "supervised_steer":
        # TCAV-style: a <- a + s * (unit supervised direction). Same naive
        # mechanism, but the direction came from LABELS (the target probe).
        d = supervised_dir / (supervised_dir.norm() + 1e-8)
        return acts_t + strength * d

    steer = build_steer(variant, model.cfg)
    out = steer(acts_t, edit_dir, strength, sae=model.sae, concept=concept,
                basis=U_r_t)
    return out


# --------------------------------------------------------------------------- #
# The three MEASURED components for one method.                               #
# --------------------------------------------------------------------------- #
def measure_components(variant, cfg, acts, dirs, dec, W, b, U_r, model):
    """MEASURE (monotonicity, specificity, sufficiency) for one steering method.

    Every value is computed from the data: Spearman over the knob sweep,
    off-target probe drift, and a Cohen's-d effect size. Returns a dict.
    """
    import torch
    from scipy.stats import spearmanr

    tgt = int(cfg["target_concept"])
    n_c = int(cfg["n_concepts"])
    s_max = float(cfg["steer_strength"])
    n_steps = int(cfg["n_knob_steps"])
    d_ample = float(cfg["cohen_d_ample"])

    # Tensors the steerers operate on.
    acts_t = torch.from_numpy(acts).float()                # [N, dim]
    U_r_t = torch.from_numpy(U_r).float()                  # [dim, r]
    # The SAE edit direction = the SAE decoder column whose direction best matches
    # the planted TARGET concept (the feature the SAE discovered for the target).
    dec_t = torch.from_numpy(dec).float()                  # [dim, sae_dim]
    tgt_dir = torch.from_numpy(dirs[tgt]).float()          # planted target dir
    cos = (dec_t.T @ tgt_dir) / (dec_t.norm(dim=0) + 1e-8) / (tgt_dir.norm() + 1e-8)
    concept = int(cos.abs().argmax())                      # the matching SAE feature
    sign = float(torch.sign(cos[concept]))                 # align edit with +target
    edit_dir = dec_t[:, concept] * sign                    # raw SAE edit direction
    supervised_dir = torch.from_numpy(W[tgt]).float()      # TCAV-style direction

    # Probe rulers (raw linear score) for the target and the off-target concepts.
    w_tgt, b_tgt = W[tgt], float(b[tgt])
    off_idx = [c for c in range(n_c) if c != tgt]

    # ---- sweep the knob and read every concept at every setting -------------
    knobs = np.linspace(0.0, s_max, n_steps)
    tgt_means = []                       # mean target readout at each knob
    off_means = {c: [] for c in off_idx}  # mean off-target readout at each knob
    tgt_at_0 = tgt_at_max = None         # per-item readouts at the endpoints
    with torch.no_grad():
        for j, s in enumerate(knobs):
            a_s = steer_batch(variant, acts_t, edit_dir, float(s), model,
                              concept, U_r_t, supervised_dir).cpu().numpy()
            tgt_means.append(float(probe_readout(a_s, w_tgt, b_tgt).mean()))
            for c in off_idx:
                off_means[c].append(float(probe_readout(a_s, W[c], float(b[c])).mean()))
            if j == 0:
                tgt_at_0 = probe_readout(a_s, w_tgt, b_tgt)
            if j == n_steps - 1:
                tgt_at_max = probe_readout(a_s, w_tgt, b_tgt)

    tgt_means = np.asarray(tgt_means)

    # ---- (1) MONOTONICITY: Spearman(knob, target readout), negatives -> 0 ---
    rho, _ = spearmanr(knobs, tgt_means)
    if np.isnan(rho):                    # flat readout (no movement) -> no order
        rho = 0.0
    monotonicity = float(max(rho, 0.0))

    # ---- (2) SPECIFICITY: 1 - normalized off-target drift -------------------
    tgt_move = abs(tgt_means.max() - tgt_means.min()) + 1e-6
    drifts = []
    for c in off_idx:
        arr = np.asarray(off_means[c])
        drifts.append(abs(arr.max() - arr.min()))   # how far this off-target moved
    mean_drift = float(np.mean(drifts))
    specificity = float(np.clip(1.0 - mean_drift / tgt_move, 0.0, 1.0))

    # ---- (3) SUFFICIENCY: Cohen's-d effect size at full knob, mapped to [0,1]
    r0 = np.asarray(tgt_at_0); r1 = np.asarray(tgt_at_max)
    pooled = (r0.std() + r1.std()) / 2.0 + 1e-6
    cohen_d = float(abs(r1.mean() - r0.mean()) / pooled)
    sufficiency = float(min(cohen_d / d_ample, 1.0))

    return {
        "variant": variant,
        "monotonicity": round(monotonicity, 4),
        "specificity": round(specificity, 4),
        "sufficiency": round(sufficiency, 4),
        "_cohen_d": round(cohen_d, 3),          # diagnostic (not a CFS column)
        "_spearman_raw": round(float(rho), 3),  # diagnostic (pre-clip)
    }


def worked_spearman_example() -> None:
    """Print the 3-point Spearman example from the docstring (check by hand)."""
    from scipy.stats import spearmanr
    banner("TINY WORKED EXAMPLE: Spearman scores the ORDER, not the size")
    knobs = [0, 1, 2]
    cases = {
        "monotone up   readout [0.1,0.5,0.9]": [0.1, 0.5, 0.9],
        "one swap      readout [0.1,0.9,0.5]": [0.1, 0.9, 0.5],
        "monotone down readout [0.9,0.5,0.1]": [0.9, 0.5, 0.1],
    }
    print(f"  knobs = {knobs}  (ranks 1,2,3)")
    for name, r in cases.items():
        rho, _ = spearmanr(knobs, r)
        print(f"    {name}  -> Spearman = {rho:+.2f}")
    print("  >>> MONOTONICITY clips negatives to 0: a steer that moves the concept")
    print("      the WRONG way is not faithful, so it scores 0, not -1.")


def main() -> None:
    cfg = load_cfg()
    banner("STEP 3 — MEASURE monotonicity / specificity / sufficiency -> CFS")

    import torch

    from src.model import make_model
    from src.utils import cfs_score, set_seed

    set_seed(int(cfg["seed"]))

    # Load everything the earlier steps produced.
    acts = np.load(outpath("probe_acts.npy"))
    dirs = np.load(outpath("concept_dirs.npy"))
    dec = np.load(outpath("sae_decoder.npy"))
    W = np.load(outpath("probe_weights.npy"))
    b = np.load(outpath("probe_bias.npy"))
    U_r = np.load(outpath("U_r.npy"))
    print(f"  loaded acts {acts.shape}, concept_dirs {dirs.shape}, "
          f"sae_decoder {dec.shape}")
    print(f"  loaded probes W {W.shape}, U_r {U_r.shape} "
          f"(rerun step1/step2 if these are missing)")

    # A model whose SAE decoder we OVERWRITE with the trained decoder, so the
    # registry steerers (clamp_steer, onmanifold_steer) act on the REAL features.
    model = make_model(cfg)
    with torch.no_grad():
        model.sae.dec.weight.copy_(torch.from_numpy(dec).float())

    worked_spearman_example()

    banner("MEASURED components per steering method (computed, not looked up)")
    print(f"  {'variant':<18} {'mono':>6} {'spec':>6} {'suff':>6} "
          f"{'CFS':>7}   (cohen_d, raw_rho)")
    print("  " + "-" * 70)

    rows = []
    for variant in cfg["variants"]:
        comp = measure_components(variant, cfg, acts, dirs, dec, W, b, U_r, model)
        comp["cfs"] = round(cfs_score(comp["monotonicity"], comp["specificity"],
                                      comp["sufficiency"]), 4)
        rows.append(comp)
        print(f"  {variant:<18} {comp['monotonicity']:>6.3f} "
              f"{comp['specificity']:>6.3f} {comp['sufficiency']:>6.3f} "
              f"{comp['cfs']:>7.4f}   (d={comp['_cohen_d']}, rho={comp['_spearman_raw']})")

    # Write the headline table with pandas (the four CFS columns + cfs).
    import pandas as pd
    df = pd.DataFrame(rows)[["variant", "monotonicity", "specificity",
                             "sufficiency", "cfs"]]
    out = outpath("cfs_breakdown.csv")
    df.to_csv(out, index=False)
    print(f"\n  saved -> {out}")

    # Sanity invariants the contract requires.
    all_in_range = bool(((df["cfs"] >= 0.0) & (df["cfs"] <= 1.0)).all())
    onm = float(df.loc[df.variant == "onmanifold_steer", "cfs"].iloc[0])
    naive = float(df.loc[df.variant == "naive_steer", "cfs"].iloc[0])
    rnd = float(df.loc[df.variant == "random_steer", "cfs"].iloc[0])
    ranked = df.sort_values("cfs", ascending=False)["variant"].tolist()
    onm_top2 = "onmanifold_steer" in ranked[:2]   # among the MOST faithful
    print(f"\n  SUCCESS CRITERIA:")
    print(f"    all CFS in [0,1]                         -> "
          f"{'PASS' if all_in_range else 'FAIL'}")
    print(f"    on-manifold ({onm:.3f}) > naive ({naive:.3f}) -> "
          f"{'PASS' if onm > naive else 'FAIL'}")
    print(f"    on-manifold among the 2 MOST faithful    -> "
          f"{'PASS' if onm_top2 else 'FAIL'}  (rank order: {ranked})")
    print(f"    random ({rnd:.3f}) is the floor             -> "
          f"{'PASS' if rnd <= min(onm, naive) else 'FAIL'}")
    print("\nSTEP 3 done. Next: step4 draws the grouped-bar figure.")


# REAL RUN (M5): the exact same measurement on REAL CLIP ViT-B/16 activations.
# Swap the synthetic bank/probes for real CLIP activations + real concept probes;
# the knob sweep, Spearman monotonicity, off-target probe specificity, and
# Cohen's-d sufficiency are computed identically. Then REPEAT the whole step at
# every OOD shift level (clean -> ImageNet-R -> Sketch -> ImageNet-C 1..5 ->
# ObjectNet) to get the CFS-vs-shift curve — that loop is milestone 6.
if __name__ == "__main__":
    main()
