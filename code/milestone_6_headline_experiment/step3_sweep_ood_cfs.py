"""step3_sweep_ood_cfs.py — MEASURE CFS at every OOD rung for both methods.

==============================================================================
WHAT THIS STEP DOES (in one sentence)
==============================================================================
For each rung of the OOD shift ladder (clean -> ImageNet-R -> Sketch -> C-1..5 ->
ObjectNet) it CORRUPTS the clean activations to that severity, then -- for BOTH
on-manifold and naive steering and for EVERY concept -- it MEASURES the Causal
Faithfulness Score (CFS) by sweeping the steering knob and reading the probes;
it bootstraps a confidence interval over concepts, finds the COLLAPSE KNEE, and
writes the headline table outputs/ood_cfs_sweep.csv. NOTHING is looked up: every
CFS is computed from the corrupted data. Measuring how CFS falls IS the paper's
answer to RQ3.

==============================================================================
TEACH-FROM-ZERO: every new term, defined before it is used
==============================================================================

DISTRIBUTION SHIFT (and the kinds on the ladder)
  "Distribution shift" = the test images differ from the clean images the model
  was built on. Three flavours we simulate, named after the real benchmarks:
    * RENDITION shift (ImageNet-R): an art/cartoon/sculpture of the object -- same
      thing, very different "look". (Real paper: Hendrycks 2020.)
    * SKETCH / texture removal (ImageNet-Sketch): a pencil drawing -- the shape is
      there but the texture/colour cues are gone. (Wang 2019.)
    * CORRUPTION (ImageNet-C): the SAME photo, degraded -- blur, noise, fog, JPEG.
      It comes with a SEVERITY DIAL 1..5 (1 = barely corrupted, 5 = wrecked).
      (Hendrycks & Dietterich 2019.) Plus ObjectNet: real photos in weird poses /
      backgrounds (Barbu 2019) -- the hardest real-world shift.
  COVARIATE SHIFT is the precise name for "the inputs x changed but the meaning of
  the concept didn't" -- a dog is still a dog in a sketch; only its pixels (and so
  its activations) moved. That is exactly our setting.

HOW WE SIMULATE A RUNG OFFLINE (no datasets, no GPU)
  Each rung corrupts every clean activation a with TWO growing pushes (config):
    a_shift = a  +  style * (Off @ a)  +  gauss * noise
    * gauss * noise : isotropic Gaussian noise across all 64 dirs (the ImageNet-C
      corruption analog -- random degradation).
    * style * (Off @ a) : a fixed operator `Off` that rotates each activation
      partly toward OFF-sheet directions (the rendition/sketch analog -- same
      object, new "look", landing OFF the clean sheet the model learned on photos).
  As `style` and `gauss` climb rung by rung, the activations leave the clean sheet.

WHY A STEER FAITHFUL ON CLEAN CAN COLLAPSE UNDER SHIFT (the headline mechanism)
  on-manifold steering projects the edit onto U_r -- but U_r was FROZEN on CLEAN
  (step2). Once the shifted activations have drifted OFF the clean sheet, U_r
  describes the WRONG sheet for them: the projection P_M = U_r U_r^T now points the
  edit at where the clean manifold USED to be, not where the shifted activations
  actually are. The probe (also clean-trained) reads through ever more off-sheet
  junk. Both effects drag CFS down. The question is the SHAPE of that fall and
  whether naive (which never projected at all) falls at least as fast.

THE THREE CFS COMPONENTS (measured per concept; combined by the harmonic mean)
  (1) MONOTONICITY = does turning the knob UP move the target readout UP, in
      order? Scored by the SPEARMAN rank correlation (Pearson on RANKS): +1 =
      perfectly ordered, 0 = no order, -1 = reversed. Negatives clip to 0 (a steer
      that moves the concept the WRONG way is not faithful). Tiny example: knobs
      [0,1,2], readout [0.1,0.5,0.9] -> Spearman +1.0; [0.1,0.9,0.5] -> +0.5.
  (2) SPECIFICITY = did ONLY the target move while OFF-target probes stayed flat?
      = 1 - (mean off-target drift / target move), clipped to [0,1]. Off-manifold
      edits smear into other concepts -> low specificity.
  (3) SUFFICIENCY = was the effect BIG enough? = Cohen's d (mean change in
      std-dev units) mapped to [0,1] by min(d / cohen_d_ample, 1). d~4 = ample.
  CFS = HARMONIC MEAN of the three (DESIGN_BRIEF §13/§14), reused from src.utils:
  conjunctive -- if ANY one is near zero, CFS is near zero. We call the project's
  cfs_score() so every milestone scores identically.

A BOOTSTRAP CONFIDENCE INTERVAL (over concepts), FROM ZERO
  We only measure a handful of concepts, so a single mean CFS could be luck. The
  BOOTSTRAP asks "how much would the mean wobble if we'd drawn a different handful
  of concepts?" by RESAMPLING the per-concept CFS list WITH REPLACEMENT many times
  (n_boot) and reading the spread of the resampled means. The central ci_pct% band
  (e.g. 5th..95th percentile) is the confidence band drawn around the curve.
  Tiny example: per-concept CFS = [0.8, 0.7, 0.9]. One resample (with replacement)
  might be [0.8, 0.8, 0.7] -> mean 0.767; another [0.9, 0.9, 0.8] -> mean 0.867.
  Do that 400 times and the 5th..95th percentile of those means is the band.

THE COLLAPSE KNEE
  The KNEE is the FIRST rung whose mean CFS drops BELOW the usability floor
  (cfs_floor). Left of the knee the concept is still faithful; at/after it,
  faithfulness has collapsed. We report the knee rung for each method.

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step3_sweep_ood_cfs.py
Reads outputs/clean_acts.npy, labels.npy, concept_dirs.npy, U_r.npy,
      probe_weights.npy, probe_bias.npy  (steps 1-2).
Writes outputs/ood_cfs_sweep.csv  (one row per shift_level x variant, with the
      mean CFS, its bootstrap CI, the three components, and the mean off-manifold
      residual + mean off-sheet energy of the shifted activations).
"""
from __future__ import annotations

import numpy as np

from _common import banner, load_cfg, outpath


# --------------------------------------------------------------------------- #
# Build the fixed STYLE operator (same operator at every rung; only its SCALE   #
# grows). It reads each activation's ON-sheet content and EMITS it into the     #
# shared off-sheet STYLE subspace S -- the very directions the concept slivers  #
# also live in (step1). So a bigger `style` floods S with junk, exactly where a #
# naive edit wasted its off-sheet strength. on-manifold put nothing in S.       #
# --------------------------------------------------------------------------- #
def style_operator(B: np.ndarray, S: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """Return a [dim, dim] operator that maps an activation's on-sheet content
    INTO the shared off-sheet style subspace S. `Style @ a` therefore lands in S
    (off the clean sheet, where the concept slivers live) -- the honest collision.
    """
    rng = np.random.default_rng(seed)
    n_style = S.shape[1]
    true_r = B.shape[1]
    P = B @ B.T                                  # projector onto the clean sheet
    M = rng.standard_normal((n_style, true_r)) / np.sqrt(true_r)  # sheet -> style coords
    # read on-sheet coords (B^T) -> mix (M) -> emit into the style subspace (S):
    return (S @ M @ B.T @ P).astype(np.float32)  # [dim, dim], range lives in S


def corrupt(acts: np.ndarray, Style: np.ndarray, gauss: float, style: float,
            seed: int) -> np.ndarray:
    """Apply ONE rung's corruption: a_shift = a + style*(Style@a) + gauss*noise.

    The style term pushes each activation OFF the clean sheet into the shared
    style subspace S; the gauss term adds isotropic noise (the ImageNet-C analog).
    """
    rng = np.random.default_rng(seed)
    noise = gauss * rng.standard_normal(acts.shape).astype(np.float32)
    return (acts + style * (acts @ Style.T) + noise).astype(np.float32)


# --------------------------------------------------------------------------- #
# Apply one steering method to a batch of (shifted) activations at knob s.     #
# The four registry steerers are reused verbatim (we do NOT re-implement them).#
# --------------------------------------------------------------------------- #
def steer_batch(variant, acts_t, edit_dir, strength, model, concept, U_r_t):
    import torch
    from src.model import build_steer
    steer = build_steer(variant, model.cfg)
    return steer(acts_t, edit_dir, strength, sae=model.sae, concept=concept,
                 basis=U_r_t)


# --------------------------------------------------------------------------- #
# MEASURE the three CFS components for ONE concept on ONE (already shifted)    #
# activation bank, for ONE method. Mirrors milestone 5's measurement exactly,  #
# only now the activations have been corrupted to a rung.                      #
# --------------------------------------------------------------------------- #
def measure_concept_cfs(variant, shifted, tgt, dirs, W, b, U_r, model, cfg):
    """Return (cfs, mono, spec, suff) for steering concept `tgt` on `shifted`."""
    import torch
    from scipy.stats import spearmanr
    from src.utils import cfs_score

    n_c = int(cfg["n_concepts"])
    s_max = float(cfg["steer_strength"])
    n_steps = int(cfg["n_knob_steps"])
    d_ample = float(cfg["cohen_d_ample"])

    acts_t = torch.from_numpy(shifted).float()             # [N, dim]
    U_r_t = torch.from_numpy(U_r).float()                  # [dim, r]
    edit_dir = torch.from_numpy(dirs[tgt]).float()         # raw concept edit Delta
    edit_dir = edit_dir / (edit_dir.norm() + 1e-8)

    w_tgt, b_tgt = W[tgt], float(b[tgt])
    off_idx = [c for c in range(n_c) if c != tgt]

    knobs = np.linspace(0.0, s_max, n_steps)
    tgt_means = []
    off_means = {c: [] for c in off_idx}
    tgt_at_0 = tgt_at_max = None
    with torch.no_grad():
        for j, s in enumerate(knobs):
            a_s = steer_batch(variant, acts_t, edit_dir, float(s), model, tgt,
                              U_r_t).cpu().numpy()
            tgt_means.append(float((a_s @ w_tgt + b_tgt).mean()))
            for c in off_idx:
                off_means[c].append(float((a_s @ W[c] + float(b[c])).mean()))
            if j == 0:
                tgt_at_0 = a_s @ w_tgt + b_tgt
            if j == n_steps - 1:
                tgt_at_max = a_s @ w_tgt + b_tgt
    tgt_means = np.asarray(tgt_means)

    # (1) MONOTONICITY: Spearman(knob, target readout), negatives -> 0.
    rho, _ = spearmanr(knobs, tgt_means)
    mono = float(max(0.0 if np.isnan(rho) else rho, 0.0))
    # (2) SPECIFICITY: 1 - normalized off-target drift.
    tgt_move = abs(tgt_means.max() - tgt_means.min()) + 1e-6
    drifts = [abs(np.asarray(off_means[c]).max() - np.asarray(off_means[c]).min())
              for c in off_idx]
    spec = float(np.clip(1.0 - float(np.mean(drifts)) / tgt_move, 0.0, 1.0))
    # (3) SUFFICIENCY: Cohen's-d effect size at full knob, mapped to [0,1].
    r0, r1 = np.asarray(tgt_at_0), np.asarray(tgt_at_max)
    pooled = (r0.std() + r1.std()) / 2.0 + 1e-6
    d_eff = float(abs(r1.mean() - r0.mean()) / pooled)
    suff = float(min(d_eff / d_ample, 1.0))

    cfs = float(cfs_score(mono, spec, suff))
    return cfs, mono, spec, suff


def bootstrap_ci(values, n_boot: int, ci_pct: float, seed: int):
    """Bootstrap CI of the MEAN over a small list of per-concept CFS values.

    Resample `values` with replacement n_boot times, take each resample's mean,
    return (lo, hi) = the central ci_pct% percentile band of those means.
    """
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return 0.0, 0.0
    means = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                      for _ in range(int(n_boot))])
    lo = float(np.percentile(means, (100 - ci_pct) / 2))
    hi = float(np.percentile(means, 100 - (100 - ci_pct) / 2))
    return lo, hi


def offsheet_energy(acts: np.ndarray, U_r: np.ndarray) -> float:
    """Mean fraction of each activation's length that lies OFF the clean sheet
    U_r  (0 = fully on the clean sheet, 1 = fully off it). This is the diagnostic
    that should CLIMB rung by rung: it is the activations leaving the clean sheet.
    """
    P = U_r @ U_r.T
    off = acts - acts @ P.T
    return float(np.mean(np.linalg.norm(off, axis=1)
                         / (np.linalg.norm(acts, axis=1) + 1e-8)))


def main() -> None:
    cfg = load_cfg()
    banner("STEP 3 — sweep the OOD ladder, MEASURE CFS per rung per method")

    import torch
    from src.model import make_model
    from src.utils import onmanifold_projection_residual, set_seed

    set_seed(int(cfg["seed"]))

    clean = np.load(outpath("clean_acts.npy"))
    dirs = np.load(outpath("concept_dirs.npy"))
    U_r = np.load(outpath("U_r.npy"))
    W = np.load(outpath("probe_weights.npy"))
    b = np.load(outpath("probe_bias.npy"))
    B_true = np.load(outpath("sheet_basis.npy"))     # the true clean sheet (step1)
    S = np.load(outpath("style_basis.npy"))          # the off-sheet style subspace
    print(f"  loaded clean acts {clean.shape}, concept_dirs {dirs.shape}, "
          f"U_r {U_r.shape}, probes {W.shape}, style_subspace {S.shape}")
    print(f"  rerun step1/step2 if any of these are missing.")

    # A model only so the registry steerers have a .cfg and an SAE handle. The
    # steerers we use (naive, onmanifold) operate purely on (acts, dir, basis).
    model = make_model(cfg)

    dim = int(cfg["dim"])
    n_c = int(cfg["n_concepts"])
    U_r_t = torch.from_numpy(U_r).float()
    # The style push EMITS into S (where the concept slivers live) -- the honest
    # collision that makes naive collapse faster than on-manifold under shift.
    Style = style_operator(B_true, S, dim, seed=int(cfg["seed"]) + 99)
    ladder = cfg["shift_ladder"]
    variants = cfg["variants"]

    print(f"\n  measuring CFS for {len(variants)} methods x {len(ladder)} rungs x "
          f"{n_c} concepts (knob sweep each). This is the real computation.\n")
    print(f"  {'shift_level':<16} {'sev':>3} {'variant':<16} "
          f"{'CFS':>6} {'[ci_lo':>7} {'ci_hi]':>7} "
          f"{'mono':>5} {'spec':>5} {'suff':>5} {'offsheet':>8} {'resid':>6}")
    print("  " + "-" * 96)

    rows = []
    for rung in ladder:
        name = rung["name"]
        sev = int(rung["severity_index"])
        # Corrupt the clean bank to THIS rung (same corruption for both methods).
        shifted = corrupt(clean, Style, float(rung["gauss"]), float(rung["style"]),
                          seed=int(cfg["seed"]) + 1000 + sev)
        off_e = offsheet_energy(shifted, U_r)         # how far off the clean sheet

        for variant in variants:
            # MEASURE CFS for every concept (each concept gets its turn as target).
            per_concept = []
            comp_acc = np.zeros(3)
            for tgt in range(n_c):
                cfs, mono, spec, suff = measure_concept_cfs(
                    variant, shifted, tgt, dirs, W, b, U_r, model, cfg)
                per_concept.append(cfs)
                comp_acc += np.array([mono, spec, suff])
            per_concept = np.asarray(per_concept)
            mean_cfs = float(per_concept.mean())
            ci_lo, ci_hi = bootstrap_ci(per_concept, int(cfg["n_boot"]),
                                        float(cfg["ci_pct"]), seed=int(cfg["seed"]) + sev)
            mono_m, spec_m, suff_m = (comp_acc / n_c).tolist()

            # Mean off-manifold residual of the EFFECTIVE edit each method applies,
            # measured against the FROZEN clean U_r (the manifold diagnostic).
            with torch.no_grad():
                edit = torch.from_numpy(dirs[int(cfg["target_concept"])]).float()
                edit = edit / (edit.norm() + 1e-8)
                a0 = torch.from_numpy(shifted[:64]).float()
                a1 = steer_batch(variant, a0, edit, float(cfg["steer_strength"]),
                                 model, int(cfg["target_concept"]), U_r_t)
                eff = (a1 - a0).reshape(-1, dim).mean(0)
                resid = onmanifold_projection_residual(eff, U_r_t)

            rows.append({
                "shift_level": name, "severity_index": sev, "variant": variant,
                "cfs": round(mean_cfs, 4),
                "cfs_ci_lo": round(ci_lo, 4), "cfs_ci_hi": round(ci_hi, 4),
                "monotonicity": round(mono_m, 4), "specificity": round(spec_m, 4),
                "sufficiency": round(suff_m, 4),
                "offsheet_energy": round(off_e, 4),
                "offmanifold_residual": round(resid, 4),
            })
            print(f"  {name:<16} {sev:>3} {variant:<16} "
                  f"{mean_cfs:>6.3f} {ci_lo:>7.3f} {ci_hi:>7.3f} "
                  f"{mono_m:>5.2f} {spec_m:>5.2f} {suff_m:>5.2f} "
                  f"{off_e:>8.3f} {resid:>6.3f}")

    # ---- write the headline table ------------------------------------------
    import pandas as pd
    df = pd.DataFrame(rows)
    out = outpath(cfg["output_csv"].split("/")[-1])
    df.to_csv(out, index=False)
    print(f"\n  saved -> {out}")

    # ---- collapse knee per method (first rung below the usability floor) ----
    floor = float(cfg["cfs_floor"])
    banner("COLLAPSE KNEE (first rung whose mean CFS drops below the floor)")
    knees = {}
    for variant in variants:
        sub = df[df.variant == variant].sort_values("severity_index")
        below = sub[sub.cfs < floor]
        if len(below):
            knee_row = below.iloc[0]
            knees[variant] = int(knee_row["severity_index"])
            print(f"  {variant:<16}: knee at sev {int(knee_row['severity_index'])} "
                  f"('{knee_row['shift_level']}'), CFS {knee_row['cfs']:.3f} < floor {floor}")
        else:
            knees[variant] = None
            print(f"  {variant:<16}: NEVER drops below floor {floor} "
                  f"(faithfulness survives the whole ladder)")

    # ---- the contract's success check + range invariant --------------------
    banner("SUCCESS CRITERIA")
    all_in_range = bool(((df["cfs"] >= 0.0) & (df["cfs"] <= 1.0)).all())
    print(f"  all CFS in [0,1] across all rungs -> {'PASS' if all_in_range else 'FAIL'}")

    onm_knee = knees.get("onmanifold_steer")
    nv_knee = knees.get("naive_steer")
    # "naive collapses at least as fast as on-manifold": naive's knee is at the
    # SAME or an EARLIER (smaller) severity than on-manifold's. A method that never
    # collapses is treated as a knee past the end of the ladder.
    last_sev = max(int(r["severity_index"]) for r in ladder)
    onm_k = last_sev + 1 if onm_knee is None else onm_knee
    nv_k = last_sev + 1 if nv_knee is None else nv_knee
    naive_first = nv_k <= onm_k
    print(f"  on-manifold knee severity = {onm_knee if onm_knee is not None else 'none'} "
          f"(treated as {onm_k}); naive knee severity = "
          f"{nv_knee if nv_knee is not None else 'none'} (treated as {nv_k})")
    print(f"  NAIVE COLLAPSES AT LEAST AS FAST as on-manifold -> "
          f"{'PASS' if naive_first else 'FAIL'}")

    # Also report the area-under-curve gap (a shape-level summary): higher AUC =
    # faithfulness survived further out.
    auc = {v: float(df[df.variant == v].sort_values("severity_index")["cfs"].mean())
           for v in variants}
    print(f"  mean CFS across the ladder (curve height): "
          + ", ".join(f"{v}={auc[v]:.3f}" for v in variants))
    print(f"  on-manifold mean-CFS >= naive mean-CFS -> "
          f"{'PASS' if auc.get('onmanifold_steer', 0) >= auc.get('naive_steer', 0) else 'FAIL'}")
    print("\nSTEP 3 done. Next: step4 draws the HEADLINE CFS-vs-shift curve.")


# REAL RUN (M6): replace `corrupt()` with REAL shifted activations. Run real CLIP
# ViT-B/16 over each real dataset (ImageNet-R, ImageNet-Sketch, ImageNet-C at
# severities 1-5, ObjectNet), cache the patch activations per rung, and feed each
# cached bank in where `shifted` is used. EVERYTHING else is identical: U_r and the
# probes stay FROZEN on clean, the knob sweep / Spearman / specificity / Cohen's-d
# measurement is unchanged, and the bootstrap CI is over the real concept set.
if __name__ == "__main__":
    main()
