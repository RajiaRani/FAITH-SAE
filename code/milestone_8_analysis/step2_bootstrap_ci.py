#!/usr/bin/env python3
# ===========================================================================
#  step2_bootstrap_ci.py  —  Milestone 8 (Analysis), Step 2
#  Take the per-concept CFS table from step1 and put HONEST ERROR BARS on it.
#  We bootstrap: resample the concept set with replacement ~2000 times to get a
#  95% confidence interval on each method's mean CFS, so we can say the
#  on-manifold-vs-naive gap is REAL, not noise. We also compute the gap CI and
#  the bootstrap p-value of "on-manifold beats naive".
#  FAITH-SAE  ·  author: Rajia Rani  ·  for research and educational purposes only
# ===========================================================================
#
#  ============ THE BOOTSTRAP, FROM ABSOLUTE ZERO ============
#
#  THE PROBLEM
#    We measured a mean CFS per method over 24 concepts. But 24 concepts is a
#    SAMPLE — if we'd happened to discover 24 slightly different concepts, the
#    mean would wobble. How much would it wobble? If on-manifold scores 0.78 and
#    naive scores 0.55, is that 0.23 gap REAL, or could the wobble alone explain
#    it? We need an error bar on each mean. The bootstrap gives us one using ONLY
#    the data we already have — no formula, no assumption about a bell curve.
#
#  THE CORE IDEA (resample with replacement)
#    Pretend our 24 concepts ARE the whole world. Draw a NEW set of 24 by picking
#    concepts from our set AT RANDOM, WITH REPLACEMENT (the same concept can be
#    picked twice, another not at all). Compute the mean CFS of that new set.
#    Repeat ~2000 times. Those 2000 means form a cloud that shows how much the
#    mean wobbles. The middle 95% of that cloud IS the 95% confidence interval.
#    Analogy: you have a bag of 24 marbles with numbers on them. To feel how
#    "lucky" your average was, you keep drawing 24 marbles (putting each back
#    after reading it), averaging, and writing the average down. After 2000 rounds
#    the spread of your written-down averages tells you how stable the average is.
#
#  TINY 5-NUMBER WORKED EXAMPLE (do this by hand once)
#    Suppose a method's per-concept CFS for 5 concepts is:
#        [0.6, 0.8, 0.7, 0.9, 0.5]      mean = 3.5 / 5 = 0.70
#    One bootstrap resample (draw 5 indices 0..4 WITH replacement), say we draw
#        indices (0, 0, 3, 1, 4)  ->  values [0.6, 0.6, 0.9, 0.8, 0.5]
#        bootstrap mean = 3.4 / 5 = 0.68
#    Another resample, indices (2, 3, 3, 1, 2) -> [0.7, 0.9, 0.9, 0.8, 0.7]
#        bootstrap mean = 4.0 / 5 = 0.80
#    Do this 2000 times -> 2000 means, e.g. spread roughly 0.60 .. 0.84. Sort
#    them; the 2.5th percentile (~0.62) is ci_low, the 97.5th (~0.83) is ci_high.
#    Report: mean 0.70, 95% CI [0.62, 0.83]. The original mean ALWAYS sits inside
#    its own CI (ci_low <= mean <= ci_high) — a sanity check this script asserts.
#
#  WHAT A CONFIDENCE INTERVAL (CI) MEANS
#    A 95% CI is a range built so that, if we repeated the whole experiment many
#    times, ~95% of the intervals we'd build would contain the true mean. Loosely:
#    "we're fairly sure the real mean lives in here." A WIDE CI = noisy / few
#    concepts; a NARROW CI = stable / many concepts.
#
#  "NON-OVERLAPPING CIs => THE DIFFERENCE IS REAL"
#    If on-manifold's CI is [0.74, 0.82] and naive's is [0.49, 0.61], they do NOT
#    overlap — there's no single value both methods could plausibly share, so the
#    gap is not just luck of which concepts we picked: it is a REAL difference.
#    (If they DID overlap, we could not rule out "same true mean, different draw".)
#    The cleanest test is to bootstrap the GAP itself (on-manifold minus naive,
#    using the SAME resampled concepts each round): if the gap's 95% CI is
#    entirely above 0, on-manifold is significantly more faithful. We report both.
#
#  WHY RESAMPLE CONCEPTS (not images or knob steps)?
#    The claim is "on-manifold steers CONCEPTS more faithfully". The unit we
#    generalize over is the concept, so the concept is what we resample. (Design
#    brief §7: "bootstrap confidence intervals over concepts".)
#
#  ============ WHAT THIS SCRIPT DOES ============
#    1. Read outputs/per_concept_cfs.csv (from step1).
#    2. For each (variant, shift rung): resample the per-concept CFS values with
#       replacement n_boot times, take each resample's mean, and read off the
#       2.5th / 97.5th percentiles -> (mean_cfs, ci_low, ci_high).
#    3. Also bootstrap the on-manifold-minus-naive GAP per shift (paired on the
#       same resampled concepts) -> gap mean, gap CI, and a bootstrap p-value.
#    4. Write outputs/bootstrap_ci.csv (variant, shift, mean_cfs, ci_low, ci_high,
#       n_concepts) and print the gap analysis. Assert every CI is well-formed.
#
#  RUN:  /usr/bin/python3 step2_bootstrap_ci.py   (needs step1's CSV)
#  ========================================================================

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from _common import banner, load_cfg, outpath


def worked_5_number_example() -> None:
    """Print the by-hand 5-number bootstrap from the docstring so the reader can
    verify the mechanic before trusting the 2000-resample version."""
    banner("BOOTSTRAP — tiny 5-number worked example (check the mechanic by hand)")
    vals = np.array([0.6, 0.8, 0.7, 0.9, 0.5])
    print(f"  5 per-concept CFS values : {list(vals)}   mean = {vals.mean():.2f}")
    rng = np.random.default_rng(0)
    # show two example resamples explicitly
    for _ in range(2):
        idx = rng.integers(0, len(vals), size=len(vals))
        print(f"    resample indices {tuple(int(i) for i in idx)} -> "
              f"values {[round(float(v),1) for v in vals[idx]]} -> "
              f"bootstrap mean = {vals[idx].mean():.2f}")
    # now do 2000 and read off the CI
    means = np.array([vals[rng.integers(0, len(vals), len(vals))].mean()
                      for _ in range(2000)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    print(f"  ... 2000 resamples -> 95% CI of the mean = [{lo:.2f}, {hi:.2f}]")
    print(f"  the original mean {vals.mean():.2f} sits INSIDE its CI "
          f"({lo:.2f} <= {vals.mean():.2f} <= {hi:.2f}) — always true; a sanity check.")


def read_per_concept(path: str):
    """variant -> shift -> {concept_id -> cfs}.  We key by concept_id so the GAP
    bootstrap can pair on-manifold and naive on the SAME resampled concepts."""
    table = defaultdict(lambda: defaultdict(dict))
    shift_order, seen = [], set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table[row["variant"]][row["shift"]][row["concept_id"]] = float(row["cfs"])
            if row["shift"] not in seen:
                seen.add(row["shift"])
                shift_order.append(row["shift"])
    return table, shift_order


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, ci_pct: float, rng):
    """Percentile bootstrap CI of the MEAN of `values`.

    Resample indices with replacement n_boot times, take each resample's mean,
    and read the (100-ci_pct)/2 and (100+ci_pct)/2 percentiles. Returns
    (mean, ci_low, ci_high). The mean is the plain sample mean (the point estimate)."""
    n = len(values)
    # [n_boot, n] matrix of resampled indices, then index -> [n_boot, n] values.
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)           # one mean per resample
    half = (100.0 - ci_pct) / 2.0
    lo, hi = np.percentile(boot_means, [half, 100.0 - half])
    return float(values.mean()), float(lo), float(hi)


def bootstrap_gap_ci(vals_a: np.ndarray, vals_b: np.ndarray, n_boot, ci_pct, rng):
    """PAIRED bootstrap of the gap mean(A) - mean(B) over the SAME resampled
    concepts each round (A and B share the concept axis, aligned by concept_id).
    Returns (gap, ci_low, ci_high, p_value) where p_value is the bootstrap
    fraction of resamples in which the gap is <= 0 (one-sided 'A does NOT beat B').
    """
    n = len(vals_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    gaps = vals_a[idx].mean(axis=1) - vals_b[idx].mean(axis=1)
    half = (100.0 - ci_pct) / 2.0
    lo, hi = np.percentile(gaps, [half, 100.0 - half])
    p_not_better = float(np.mean(gaps <= 0.0))      # bootstrap one-sided p-value
    gap_point = float(vals_a.mean() - vals_b.mean())
    return gap_point, float(lo), float(hi), p_not_better


def main() -> None:
    cfg = load_cfg()
    n_boot = int(cfg["n_boot"])
    ci_pct = float(cfg["ci_pct"])
    banner(f"STEP 2 — bootstrap {ci_pct:.0f}% CIs over concepts "
           f"({n_boot} resamples per cell)")

    worked_5_number_example()

    table, shift_order = read_per_concept(outpath("per_concept_cfs.csv"))
    variants = list(cfg["variants"])
    rng = np.random.default_rng(int(cfg["seed"]))   # fixed seed -> reproducible CIs

    # ---- 1. Per (variant, shift) mean CFS + 95% CI --------------------------
    rows = []
    print("\n  Per-method, per-shift mean CFS with bootstrap 95% CI:")
    print(f"  {'variant':<18} {'shift':>10} {'mean':>7} {'ci_low':>8} {'ci_high':>8}  n")
    print("  " + "-" * 62)
    for variant in variants:
        for shift in shift_order:
            by_concept = table[variant][shift]
            vals = np.array([by_concept[c] for c in sorted(by_concept)], dtype=float)
            mean, lo, hi = bootstrap_mean_ci(vals, n_boot, ci_pct, rng)
            # sanity: the point mean must lie inside its own CI.
            assert lo - 1e-6 <= mean <= hi + 1e-6, (variant, shift, lo, mean, hi)
            rows.append({
                "variant": variant, "shift": shift,
                "mean_cfs": round(mean, 4),
                "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                "n_concepts": len(vals),
            })
            print(f"  {variant:<18} {shift:>10} {mean:>7.3f} {lo:>8.3f} {hi:>8.3f}  "
                  f"{len(vals)}")

    out = outpath("bootstrap_ci.csv")
    fields = ["variant", "shift", "mean_cfs", "ci_low", "ci_high", "n_concepts"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  saved -> {out}")

    # ---- 2. The headline GAP: on-manifold minus naive, per shift ------------
    banner("Is the on-manifold vs naive gap REAL? (paired bootstrap of the gap)")
    print(f"  {'shift':>10} {'gap':>7} {'gap_ci_low':>11} {'gap_ci_high':>11} "
          f"{'p(<=0)':>8}  verdict")
    print("  " + "-" * 66)
    gap_rows = []
    for shift in shift_order:
        a = table["onmanifold_steer"][shift]
        b = table["naive_steer"][shift]
        common = sorted(set(a) & set(b))            # pair on shared concept ids
        va = np.array([a[c] for c in common], dtype=float)
        vb = np.array([b[c] for c in common], dtype=float)
        gap, glo, ghi, p = bootstrap_gap_ci(va, vb, n_boot, ci_pct, rng)
        # "real" if the gap's 95% CI is entirely above 0 (CIs do not straddle 0).
        real = glo > 0.0
        verdict = "REAL (CI > 0)" if real else "not significant"
        print(f"  {shift:>10} {gap:>7.3f} {glo:>11.3f} {ghi:>11.3f} {p:>8.4f}  {verdict}")
        gap_rows.append({"shift": shift, "gap": gap, "ci_low": glo,
                         "ci_high": ghi, "p_value": p, "significant": real})

    # Also report the headline CLEAN-rung gap and CI-overlap reading.
    clean = next(g for g in gap_rows if g["shift"] == shift_order[0])
    on_clean = next(r for r in rows
                    if r["variant"] == "onmanifold_steer" and r["shift"] == shift_order[0])
    na_clean = next(r for r in rows
                    if r["variant"] == "naive_steer" and r["shift"] == shift_order[0])
    overlap = not (on_clean["ci_low"] > na_clean["ci_high"]
                   or na_clean["ci_low"] > on_clean["ci_high"])
    print(f"\n  CLEAN rung: on-manifold CI [{on_clean['ci_low']:.3f}, {on_clean['ci_high']:.3f}]"
          f"  vs naive CI [{na_clean['ci_low']:.3f}, {na_clean['ci_high']:.3f}]")
    print(f"    CIs overlap? {'YES' if overlap else 'NO'} -> "
          f"{'gap not separable by CI alone' if overlap else 'gap is REAL (non-overlapping CIs)'}")
    print(f"    gap = {clean['gap']:.3f}, 95% CI [{clean['ci_low']:.3f}, {clean['ci_high']:.3f}], "
          f"bootstrap p(gap<=0) = {clean['p_value']:.4f}")

    print("\nSTEP 2 done. Next: step3 renders the two figures with these CI bands.")


# REAL RUN (M8): identical. The bootstrap is data-agnostic — point step1 at the
# real per-concept CFS table (real CLIP ViT-B/16 across the OOD ladder) and this
# step produces the real CIs and gap significance with no code change. For very
# large concept sets, raise n_boot for smoother CI edges (cost is linear).
if __name__ == "__main__":
    main()
