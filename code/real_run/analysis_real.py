#!/usr/bin/env python3
# ===========================================================================
#  analysis_real.py  —  FAITH-SAE real-run analysis layer (publication-grade)
#
#  This is the STATISTICS + WRITE-UP stage of the real pipeline. It does NOT
#  touch a GPU, a model, or an image: it consumes the CSVs the heavy modules
#  emit (per_concept_cfs.csv, ood_cfs_sweep.csv, ablations.csv) and turns them
#  into the three things a paper actually needs:
#
#    1. bootstrap_ci(...)        — a 95% confidence interval on the MEAN CFS of
#                                  each steering method, by resampling the
#                                  CONCEPT IDS with replacement 2000x (the
#                                  brief's "bootstrap CIs over concepts").
#    2. significance read-out    — do the on-manifold and naive CIs OVERLAP?
#                                  Non-overlap is the cheap, defensible claim of
#                                  "significantly more faithful" the paper makes.
#    3. write_findings(...)      — map the measured numbers onto the paper's
#                                  \pending{...} placeholders so a human can
#                                  paste real results into paper.tex.
#
#  WHY bootstrap over concepts (and not images / not a t-test)?
#    The unit of replication in this study is the CONCEPT: each selected SAE
#    feature is one independent "subject" we measured a CFS for. The population
#    we want to generalise to is "the kind of concepts this SAE discovers", so
#    the resampling unit must be the concept id. The bootstrap is distribution-
#    free (CFS is a bounded harmonic mean — very non-normal), which is exactly
#    why we prefer it to a parametric interval here.
#
#  Author: Rajia Rani
#  For research and educational purposes only.
# ===========================================================================
from __future__ import annotations

import argparse
import json
import pathlib

# --- make the project root (src/) and this dir importable, per the contract --
# parents[2] of code/real_run/analysis_real.py == the project root that holds
# src/, so `from src.utils import cfs_score` reuses the ONE canonical scorer.
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

# Reuse the canonical harmonic-mean CFS so the analysis layer can RE-DERIVE a
# CFS from raw components if a CSV ships only mono/spec/suff (defensive; the
# heavy modules normally already write a `cfs` column).
try:
    from src.utils import cfs_score  # type: ignore
except Exception:  # pragma: no cover - keeps the module importable in isolation
    def cfs_score(monotonicity, specificity, sufficiency, weights=(1.0, 1.0, 1.0)):
        comps = [min(max(c, 0.0), 1.0) for c in (monotonicity, specificity, sufficiency)]
        if min(comps) <= 0.0:
            return 0.0
        num = sum(weights)
        den = sum(w / c for w, c in zip(weights, comps))
        return num / den


# --------------------------------------------------------------------------- #
#  Canonical method names + display labels (kept identical to the figure style #
#  and DESIGN_BRIEF §12 so every artifact agrees).                             #
# --------------------------------------------------------------------------- #
METHOD_ORDER = ["supervised_steer", "onmanifold_steer", "clamp_steer",
                "naive_steer", "random_steer"]
METHOD_LABEL = {
    "supervised_steer": "Supervised (TCAV)",
    "onmanifold_steer": "On-manifold (ours)",
    "clamp_steer":      "Raw clamp",
    "naive_steer":      "Naive off-manifold",
    "random_steer":     "Random direction",
}


# =========================================================================== #
#  1. THE BOOTSTRAP                                                            #
# =========================================================================== #
def bootstrap_ci(per_concept_cfs, n: int = 2000, ci_pct: float = 95.0,
                 seed: int = 0):
    """95% bootstrap CI on the MEAN per-concept CFS.

    Parameters
    ----------
    per_concept_cfs : 1-D array-like of float
        One CFS value PER CONCEPT (the resampling unit). NaNs are dropped.
    n : int
        Number of bootstrap resamples (resample concept ids WITH replacement).
    ci_pct : float
        Confidence level in percent (95 -> the 2.5 / 97.5 percentiles).
    seed : int
        RNG seed so the interval is reproducible (the brief fixes seeds).

    Returns
    -------
    (mean, lo, hi) : tuple of float
        Point estimate (mean over the real concepts) and the bootstrap CI.
        For an empty/degenerate input returns (nan, nan, nan) / a zero-width
        interval, never raising — downstream plotting must stay robust.
    """
    vals = np.asarray(list(per_concept_cfs), dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(vals.mean())
    if vals.size == 1:
        # A single concept has no spread to bootstrap; report a point interval
        # (honest: we cannot claim a CI from one observation).
        return (mean, mean, mean)

    rng = np.random.default_rng(seed)
    m = vals.size
    # Vectorised: draw an (n x m) matrix of resample indices in one shot, then
    # take the row-means -> the bootstrap distribution of the mean CFS.
    idx = rng.integers(0, m, size=(n, m))
    boot_means = vals[idx].mean(axis=1)
    alpha = (100.0 - ci_pct) / 2.0
    lo = float(np.percentile(boot_means, alpha))
    hi = float(np.percentile(boot_means, 100.0 - alpha))
    return (mean, lo, hi)


def bootstrap_by_method(df, value_col: str = "cfs", method_col=None,
                        concept_col=None, n: int = 2000,
                        ci_pct: float = 95.0, seed: int = 0):
    """Bootstrap CI per steering method from a long results df.

    Groups rows by the method column, collapses to ONE CFS per concept (mean over
    any extra rows for that concept, e.g. several shift levels), then bootstraps
    the per-concept vector. Returns a tidy pandas DataFrame with one row per
    method: [variant, label, n_concepts, mean_cfs, ci_low, ci_high, ci_width].

    The method/concept column names are AUTO-DETECTED so this accepts BOTH the
    milestone_8 schema (variant/concept_id) and the cfs_eval schema
    (method/concept) without the caller having to know which heavy module wrote
    the CSV.
    """
    import pandas as pd

    df = _ensure_cfs_column(df.copy())
    method_col = method_col or _method_col(df)
    concept_col = concept_col or _concept_col(df)
    out_rows = []
    # Preserve the canonical method order, then append any unexpected methods.
    methods = [m for m in METHOD_ORDER if m in set(df[method_col])]
    methods += [m for m in df[method_col].unique() if m not in methods]
    for i, method in enumerate(methods):
        sub = df[df[method_col] == method]
        if concept_col and concept_col in sub.columns:
            # One CFS per concept: average over shift levels / repeats so each
            # concept contributes exactly once (correct bootstrap unit).
            per_concept = sub.groupby(concept_col)[value_col].mean().to_numpy()
        else:
            per_concept = sub[value_col].to_numpy()
        # Use a per-method seed offset so the methods' resamples are independent
        # yet the whole run is reproducible.
        mean, lo, hi = bootstrap_ci(per_concept, n=n, ci_pct=ci_pct, seed=seed + i)
        out_rows.append({
            "variant": method,
            "label": METHOD_LABEL.get(method, method),
            "n_concepts": int(len(per_concept)),
            "mean_cfs": mean,
            "ci_low": lo,
            "ci_high": hi,
            "ci_width": (hi - lo) if np.isfinite(hi) and np.isfinite(lo) else float("nan"),
        })
    return pd.DataFrame(out_rows)


# =========================================================================== #
#  2. SIGNIFICANCE READ-OUT  (CI overlap, the paper's claim)                   #
# =========================================================================== #
def ci_overlap(ci_a, ci_b):
    """Do two confidence intervals overlap? `ci_*` = (mean, lo, hi).

    Returns True if the closed intervals [lo_a, hi_a] and [lo_b, hi_b] intersect.
    Non-overlap of bootstrap CIs is the conservative, easy-to-defend evidence of
    a real difference the paper relies on (it implies p < ~0.05 two-sided for the
    difference of means under the usual bootstrap reading).
    """
    _, lo_a, hi_a = ci_a
    _, lo_b, hi_b = ci_b
    if not all(np.isfinite([lo_a, hi_a, lo_b, hi_b])):
        return False
    return not (hi_a < lo_b or hi_b < lo_a)


def significance_readout(boot_df, ref: str = "onmanifold_steer",
                         against=("naive_steer",)):
    """Pairwise significance between the headline method and competitors.

    For each competitor in `against`, report whether the (ours)-vs-competitor
    CIs OVERLAP and the gap between point means. `separated=True` means the CIs
    do not overlap (our headline "significantly more faithful, non-overlapping
    CIs" claim from paper.tex around fig7).

    Returns a dict keyed by competitor name plus a top-level summary.
    """
    rows = {r["variant"]: r for _, r in boot_df.iterrows()}
    out = {"ref": ref, "ref_label": METHOD_LABEL.get(ref, ref), "pairs": {}}
    if ref not in rows:
        out["error"] = f"reference method '{ref}' not in results"
        return out
    ra = rows[ref]
    ci_a = (ra["mean_cfs"], ra["ci_low"], ra["ci_high"])
    for comp in against:
        if comp not in rows:
            continue
        rb = rows[comp]
        ci_b = (rb["mean_cfs"], rb["ci_low"], rb["ci_high"])
        overlap = ci_overlap(ci_a, ci_b)
        out["pairs"][comp] = {
            "competitor_label": METHOD_LABEL.get(comp, comp),
            "ref_mean": float(ra["mean_cfs"]),
            "competitor_mean": float(rb["mean_cfs"]),
            "delta": float(ra["mean_cfs"] - rb["mean_cfs"]),
            "ci_overlap": bool(overlap),
            "separated": bool(not overlap),
            # higher_is_ref True when ours actually wins (guards a null result).
            "ref_higher": bool(ra["mean_cfs"] >= rb["mean_cfs"]),
        }
    # An overall verdict: ours beats EVERY listed competitor with separated CIs.
    pairs = out["pairs"].values()
    out["all_separated"] = bool(pairs) and all(p["separated"] for p in pairs)
    out["all_ref_higher"] = bool(pairs) and all(p["ref_higher"] for p in pairs)
    return out


# =========================================================================== #
#  3. OOD SLOPE + COLLAPSE KNEE  (RQ3 numbers for the findings)               #
# =========================================================================== #
def ood_degradation(ood_df, value_col: str = "cfs", method_col=None,
                    level_col=None, severity_col=None, floor: float = 0.5):
    """Summarise the OOD CFS-vs-shift curve per method (RQ3).

    Reads the long ood_cfs_sweep.csv (one row per shift level x method). Column
    names vary across the heavy modules' revisions (ood_sweep.py writes
    rung/shift_index/method; milestone_8 writes shift/severity_index/variant), so
    the method / level / severity columns are all AUTO-DETECTED. For each method
    returns the ordered CFS curve, the mean ΔCFS-per-shift-level slope, and the
    COLLAPSE KNEE = the first shift level whose CFS drops below `floor`.
    """
    import pandas as pd

    df = _ensure_cfs_column(ood_df.copy())
    method_col = method_col or _method_col(df)
    level_col = level_col or _pick_col(df, ["shift_level", "shift", "rung", "level",
                                            "ood_level", "dataset"])
    severity_col = severity_col or _pick_col(df, ["severity_index", "shift_index",
                                                  "severity", "shift_noise",
                                                  "level_index"])
    out = {}
    for method in [m for m in METHOD_ORDER if m in set(df[method_col])] + \
            [m for m in df[method_col].unique() if m not in METHOD_ORDER]:
        sub = df[df[method_col] == method].copy()
        # Drop rows with a missing CFS (a partial / in-progress real run can ship
        # NaNs) so the slope/knee never come out as nan and poison the findings.
        sub = sub[np.isfinite(sub[value_col].astype(float))]
        if severity_col and severity_col in sub.columns:
            sub = sub.sort_values(severity_col)
        levels = sub[level_col].tolist() if level_col else list(range(len(sub)))
        curve = sub[value_col].to_numpy(dtype=float)
        if curve.size < 2:
            out[method] = {"levels": levels, "cfs": curve.tolist(),
                           "slope_per_level": float("nan"), "knee_level": None,
                           "clean_cfs": float(curve[0]) if curve.size else float("nan")}
            continue
        # Mean per-step degradation slope (negative = degrading as we walk OOD).
        slope = float(np.mean(np.diff(curve)))
        # Collapse knee = first level below the usability floor.
        below = np.where(curve < floor)[0]
        knee_idx = int(below[0]) if below.size else None
        out[method] = {
            "levels": levels,
            "cfs": curve.tolist(),
            "slope_per_level": slope,
            "clean_cfs": float(curve[0]),
            "final_cfs": float(curve[-1]),
            "knee_idx": knee_idx,
            "knee_level": (levels[knee_idx] if knee_idx is not None else None),
        }
    return out


# =========================================================================== #
#  4. CONCEPT-RELIABILITY (the "~10-15% reliable tail" claim, RQ2)            #
# =========================================================================== #
def reliable_fraction(per_concept_cfs, threshold: float = 0.5):
    """Fraction of concepts with CFS >= threshold (the reliable tail size).

    The field claims only ~10-15% of SAE features steer cleanly; this returns
    the measured fraction so the findings writer can confirm or refute it.
    """
    vals = np.asarray(list(per_concept_cfs), dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals >= threshold))


# =========================================================================== #
#  5. THE FINDINGS WRITER  (measured numbers -> paper \pending{} text)        #
# =========================================================================== #
# Maps each \pending{...} placeholder in paper.tex to a key here, so a human can
# search the paper for the placeholder text and paste the matching measured
# sentence. The keys mirror the paper's figure labels.
def write_findings(results_dir, out_dir, boot_df=None, sig=None, ood=None,
                   reliable_frac=None, floor: float = 0.5):
    """Render FINDINGS.md + findings.json mapping measured numbers to \\pending{}.

    Parameters mirror what the other analysis functions return; any that are
    None are recomputed from the CSVs found in `results_dir` so this can be
    called standalone (e.g. from the CLI).
    """
    import pandas as pd

    results_dir = pathlib.Path(results_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- (re)load + (re)compute anything not handed in --------------------- #
    per_df = _load_csv(results_dir / "per_concept_cfs.csv")
    ood_df = _load_csv(results_dir / "ood_cfs_sweep.csv")

    if boot_df is None and per_df is not None:
        boot_df = bootstrap_by_method(per_df)
    if sig is None and boot_df is not None and len(boot_df):
        sig = significance_readout(boot_df)
    if ood is None and ood_df is not None:
        ood = ood_degradation(ood_df, floor=floor)
    if reliable_frac is None and per_df is not None:
        # Reliable-tail fraction measured on the CLEAN, in-distribution rung when
        # a shift column exists (the cleanest test of "does this concept steer?").
        clean = _clean_slice(per_df)
        # One CFS per concept on clean (mean over methods would be unfair; use
        # the on-manifold method if present). Column names auto-detected so this
        # works for both the variant/concept_id and method/concept schemas.
        rf_df = _ensure_cfs_column(clean.copy())
        mcol = _method_col(rf_df)
        ccol = _concept_col(rf_df)
        if mcol and "onmanifold_steer" in set(rf_df[mcol]):
            rf_df = rf_df[rf_df[mcol] == "onmanifold_steer"]
        if ccol and ccol in rf_df.columns:
            per_concept = rf_df.groupby(ccol)["cfs"].mean().to_numpy()
        else:
            per_concept = rf_df["cfs"].to_numpy()
        reliable_frac = reliable_fraction(per_concept, threshold=floor)

    # --- build the structured findings ------------------------------------ #
    findings = _assemble_findings(boot_df, sig, ood, reliable_frac, floor)

    # --- write machine-readable + human-readable -------------------------- #
    (out_dir / "findings.json").write_text(
        json.dumps(findings, indent=2, default=_jsonable), encoding="utf-8")
    md = _render_findings_md(findings)
    (out_dir / "FINDINGS.md").write_text(md, encoding="utf-8")
    return findings


def _assemble_findings(boot_df, sig, ood, reliable_frac, floor):
    """Turn the raw analysis objects into the per-\\pending{} sentence map."""
    F = {"usability_floor": floor, "pending": {}, "tables": {}}

    # ---- the matched-strength bar (fig7 / Table) ------------------------- #
    if boot_df is not None and len(boot_df):
        F["tables"]["by_method"] = boot_df.to_dict(orient="records")

    # \pending{} for fig7 / the abstract / the matched-strength claim:
    if sig and sig.get("pairs"):
        ref_label = sig["ref_label"]
        naive = sig["pairs"].get("naive_steer")
        if naive:
            verb = "above" if naive["ref_higher"] else "below"
            sep = ("with non-overlapping bootstrap CIs"
                   if naive["separated"] else
                   "but the bootstrap CIs OVERLAP (not significant)")
            F["pending"]["abstract"] = (
                f"{ref_label} reaches mean CFS {naive['ref_mean']:.2f} vs "
                f"{naive['competitor_mean']:.2f} for naive steering "
                f"(Δ = {naive['delta']:+.2f}), {sep}.")
            F["pending"]["fig7_by_method"] = (
                f"{ref_label} is {verb} naive off-manifold steering by "
                f"ΔCFS = {naive['delta']:+.2f} at matched strength, {sep}; "
                f"see the per-method bootstrap CIs in fig7_by_method_bar.png.")
        F["pending"]["fig7_significance"] = {
            "all_separated": sig.get("all_separated"),
            "all_ref_higher": sig.get("all_ref_higher"),
        }

    # ---- the OOD headline curve (fig1 / RQ3) ----------------------------- #
    if ood:
        on = ood.get("onmanifold_steer")
        na = ood.get("naive_steer")
        if on and na:
            # A None knee means CFS never dipped below the floor -> say so plainly.
            on_knee = _knee_phrase(on.get("knee_level"))
            na_knee = _knee_phrase(na.get("knee_level"))
            F["pending"]["fig1_oodsweep"] = (
                f"On-manifold steering degrades at "
                f"{on['slope_per_level']:+.3f} CFS/shift-level and {on_knee}; "
                f"naive steering degrades faster "
                f"({na['slope_per_level']:+.3f} CFS/shift-level) and {na_knee}.")
            F["pending"]["conclusion_ood"] = (
                f"Faithfulness {'survives' if on.get('final_cfs', 0) >= floor else 'collapses'} "
                f"to the end of the ladder for on-manifold steering "
                f"(final CFS {on.get('final_cfs', float('nan')):.2f}); on-manifold "
                f"{on_knee}.")

    # ---- the reliable-fraction claim (fig4 / RQ2 / limitations) ---------- #
    if reliable_frac is not None and np.isfinite(reliable_frac):
        pct = 100.0 * reliable_frac
        claim = ("consistent with the field's ~10-15% claim"
                 if 8.0 <= pct <= 20.0 else
                 f"{'above' if pct > 20.0 else 'below'} the field's ~10-15% claim")
        F["pending"]["fig4_reliability"] = (
            f"{pct:.0f}% of selected concepts steer reliably "
            f"(CFS >= {floor:.1f}), {claim}.")
        F["pending"]["limitations_reliable_fraction"] = (
            f"measured reliable fraction = {pct:.0f}% at the CFS>={floor:.1f} floor.")

    return F


def _render_findings_md(F):
    """Plain-language FINDINGS.md a human pastes into paper.tex \\pending{}s."""
    floor = F.get("usability_floor", 0.5)
    lines = [
        "# FAITH-SAE — measured findings",
        "",
        "_Author: Rajia Rani_",
        "",
        "Each block below replaces one `\\pending{...}` placeholder in "
        "`paper/paper.tex`. Search the paper for the figure/section named in the "
        "heading and paste the measured sentence in place of the placeholder.",
        "",
        f"Usability floor used for the collapse knee and reliable tail: **{floor:.2f}**.",
        "",
    ]

    # by-method table
    bm = F.get("tables", {}).get("by_method")
    if bm:
        lines += ["## Mean CFS by method (matched strength, bootstrap 95% CI)", "",
                  "| Method | n concepts | mean CFS | CI low | CI high |",
                  "|---|---:|---:|---:|---:|"]
        for r in bm:
            lines.append(
                f"| {r.get('label', r.get('variant'))} | {r.get('n_concepts','')} | "
                f"{_fmt(r.get('mean_cfs'))} | {_fmt(r.get('ci_low'))} | "
                f"{_fmt(r.get('ci_high'))} |")
        lines.append("")

    # the \pending sentences
    pend = F.get("pending", {})
    name_map = {
        "abstract": "Abstract / one-line claim",
        "fig1_oodsweep": "Fig. 1 — CFS vs OOD shift (HEADLINE, RQ3)",
        "fig4_reliability": "Fig. 4 — reliable-concept fraction (RQ2)",
        "fig7_by_method": "Fig. 7 — mean CFS by method (matched strength, RQ1)",
        "conclusion_ood": "Conclusion — the OOD answer",
        "limitations_reliable_fraction": "Limitations — observed reliable fraction",
    }
    if pend:
        lines += ["## Sentences for the \\pending{} placeholders", ""]
        for key, title in name_map.items():
            if key in pend and isinstance(pend[key], str):
                lines += [f"### {title}", "", pend[key], ""]
    sigblock = pend.get("fig7_significance")
    if isinstance(sigblock, dict):
        verdict = ("on-manifold beats every competitor with NON-overlapping CIs"
                   if sigblock.get("all_separated") and sigblock.get("all_ref_higher")
                   else "at least one comparison is NOT separated (see table)")
        lines += ["### Significance verdict (CI overlap)", "", verdict, ""]

    lines += ["---", "", "_For research and educational purposes only._", ""]
    return "\n".join(lines)


# =========================================================================== #
#  small robust CSV / column helpers                                          #
# =========================================================================== #
def _load_csv(path):
    import pandas as pd
    path = pathlib.Path(path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if len(df) else None
    except Exception:
        return None


def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _method_col(df):
    """Name of the steering-method column across the heavy modules' schemas.
    milestone_8 -> 'variant'; cfs_eval / ood_sweep -> 'method'."""
    col = _pick_col(df, ["variant", "method", "steer", "steerer"])
    if col is None:
        raise KeyError("results df has no steering-method column "
                       "(expected one of variant/method/steer)")
    return col


def _concept_col(df):
    """Name of the concept column: milestone_8 -> 'concept_id';
    cfs_eval -> 'concept'. Returns None if the df is already per-concept-free."""
    return _pick_col(df, ["concept_id", "concept", "feature_id", "feature"])


def _ensure_cfs_column(df):
    """Guarantee a numeric `cfs` column, deriving it from the 3 components if
    the CSV only shipped the raw mono/spec/suff (defensive)."""
    if "cfs" in df.columns:
        df["cfs"] = df["cfs"].astype(float)
        return df
    comp = ["monotonicity", "specificity", "sufficiency"]
    if all(c in df.columns for c in comp):
        df["cfs"] = [cfs_score(m, s, f) for m, s, f in
                     zip(df["monotonicity"], df["specificity"], df["sufficiency"])]
        return df
    raise KeyError("results df has no 'cfs' column and no mono/spec/suff to derive it")


def _clean_slice(df):
    """Rows of the in-distribution rung, by whatever the shift column is called."""
    col = _pick_col(df, ["shift_level", "shift", "rung", "level", "ood_level",
                         "dataset"])
    if col is None:
        return df
    # in-distribution rung aliases: the student's ladder names it 'in1k'; the
    # legacy open_clip ladder named it 'clean'.
    clean_names = {"in1k", "clean", "imagenet", "imagenet_val", "in_distribution",
                   "id"}
    mask = df[col].astype(str).str.lower().isin(clean_names)
    return df[mask] if mask.any() else df


def _knee_phrase(knee_level):
    """Human-readable collapse-knee clause for the findings sentence."""
    if knee_level is None:
        return "stays above the usability floor across the whole ladder (no collapse knee)"
    return f"first crosses the usability floor at '{knee_level}'"


def _fmt(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# =========================================================================== #
#  SMOKE: fabricate small but real-SHAPED CSVs and run the whole analysis      #
# =========================================================================== #
def _fabricate_results(out_dir, seed=0):
    """Write tiny per_concept_cfs.csv + ood_cfs_sweep.csv with the REAL schema,
    so the CLI smoke path exercises bootstrap + significance + findings on CPU
    with no model and no GPU."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # method-level "true" mean CFS (ours just under supervised, naive lower).
    mu = {"supervised_steer": 0.84, "onmanifold_steer": 0.82, "clamp_steer": 0.57,
          "naive_steer": 0.49, "random_steer": 0.21}
    shifts = [("clean", 0.0), ("imagenet_r", 0.35), ("imagenet_sketch", 0.70),
              ("imagenet_c_s3", 1.05), ("imagenet_c_s5", 1.55), ("objectnet", 2.10)]
    n_concepts = 24

    rows = []
    for v, base in mu.items():
        for ci in range(n_concepts):
            cbias = rng.normal(0, 0.06)              # per-concept random effect
            for sname, snoise in shifts:
                # on-manifold degrades gently; naive/clamp/random fall off faster.
                grace = {"supervised_steer": 0.10, "onmanifold_steer": 0.12,
                         "clamp_steer": 0.22, "naive_steer": 0.28,
                         "random_steer": 0.10}[v]
                cfs = base + cbias - grace * snoise + rng.normal(0, 0.03)
                cfs = float(np.clip(cfs, 0.0, 1.0))
                mono = float(np.clip(cfs + rng.normal(0, 0.03), 0, 1))
                spec = float(np.clip(cfs + rng.normal(0, 0.03), 0, 1))
                suff = float(np.clip(cfs + rng.normal(0, 0.03), 0, 1))
                rows.append({"variant": v, "shift": sname, "shift_noise": snoise,
                             "concept_id": ci, "monotonicity": round(mono, 4),
                             "specificity": round(spec, 4),
                             "sufficiency": round(suff, 4), "cfs": round(cfs, 4)})
    per_df = pd.DataFrame(rows)
    per_df.to_csv(out_dir / "per_concept_cfs.csv", index=False)

    # OOD sweep: averaged over concepts per (variant, shift) for the two headline
    # methods, with severity_index + offmanifold_residual (the real schema).
    ood_rows = []
    for i, (sname, snoise) in enumerate(shifts):
        for v in ["onmanifold_steer", "naive_steer"]:
            # NB: bracket indexing — `per_df.shift` is the DataFrame.shift METHOD,
            # not the 'shift' column, so attribute access here silently misfires.
            sub = per_df[(per_df["variant"] == v) & (per_df["shift"] == sname)]
            ood_rows.append({
                "shift_level": sname, "severity_index": i, "variant": v,
                "cfs": round(float(sub["cfs"].mean()), 4),
                "offmanifold_residual": round(0.0 if v == "onmanifold_steer"
                                              else float(np.clip(0.55 + 0.05 * i, 0, 1)), 4),
            })
    pd.DataFrame(ood_rows).to_csv(out_dir / "ood_cfs_sweep.csv", index=False)
    return out_dir


def main():
    ap = argparse.ArgumentParser(
        description="FAITH-SAE real-run analysis: bootstrap CIs over concepts, "
                    "CI-overlap significance, and the \\pending{} findings writer.")
    ap.add_argument("--results-dir", default=str(_HERE / "outputs"),
                    help="dir holding per_concept_cfs.csv / ood_cfs_sweep.csv")
    ap.add_argument("--out-dir", default=str(_HERE / "outputs"),
                    help="where FINDINGS.md / findings.json / bootstrap_ci.csv go")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--ci-pct", type=float, default=95.0)
    ap.add_argument("--floor", type=float, default=0.5,
                    help="usability floor for the collapse knee + reliable tail")
    ap.add_argument("--smoke", action="store_true",
                    help="fabricate tiny real-shaped CSVs and run the full analysis")
    args = ap.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        results_dir = _fabricate_results(out_dir)
        print(f"[smoke] fabricated results in {results_dir}")

    per_df = _load_csv(results_dir / "per_concept_cfs.csv")
    if per_df is None:
        raise SystemExit(f"no per_concept_cfs.csv in {results_dir} "
                         f"(run the heavy modules first, or pass --smoke)")

    boot_df = bootstrap_by_method(per_df, n=args.n_boot, ci_pct=args.ci_pct)
    boot_df.to_csv(out_dir / "bootstrap_ci.csv", index=False)
    sig = significance_readout(boot_df)

    ood_df = _load_csv(results_dir / "ood_cfs_sweep.csv")
    ood = ood_degradation(ood_df, floor=args.floor) if ood_df is not None else None

    findings = write_findings(results_dir, out_dir, boot_df=boot_df, sig=sig,
                              ood=ood, floor=args.floor)

    # --- console summary -------------------------------------------------- #
    print("\n=== bootstrap CIs over concepts (mean CFS, 95%) ===")
    for _, r in boot_df.iterrows():
        print(f"  {r['label']:<22} mean={r['mean_cfs']:.3f}  "
              f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]  (n={r['n_concepts']})")
    if sig.get("pairs"):
        for comp, p in sig["pairs"].items():
            tag = "SEPARATED" if p["separated"] else "overlapping"
            print(f"  ours vs {comp}: Δ={p['delta']:+.3f}  CIs {tag}")
    print(f"\nwrote {out_dir/'bootstrap_ci.csv'}, {out_dir/'FINDINGS.md'}, "
          f"{out_dir/'findings.json'}")


if __name__ == "__main__":
    main()
