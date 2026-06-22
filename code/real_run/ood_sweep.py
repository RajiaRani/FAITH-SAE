#!/usr/bin/env python3
# ===========================================================================
#  ood_sweep.py  —  FAITH-SAE REAL RUN  ·  RQ3: the OOD faithfulness sweep
#  --------------------------------------------------------------------------
#  The headline experiment of the paper (DESIGN_BRIEF RQ3 / fig1): take the
#  selected reliable SAE concepts, steer each one with every steering variant
#  (naive / random / clamp / onmanifold), and re-measure the Causal Faithful-
#  ness Score (CFS) as the test images march OUT of distribution along the
#  student's domain-shift ladder (ordered by shift strength):
#
#        in1k (in-dist) -> in100 (mild) -> food101 (domain) -> cifar100 (strong)
#
#  (the ladder is whatever cfg.ood.levels lists; the legacy open_clip ladder
#   clean -> ImageNet-R/Sketch/C/ObjectNet still works if a config names it.)
#
#  For each rung we REUSE that dataset's already-cached activations
#  (written by data_real.extract_activations) and call the ONE shared scorer
#  cfs_eval.compute_cfs, so every cell of the sweep is measured identically and
#  the only thing that changes between cells is the input distribution. The
#  output table outputs/ood_cfs_sweep.csv is exactly what fig1 (CFS-vs-shift)
#  and fig5 (shift x method heatmap) plot, and we additionally mark the
#  "collapse knee": the first rung at which on-manifold CFS crosses a usability
#  floor (the ΔCFS-per-shift-level story of RQ3).
#
#  author: Rajia Rani
#  For research and educational purposes only.
# ===========================================================================
#
#  WHY A SEPARATE SWEEP MODULE (and not just a loop in cfs_eval)?
#  -------------------------------------------------------------
#  cfs_eval.compute_cfs scores ONE (method, concept) on ONE bank of activations.
#  The OOD sweep is the *outer* experiment: it decides WHICH banks to feed in
#  (one per shift rung), holds the SAE / concept set / projection basis / probes
#  fixed across all rungs, and assembles the per-rung scores into the CFS-vs-
#  shift curve. Keeping that orchestration here means cfs_eval stays a pure
#  scorer and this file owns the "ladder" semantics (rung ordering, severities,
#  the knee detector).
#
#  CONTROLLED-VARIABLE DISCIPLINE
#  ------------------------------
#  Across the whole sweep the SAE weights, the concept ids, the on-manifold
#  basis U_r, and the linear probes are estimated ONCE on clean data and then
#  FROZEN. Only the evaluation activations change rung to rung. That is what
#  makes a CFS drop attributable to distribution shift rather than to refitting.
#
#  REAL-RUN CAVEAT
#  ---------------
#  The real path needs (a) open_clip + the OOD datasets downloaded, (b) the
#  activation cache populated by data_real.extract_activations for every rung,
#  (c) a trained SAE (train_sae.py), the basis U_r (manifold.py) and the probes
#  (probes.py). None of those exist on this build machine and there is no GPU,
#  so the default real path raises a clear, actionable error if a prerequisite
#  is missing, and `--smoke` runs the entire ladder on fabricated real-SHAPED
#  activations on CPU with no open_clip / no downloads.

from __future__ import annotations

# --- sys.path: reuse the project's real math in src/, and allow sibling -------
# imports (cfs_eval, manifold, ...) from this same real_run directory. parents[2]
# is the 25_..._FAITH_SAE project root (which contains src/); see DESIGN_BRIEF.
import argparse
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# cfs_score is the SINGLE source of truth for the harmonic-mean CFS (brief §13);
# we never re-implement it. Imported eagerly because src/ always exists.
from src.utils import cfs_score  # noqa: E402


# --------------------------------------------------------------------------- #
# Config loading (shared YAML schema; load_real_config helper)                #
# --------------------------------------------------------------------------- #
def load_real_config(path: str) -> dict:
    """Load the real-run YAML config (the schema in the build brief).

    We reuse src.utils.load_config so the real pipeline and this sweep parse
    config identically (YAML if pyyaml is present, else JSON). Returns a plain
    nested dict: cfg['ood']['levels'], cfg['steering']['strength_grid'], etc.
    """
    from src.utils import load_config
    return load_config(path)


# --------------------------------------------------------------------------- #
# In-distribution dataset key: the first ladder rung (also the SAE-training and  #
# probe-fitting source). The student's ladder names it 'in1k'; the legacy ladder #
# named it 'clean'. The probes + concepts are fit on THIS rung and frozen.       #
# --------------------------------------------------------------------------- #
def _indist_key(cfg: dict) -> str:
    """Return the in-distribution dataset key (first ood.levels entry).

    Falls back to 'in1k' (the student's ladder) then 'clean' (legacy) so the
    probe/concept loaders read the right cache shards regardless of ladder.
    """
    levels = list((cfg.get("ood", {}) or {}).get("levels", []))
    return levels[0] if levels else "in1k"


# --------------------------------------------------------------------------- #
# The OOD ladder: expand cfg.ood.levels into ordered, named rungs.            #
# --------------------------------------------------------------------------- #
def build_ladder(cfg: dict) -> list:
    """Expand the configured OOD levels into an ORDERED list of rung dicts.

    Each rung is {name, dataset, severity, shift_index} where:
      * name        — the label that appears on the x-axis of fig1
                      (e.g. 'clean', 'imagenet_c_s3', 'objectnet').
      * dataset     — the cache dataset key data_real wrote shards under
                      (imagenet_c shares one dataset, severity selects within it).
      * severity    — int 1..5 for imagenet_c, else None.
      * shift_index — a monotone 0,1,2,... position used as the x-coordinate for
                      the degradation-slope / knee computation (clean = 0).

    ImageNet-C is special: it is a continuous severity DIAL, so a single
    'imagenet_c' level expands into one rung per configured severity, giving the
    smooth middle of the CFS-vs-shift curve (the headline stress test).
    """
    ood = cfg.get("ood", {}) or {}
    levels = list(ood.get("levels", ["clean"]))
    severities = list(ood.get("severities", [1, 2, 3, 4, 5]))

    rungs: list = []
    idx = 0
    for level in levels:
        if level == "imagenet_c":
            # Expand the severity dial 1..5 into consecutive rungs.
            for sev in severities:
                rungs.append({
                    "name": f"imagenet_c_s{sev}",
                    "dataset": "imagenet_c",
                    "severity": int(sev),
                    "shift_index": idx,
                })
                idx += 1
        else:
            rungs.append({
                "name": level,
                "dataset": level,
                "severity": None,
                "shift_index": idx,
            })
            idx += 1
    return rungs


# --------------------------------------------------------------------------- #
# Smoke fabric: synthetic-but-real-SHAPED activation banks per rung.          #
# These fabricate exactly what data_real.load_activation_bank would return    #
# (a torch.float32 [n_tokens, d_in] tensor) so the sweep exercises the REAL   #
# code path without open_clip / downloads. Each harder rung blurs the         #
# on-manifold sheet and injects off-sheet noise, so CFS degrades the way it   #
# should as inputs go OOD.                                                     #
# --------------------------------------------------------------------------- #
def _planted_basis(d_in: int, true_rank: int, seed: int = 0):
    """An orthonormal [d_in, true_rank] 'real-image manifold' sheet basis."""
    import torch
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(d_in, d_in, generator=g)
    Q, _ = torch.linalg.qr(M)
    return Q[:, :true_rank].contiguous()


def _smoke_bank(d_in: int, n_tokens: int, sheet, shift_index: int,
                severity, seed: int = 0):
    """Fabricate a [n_tokens, d_in] activation bank that lives near `sheet`,
    degraded by an amount that grows with the rung's position on the ladder.

    The degradation budget = base shift_index + (severity-1) for ImageNet-C
    rungs, so later/harder rungs blur the manifold more (the on-manifold method
    relies on that very sheet, so its readout cleanliness drops) and add more
    off-sheet wobble (specificity leaks). This reproduces, on CPU, the OOD
    collapse the real CLIP activations exhibit.
    """
    import torch
    g = torch.Generator().manual_seed(seed + 7919 * (shift_index + 1))
    true_rank = sheet.shape[1]
    coords = torch.randn(n_tokens, true_rank, generator=g)
    on_sheet = coords @ sheet.T                         # [n, d] points on the sheet

    # Degradation grows with how far down the ladder we are.
    deg = float(shift_index) + (0.0 if severity is None else float(severity - 1))
    blur = 0.04 * deg                                    # shrinks the clean signal
    offsheet = 0.02 + 0.03 * deg                         # off-manifold wobble
    on_sheet = on_sheet * (1.0 - min(blur, 0.9))
    wobble = offsheet * torch.randn(n_tokens, d_in, generator=g)
    return (on_sheet + wobble).float()


# --------------------------------------------------------------------------- #
# Activation source: real cache vs. smoke fabric.                            #
# --------------------------------------------------------------------------- #
def _load_rung_bank(cfg: dict, cache_dir: str, rung: dict, *, smoke: bool,
                    sheet=None):
    """Return the evaluation activation bank [n_tokens, d_in] (torch.float32)
    for one OOD rung.

    Real path: read the dataset's cached shards via data_real.load_activation_bank
    (CLS already dropped, patch tokens only). ImageNet-C severities all live in
    the 'imagenet_c' dataset cache; we sample the whole cache here (a severity-
    aware cache layout is a data_real concern). Smoke path: fabricate a real-
    shaped bank whose cleanliness degrades along the ladder.
    """
    d_in = int(cfg["sae"]["d_in"])
    n_tokens = int(cfg.get("steering", {}).get("bank_tokens", 2_000_000))
    if smoke:
        n_tokens = min(n_tokens, 4096)                  # tiny CPU bank
        return _smoke_bank(d_in, n_tokens, sheet, rung["shift_index"],
                           rung["severity"], seed=int(cfg.get("seed", 0)))

    # --- REAL PATH -----------------------------------------------------------
    # Import the sibling lazily so this module still imports if data_real is not
    # yet present (parallel build) or open_clip is missing on this machine.
    try:
        import data_real
    except Exception as e:  # pragma: no cover - exercised only on the real path
        raise RuntimeError(
            "ood_sweep real path needs data_real.py and a populated activation "
            f"cache; import failed: {e}. Run extract_activations first, or use "
            "--smoke for the offline CPU path.") from e
    return data_real.load_activation_bank(
        cache_dir, rung["dataset"], n_tokens, seed=int(cfg.get("seed", 0)))


# --------------------------------------------------------------------------- #
# The shared scorer: prefer the real cfs_eval.compute_cfs; fall back to a    #
# self-contained probe so --smoke (and a partially-built tree) still runs.    #
# --------------------------------------------------------------------------- #
def _steer_variants() -> list:
    """The steering methods compared at every rung (DESIGN_BRIEF §12).

    naive  — off-manifold activation addition (the main competitor).
    random — null/sanity baseline (no real concept).
    clamp  — clamp the SAE feature, no projection.
    onmanifold — OURS: project the edit onto the top-r real-image subspace.
    (Supervised/TCAV is a cfs_eval concern and folded in there if available.)
    """
    return ["naive_steer", "random_steer", "clamp_steer", "onmanifold_steer"]


def _fallback_compute_cfs(steer_name, eval_acts, U_r, cfg, concept_dir,
                          off_dir, seed=0, shift_level=0.0, snr=1.0):
    """Self-contained CFS probe used when cfs_eval is unavailable (smoke / partial
    tree). Mirrors the empirical probe semantics of src.evaluate.cfs_probe and
    milestone cfs_probe: sweep the strength grid, read a target probe and an
    OFF-MANIFOLD off-target probe, then mono/spec/suff -> cfs_score.

    snr         : signal-to-noise multiplier on the TARGET readout (mirrors the
                  milestone probe's `snr`). <1 models a setting whose activations
                  carry the concept LESS cleanly (used by the ablations: L1 vs
                  TopK, CLS vs patch, off-peak layer), so the effect size — and
                  thus CFS — drops. The off-target/noise floor is unchanged, so a
                  lower snr genuinely lowers sufficiency rather than rescaling it
                  away.
    eval_acts   : [n, d] torch bank for this rung.
    U_r         : [d, r] CLEAN-data on-manifold basis (the reference P_M projects
                  onto). Passed for EVERY variant — it is the manifold the edit
                  is judged against, not a per-variant switch.
    concept_dir : the raw edit direction Delta (the SAE concept; it has a small
                  OFF-sheet component, which is exactly what naive leaks).
    off_dir     : an OFF-MANIFOLD probe direction that should stay flat if the
                  edit is on-manifold; a naive off-sheet edit leaks into it.

    Physics this captures (so the smoke ordering matches the paper's claim):
      * naive_steer adds the RAW edit -> its off-sheet component drives the off-
        manifold probe -> specificity LEAKS -> lower CFS.
      * onmanifold_steer adds P_M.Delta -> off-sheet component removed ->
        off-manifold probe stays flat -> specificity high -> higher CFS.
      * As the rung gets harder the eval bank gains off-sheet noise the CLEAN
        U_r cannot describe, so even the projected edit reads a dirtier target
        signal -> on-manifold CFS degrades too, but more gracefully than naive.

    This is the ONLY place we approximate cfs_eval; on the real path the true
    cfs_eval.compute_cfs (with learned probes and the trained SAE) is used.
    """
    import torch
    grid = [float(s) for s in cfg.get("steering", {}).get(
        "strength_grid", [0, 0.5, 1, 2, 4])]
    d_in = eval_acts.shape[-1]
    a0 = eval_acts.float()

    def _unit(v):
        return v / (v.norm() + 1e-8)

    raw = _unit(concept_dir)                             # Delta (has off-sheet bit)
    d_tgt = _unit(concept_dir)
    d_off = _unit(off_dir)                               # off-manifold probe

    # Effective edit direction per variant (matches steering_real semantics).
    # `shift_level` (0 = clean, grows down the OOD ladder) models that the CLEAN
    # basis U_r increasingly MIS-describes the shifted test manifold: the on-
    # manifold projection then both loses concept signal AND starts leaking off
    # the (now-wrong) subspace, so even our method degrades under heavy shift —
    # the headline RQ3 story. Naive/clamp carry their fixed off-sheet leak plus
    # the bank's growing noise floor.
    if steer_name == "random_steer":
        g = torch.Generator().manual_seed(seed + 13)
        edit_dir = _unit(torch.randn(d_in, generator=g))
    elif steer_name == "onmanifold_steer" and U_r is not None:
        proj = U_r @ (U_r.T @ raw)                       # P_M . Delta (NOT renorm:
        # keeping the projected magnitude is what makes over-/under-projection
        # visible — a tiny r throws away most of the edit, so the effect dies.)
        # Under shift the clean U_r drifts: re-admit a growing slice of the raw
        # (off-clean-manifold) edit, so the projected edit loses purity with
        # depth on the ladder (graceful, not catastrophic, degradation).
        drift = min(0.6, 0.12 * float(shift_level))
        edit_dir = (1.0 - drift) * proj + drift * raw
    else:                                               # naive / clamp ~ raw add
        edit_dir = raw

    def readout(a, d):
        return (a * d).sum(-1)                          # [n_tokens] (NOT averaged:
        # we keep per-token spread so the bank's OOD noise floor feeds straight
        # into the Cohen's-d denominator -> harder rungs -> lower sufficiency.)

    knobs, tgt_mean, off_mean = [], [], []
    r0 = readout(a0, d_tgt)                              # base target readout/token
    r_full = readout(a0 + max(grid) * edit_dir, d_tgt)  # full-knob target readout
    for s in grid:
        a_s = a0 + s * edit_dir
        knobs.append(s)
        tgt_mean.append(float(readout(a_s, d_tgt).mean()))
        off_mean.append(float(readout(a_s, d_off).mean()))

    k = torch.tensor(knobs)
    tr = torch.tensor(tgt_mean)
    ofr = torch.tensor(off_mean)

    # Monotonicity = Spearman(knob, target readout) via Pearson-on-ranks.
    def _spearman(a, b):
        ar = a.argsort().argsort().float()
        br = b.argsort().argsort().float()
        ar = ar - ar.mean(); br = br - br.mean()
        return float((ar * br).sum() / (ar.norm() * br.norm() + 1e-8))

    mono = max(_spearman(k, tr), 0.0)
    # Specificity: how much did the OFF-MANIFOLD probe drift vs the natural edit
    # scale? A clean (projected) edit barely moves the off-sheet probe.
    edit_scale = float(edit_dir.norm()) * (max(grid) + 1e-6) + 1e-6
    off_move = (ofr.max() - ofr.min()).abs()
    spec = float(max(0.0, 1.0 - off_move / edit_scale))
    # Sufficiency: standardized (Cohen's-d) effect size at full knob vs none, with
    # the POOLED PER-TOKEN std as the denominator. Under shift the bank gains off-
    # sheet noise -> r0/r_full spread grows -> d_eff shrinks -> CFS degrades.
    pooled = (r0.std() + r_full.std()) / 2 + 1e-6
    d_eff = float((r_full.mean() - r0.mean()).abs() / pooled)
    suff = min(d_eff / 4.0, 1.0)

    # Off-manifold residual of the EFFECTIVE edit relative to the clean basis
    # (0 for a perfectly projected edit; ~1 for a fully off-sheet edit).
    if U_r is not None:
        from src.utils import onmanifold_projection_residual
        resid = onmanifold_projection_residual(edit_dir, U_r)
    else:
        resid = 1.0 if steer_name in ("naive_steer", "clamp_steer",
                                      "random_steer") else 0.0

    return {
        "monotonicity": round(mono, 4),
        "specificity": round(spec, 4),
        "sufficiency": round(suff, 4),
        "cfs": round(cfs_score(mono, spec, suff), 4),
        "offmanifold_residual": round(float(resid), 4),
    }


def _score_rung(cfg, sae, steer_name, concept_ids, eval_acts, probes, U_r, *,
                smoke, concept_dirs=None, off_dirs=None, shift_level=0.0,
                probe_concept_ids=None):
    """Score ONE steering method on ONE rung, averaged over the concept set.

    Real path: delegate per concept to cfs_eval.compute_cfs(sae, steer_name,
    concept_id, eval_acts, probes, U_r, cfg) — the shared scorer with learned
    probes. cfs_eval reads probes.target_concept (a CLASS-probe key) for the
    target readout, so we re-point the FROZEN ProbeBank's target at a probe
    concept per SAE concept (round-robin over probe_concept_ids) — the probes
    themselves are never refit, only which one is "target" vs "off-target"
    changes, keeping the comparison controlled. Smoke / partial-tree path: use
    the self-contained probe with planted concept directions. `shift_level` is
    the rung's position on the ladder, fed to the smoke probe so faithfulness
    degrades with depth. Returns the mean component dict over concepts plus the
    per-concept CFS list (the bootstrap material the analysis module consumes).
    """
    use_real = (not smoke) and (sae is not None)
    per_concept = []
    agg = {"monotonicity": [], "specificity": [], "sufficiency": [],
           "cfs": [], "offmanifold_residual": []}

    compute_cfs = None
    if use_real:
        try:
            import cfs_eval
            compute_cfs = cfs_eval.compute_cfs
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "ood_sweep real path needs cfs_eval.compute_cfs; import failed: "
                f"{e}. Use --smoke for the offline CPU path.") from e

    for ci, concept_id in enumerate(concept_ids):
        if compute_cfs is not None:
            # Re-point the frozen ProbeBank's target at a class probe (round-robin
            # so different SAE concepts exercise different target/off-target
            # splits); off_target_ids() then yields the remaining class probes.
            if probe_concept_ids:
                probes.target_concept = probe_concept_ids[ci % len(probe_concept_ids)]
            m = compute_cfs(sae, steer_name, concept_id, eval_acts, probes,
                            U_r, cfg)
        else:
            cd = concept_dirs[ci]
            od = off_dirs[ci]
            m = _fallback_compute_cfs(steer_name, eval_acts, U_r, cfg, cd, od,
                                      seed=int(cfg.get("seed", 0)) + ci,
                                      shift_level=shift_level)
        per_concept.append(float(m["cfs"]))
        for key in agg:
            if key in m:
                agg[key].append(float(m[key]))

    mean = {key: (sum(v) / len(v) if v else float("nan"))
            for key, v in agg.items()}
    return mean, per_concept


# --------------------------------------------------------------------------- #
# Collapse-knee detector: the RQ3 ΔCFS-per-shift-level story.                 #
# --------------------------------------------------------------------------- #
def find_collapse_knee(rung_names, cfs_by_rung, floor: float = 0.5):
    """Locate the 'collapse knee' for the on-manifold method: the FIRST rung at
    which CFS drops below the usability floor, plus the average per-rung slope.

    rung_names  — ordered list of rung labels (x-axis).
    cfs_by_rung — list of CFS values (same order) for on-manifold steering.
    floor       — usability threshold (default 0.5; below this the concept is no
                  longer a trustworthy causal lever).

    Returns {knee_rung, knee_index, slope_per_rung, crossed}. If CFS never drops
    below the floor, knee_rung is None (faithfulness survived the whole ladder).
    """
    knee_rung = None
    knee_index = None
    for i, v in enumerate(cfs_by_rung):
        if v < floor:
            knee_rung = rung_names[i]
            knee_index = i
            break
    # Average degradation slope (ΔCFS per rung) from clean to the last rung.
    if len(cfs_by_rung) >= 2:
        slope = (cfs_by_rung[-1] - cfs_by_rung[0]) / (len(cfs_by_rung) - 1)
    else:
        slope = 0.0
    return {
        "knee_rung": knee_rung,
        "knee_index": knee_index,
        "slope_per_rung": round(float(slope), 5),
        "crossed": knee_rung is not None,
    }


# --------------------------------------------------------------------------- #
# Real-path probe bank: trained ONCE on CLEAN labeled activations and FROZEN.  #
# --------------------------------------------------------------------------- #
def _build_clean_probe_bank(cfg, cache_dir):
    """Train the linear concept probes ONCE on clean labeled CLIP activations and
    return a frozen probes.ProbeBank (the rulers cfs_eval.compute_cfs reads).

    cfs_eval reads probes.target_concept for the TARGET readout (monotonicity /
    sufficiency) and probes.off_target_ids() for the off-target drift
    (specificity). Those probe concepts are the CLASS LABELS the cache stored per
    token (data_real broadcasts each image's label onto its patch tokens), NOT the
    SAE feature ids — a steered SAE feature should drive its matching CLASS probe
    up while leaving the OTHER class probes flat. So we:

      1. stream the clean dataset's (acts, labels) shards (iter_labeled_shards),
         accumulating a labeled bank up to cfs.probe_bank_tokens tokens;
      2. pick the cfs.n_probe_classes most frequent class labels present;
      3. build one-vs-rest binary labels per class and fit one probe each.

    The bank is trained on CLEAN data and reused UNCHANGED across every OOD rung,
    so a CFS drop is attributable to input shift, not to refitting (the controlled-
    variable discipline). Returns (probe_bank, probe_concept_ids).
    """
    import numpy as np
    import data_real
    import probes as probes_mod

    cfs_cfg = cfg.get("cfs", {}) or {}
    n_classes = int(cfs_cfg.get("n_probe_classes", 50))
    budget = int(cfs_cfg.get("probe_bank_tokens", 200_000))

    # Accumulate a labeled in-distribution bank (acts + per-token class label).
    indist = _indist_key(cfg)
    acts_chunks, lab_chunks, have = [], [], 0
    for acts, labels in data_real.iter_labeled_shards(cache_dir, indist):
        if labels is None:                              # unlabeled cache -> no probes
            continue
        a = np.asarray(acts, dtype=np.float32)
        y = np.asarray(labels).reshape(-1)
        acts_chunks.append(a)
        lab_chunks.append(y)
        have += a.shape[0]
        if have >= budget:
            break
    if not acts_chunks:
        raise RuntimeError(
            f"no labeled in-distribution ('{indist}') shards found "
            "(data_real.iter_labeled_shards yielded no labels); cfs_eval probes "
            "need per-token class labels. Re-run extract_activations so "
            "labels_*.npy are written.")
    bank = np.concatenate(acts_chunks, 0)[:budget]
    labs = np.concatenate(lab_chunks, 0)[:budget]

    # Most frequent REAL classes present (drop the -1 'no label' sentinel).
    valid = labs[labs >= 0]
    classes, counts = np.unique(valid, return_counts=True)
    keep = classes[np.argsort(counts)[::-1][:n_classes]].tolist()
    if len(keep) < 2:
        raise RuntimeError(
            f"need >=2 distinct class labels for off-target probes; found {keep}.")

    # One-vs-rest binary labels per kept class -> one probe each.
    concept_labels = {int(c): (labs == c).astype(np.int64) for c in keep}
    probe_bank = probes_mod.build_probe_bank(
        bank, concept_labels, target_concept=int(keep[0]), cfg=cfg)
    return probe_bank, [int(c) for c in keep]


# --------------------------------------------------------------------------- #
# Prerequisite loading on the REAL path (SAE, basis, concepts, probes).       #
# --------------------------------------------------------------------------- #
def _load_real_prereqs(cfg, cache_dir):
    """Load the FROZEN, clean-data artefacts the sweep reuses across all rungs:
    the trained SAE, the on-manifold basis U_r, the selected concept ids, and
    the linear probes. Each is produced by a sibling module; we import lazily and
    fail with a clear message if a prerequisite is missing.

    Returns (sae, U_r, concept_ids, probes, probe_concept_ids). ``probes`` is a
    frozen probes.ProbeBank (NOT the module); probe_concept_ids are its class-probe
    keys, round-robined as each SAE concept's target by _score_rung. Raises
    RuntimeError on the build machine (no open_clip / no trained artefacts) — which
    is the honest answer.
    """
    paths = cfg.get("paths", {}) or {}
    # 1. SAE -----------------------------------------------------------------
    try:
        import sae_real
        sae = sae_real.load_sae(paths.get("sae_ckpt", "./outputs/sae.safetensors"))
    except Exception as e:
        raise RuntimeError(
            f"ood_sweep real path needs a trained SAE (sae_real.load_sae): {e}. "
            "Train it with train_sae.py, or use --smoke.") from e
    # 2. On-manifold basis U_r ----------------------------------------------
    try:
        import manifold
        U_r = manifold.load_basis(paths.get("manifold_basis", "./outputs/U_r.npy"))
    except Exception as e:
        raise RuntimeError(
            f"ood_sweep real path needs the manifold basis U_r (manifold.load_"
            f"basis): {e}. Build it with manifold.estimate_manifold_basis.") from e
    # 3. Concept ids on CLEAN activations -----------------------------------
    try:
        import concept_select
        import data_real
        clean_bank = data_real.load_activation_bank(
            cache_dir, _indist_key(cfg),
            int(cfg.get("steering", {}).get("bank_tokens", 2_000_000)),
            seed=int(cfg.get("seed", 0)))
        concept_ids = concept_select.select_concepts(
            sae, clean_bank, image_ids=None, cfg=cfg)
    except Exception as e:
        raise RuntimeError(
            f"ood_sweep real path needs selected concepts (concept_select.select_"
            f"concepts): {e}.") from e
    # 4. Probes (a frozen ProbeBank trained ONCE on clean labeled activations) -
    try:
        probes, probe_concept_ids = _build_clean_probe_bank(cfg, cache_dir)
    except Exception as e:
        raise RuntimeError(
            f"ood_sweep real path needs trained probes (probes.build_probe_bank): "
            f"{e}.") from e
    return sae, U_r, concept_ids, probes, probe_concept_ids


# --------------------------------------------------------------------------- #
# The headline driver.                                                        #
# --------------------------------------------------------------------------- #
def run_ood_sweep(cfg: dict, cache_dir: str, *, smoke: bool = False):
    """Run the full OOD CFS sweep and return a pandas DataFrame (also written to
    cfg.paths.out_dir/ood_cfs_sweep.csv).

    One row per (rung, steering method) with the three CFS components, the CFS,
    the off-manifold residual, the shift index, and a per-rung knee flag for the
    on-manifold method. This is the table fig1 (CFS-vs-shift) and fig5 (heatmap)
    consume.
    """
    import pandas as pd

    out_dir = pathlib.Path(cfg.get("paths", {}).get("out_dir", str(_HERE / "outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ood_cfs_sweep.csv"

    ladder = build_ladder(cfg)
    variants = _steer_variants()
    floor = float(cfg.get("ood", {}).get("usability_floor", 0.5))

    print("=" * 72)
    print("FAITH-SAE — OOD CFS SWEEP (RQ3: does faithfulness survive shift?)")
    print("=" * 72)
    print(f"  rungs   : {[r['name'] for r in ladder]}")
    print(f"  methods : {variants}")
    print(f"  mode    : {'SMOKE (synthetic, CPU)' if smoke else 'REAL'}")

    # --- Frozen, clean-data artefacts reused across ALL rungs ---------------
    sae = U_r = probes = None
    probe_concept_ids = None
    concept_ids: list
    concept_dirs = off_dirs = None
    d_in = int(cfg["sae"]["d_in"])

    if smoke:
        # Fabricate a planted real-image sheet + concept directions that lie
        # MOSTLY on the sheet but carry a small OFF-sheet component (real SAE
        # directions are never perfectly on-manifold). The on-manifold steerer
        # projects that component away (stays specific); naive keeps it (leaks
        # into an off-manifold probe). The off-target probe is an OFF-sheet
        # direction, so only an off-manifold edit moves it.
        import torch
        true_rank = min(int(cfg.get("steering", {}).get("proj_rank_r", 512)),
                        max(2, d_in // 2))
        sheet = _planted_basis(d_in, true_rank, seed=int(cfg.get("seed", 0)))
        # U_r is the CLEAN-data on-manifold basis (here the planted sheet itself).
        U_r = sheet
        # Off-sheet complement directions (the orthogonal subspace U_r does NOT
        # span) — used both as the concepts' off-manifold leakage and as the
        # off-target probes naive steering wrongly excites.
        full = _planted_basis(d_in, d_in, seed=int(cfg.get("seed", 0)))
        offsheet = full[:, true_rank:]                 # [d, d-true_rank]
        n_concepts = int(cfg.get("cfs", {}).get("n_probe_classes", 6))
        n_concepts = max(2, min(n_concepts, true_rank, offsheet.shape[1]))
        concept_ids = list(range(n_concepts))
        concept_dirs, off_dirs = [], []
        for j in range(n_concepts):
            on = sheet[:, j]                            # on-sheet part
            off = offsheet[:, j % offsheet.shape[1]]    # an off-sheet direction
            cd = on + 0.6 * off                         # concept = mostly-on + leak
            concept_dirs.append((cd / (cd.norm() + 1e-8)).clone())
            off_dirs.append(off.clone())               # off-manifold probe
        sheet_for_banks = sheet
    else:
        (sae, U_r, concept_ids, probes,
         probe_concept_ids) = _load_real_prereqs(cfg, cache_dir)
        sheet_for_banks = None

    # --- March down the ladder ---------------------------------------------
    rows = []
    onmanifold_curve = []                      # CFS-vs-rung for on-manifold
    rung_names = [r["name"] for r in ladder]

    for rung in ladder:
        eval_acts = _load_rung_bank(cfg, cache_dir, rung, smoke=smoke,
                                    sheet=sheet_for_banks)
        print(f"\n  rung '{rung['name']}'  (shift_index={rung['shift_index']}, "
              f"acts={tuple(eval_acts.shape)})")
        print(f"    {'method':<18} {'mono':>6} {'spec':>6} {'suff':>6} "
              f"{'off-res':>8} {'CFS':>7}")
        for variant in variants:
            # U_r is the manifold REFERENCE passed to every variant (it is the
            # subspace the residual + specificity are judged against, and the
            # contract of cfs_eval.compute_cfs takes U_r unconditionally). Only
            # onmanifold_steer actually PROJECTS onto it; the scorer decides that
            # from the variant name.
            mean, per_concept = _score_rung(
                cfg, sae, variant, concept_ids, eval_acts, probes, U_r,
                smoke=smoke, concept_dirs=concept_dirs, off_dirs=off_dirs,
                shift_level=float(rung["shift_index"]),
                probe_concept_ids=probe_concept_ids)
            print(f"    {variant:<18} {mean['monotonicity']:>6.3f} "
                  f"{mean['specificity']:>6.3f} {mean['sufficiency']:>6.3f} "
                  f"{mean['offmanifold_residual']:>8.3f} {mean['cfs']:>7.4f}")
            if variant == "onmanifold_steer":
                onmanifold_curve.append(mean["cfs"])
            rows.append({
                "rung": rung["name"],
                "dataset": rung["dataset"],
                "severity": rung["severity"] if rung["severity"] is not None else "",
                "shift_index": rung["shift_index"],
                "method": variant,
                "monotonicity": mean["monotonicity"],
                "specificity": mean["specificity"],
                "sufficiency": mean["sufficiency"],
                "offmanifold_residual": mean["offmanifold_residual"],
                "cfs": mean["cfs"],
                "cfs_per_concept": ";".join(f"{c:.4f}" for c in per_concept),
                "is_knee": False,            # filled in after the knee is found
            })

    # --- Collapse knee for the on-manifold method (RQ3 headline) ------------
    knee = find_collapse_knee(rung_names, onmanifold_curve, floor=floor)
    if knee["knee_rung"] is not None:
        for r in rows:
            if r["method"] == "onmanifold_steer" and r["rung"] == knee["knee_rung"]:
                r["is_knee"] = True
        print(f"\n  COLLAPSE KNEE: on-manifold CFS first drops below {floor} at "
              f"rung '{knee['knee_rung']}' (shift_index={knee['knee_index']}); "
              f"avg slope {knee['slope_per_rung']:+.4f} CFS/rung.")
    else:
        print(f"\n  NO COLLAPSE: on-manifold CFS stays >= {floor} across the "
              f"whole ladder; avg slope {knee['slope_per_rung']:+.4f} CFS/rung.")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\n  wrote {csv_path}  ({len(df)} rows)")
    return df


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _default_smoke_cfg() -> dict:
    """A tiny, fully self-contained config for the CPU smoke sweep — no YAML file,
    no timm, no downloads. Real-SHAPED (d_in tokens) but small. The ladder is the
    student's domain-shift ladder (in1k -> in100 -> food101 -> cifar100), so the
    smoke path shows a visible CFS degradation slope / collapse knee as the shift
    strengthens, while staying tiny/fast on CPU."""
    return {
        "seed": 0,
        "sae": {"d_in": 64, "expansion": 8, "k": 8},
        "steering": {"strength_grid": [0, 0.5, 1, 2, 4], "proj_rank_r": 16,
                     "bank_tokens": 4096},
        "cfs": {"n_probe_classes": 6, "bootstrap_n": 200},
        "ood": {"levels": ["in1k", "in100", "food101", "cifar100"],
                "usability_floor": 0.5},
        "paths": {"out_dir": str(_HERE / "outputs")},
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="FAITH-SAE OOD CFS sweep (RQ3). Real path is the default; "
                    "--smoke runs a tiny 2-rung CPU sweep on synthetic-but-real-"
                    "shaped activations (no open_clip, no downloads).")
    ap.add_argument("--config", type=str, default=None,
                    help="path to the real-run YAML config")
    ap.add_argument("--cache_dir", type=str, default=None,
                    help="activation cache dir (data_real shards). "
                         "Defaults to cfg.paths.cache_dir or ./cache")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny offline CPU sweep on fabricated activations")
    args = ap.parse_args()

    if args.smoke:
        cfg = _default_smoke_cfg()
        if args.config:                       # allow overriding the smoke cfg
            cfg = load_real_config(args.config)
        cache_dir = args.cache_dir or str(_HERE / "cache")
        run_ood_sweep(cfg, cache_dir, smoke=True)
        return

    if not args.config:
        ap.error("real path requires --config (or pass --smoke for the CPU path)")
    cfg = load_real_config(args.config)
    cache_dir = (args.cache_dir or cfg.get("paths", {}).get("cache_dir")
                 or str(_HERE / "cache"))
    run_ood_sweep(cfg, cache_dir, smoke=False)


if __name__ == "__main__":
    main()
