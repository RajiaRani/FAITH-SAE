"""probes.py — linear concept probes ("rulers") for the real FAITH-SAE run.

Author: Rajia Rani

WHAT THIS MODULE PROVIDES (and WHY)
===================================
The CFS faithfulness metric grades a steering edit by READING concepts off the
(steered) activations. Two kinds of reader live here:

  1. A per-concept LINEAR PROBE (scikit-learn ``LogisticRegression``) trained on
     REAL CLIP patch activations. A probe is a flat decision wall in the d=1024
     activation space whose signed distance is a smooth, calibrated readout of
     "how present is concept c?". The CFS *specificity* term watches the
     OFF-TARGET probes for drift while we steer the target — only the target
     concept should move (DESIGN_BRIEF Sec 7).

  2. A TCAV-style SUPERVISED CONCEPT DIRECTION (Kim et al. 2017): the target
     probe's weight vector, unit-normalised, IS the concept's direction in
     activation space. This is the strong, label-EXPENSIVE reference an
     unsupervised SAE decoder column hopes to match — the ``supervised`` quality
     ceiling that ``cfs_eval.evaluate_all_methods`` steers along (RQ1 baseline 5).

WHY LINEAR (not a deep readout)? A flat wall is the honest test of the steering
claim itself: steering asserts a concept is written along a single DIRECTION, so
the grader must read along a single direction too. A non-linear probe could
"find" a concept the linear edit can never move, which would flatter steering
dishonestly.

REAL-SCALE NOTES
================
* d_in defaults to CLIP ViT-L/14 width 1024 (ViT-B/16 = 768, ViT-H/14 = 1280);
  everything here is width-agnostic and reads the shape off the activations.
* Activations arrive as cached float16 patch-token banks (see data_real.py
  ACTIVATION CACHE FORMAT); we up-cast to float32 for the solver and standardise
  features so the L2-regularised logistic stays well-conditioned at d=1024.
* ``train_linear_probe`` honours the SHARED INTERFACE signature exactly
  (acts, labels) -> fitted ``LogisticRegression``; ``probe_readout(clf, acts)``
  returns the signed margin (decision_function), the calibrated concept readout.

This module imports cleanly on CPU with NO open_clip and NO GPU. It reuses the
project's ``src`` only for config loading; the probe math is standard sklearn.

For research and educational purposes only.
"""
from __future__ import annotations

# --- make the project root (for `from src...`) and this real_run dir importable.
# parents[2] = the 25_..._FAITH_SAE project root that contains src/.
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np


def _as_np(x, dtype=np.float32):
    """Coerce numpy arrays OR torch tensors (incl. grad/float16/CUDA) to a CPU
    numpy array. The real pipeline mixes cached numpy activations with torch
    tensors returned by the steerers, so the rulers accept either transparently."""
    if hasattr(x, "detach"):                       # a torch.Tensor
        x = x.detach().to("cpu")
        if hasattr(x, "numpy"):
            x = x.numpy()
    return np.asarray(x, dtype=dtype)


# =========================================================================== #
# 1. The core probe: train + read.  EXACT SHARED-INTERFACE signatures.        #
# =========================================================================== #
def train_linear_probe(acts, labels, C: float = 1.0, max_iter: int = 2000,
                       standardize: bool = True, seed: int = 0):
    """Fit ONE logistic-regression probe (a 'ruler' for one concept).

    Parameters
    ----------
    acts : array-like [n, d_in]
        Activations (CLIP patch tokens). float16 cache is accepted; we up-cast.
    labels : array-like [n]
        Binary presence labels for the concept (1 = present, 0 = absent).
    C : float
        Inverse L2 regularisation strength (smaller = stronger shrinkage). At
        d=1024 a little shrinkage keeps the weight vector — i.e. the TCAV
        direction — stable across resamples.
    standardize : bool
        Z-score the features before fitting. We fold the standardisation back
        INTO the linear weights afterwards, so the returned ``clf`` reads RAW
        activations directly (no separate scaler to carry around). This keeps the
        probe a single flat wall in the original activation space — which is what
        the steering edit lives in.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        A fitted probe whose ``coef_``/``intercept_`` operate on RAW activations.
        ``clf.concept_direction_`` (unit-norm weight) is attached for TCAV use.
    """
    from sklearn.linear_model import LogisticRegression

    X = _as_np(acts, np.float32)
    y = _as_np(labels, np.int64).reshape(-1)

    # Degenerate-label guard: a probe needs both classes. If the concept is all
    # present (or all absent) in this slice, return a harmless constant reader so
    # the pipeline never crashes — its readout is flat, so it simply contributes
    # no movement to specificity (correct behaviour for an undefined concept).
    if np.unique(y).size < 2:
        clf = _ConstantProbe(d=X.shape[1], const=float(y.mean() if y.size else 0.0))
        return clf

    if standardize:
        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-6
        Xz = (X - mu) / sd
    else:
        mu = np.zeros(X.shape[1], dtype=np.float32)
        sd = np.ones(X.shape[1], dtype=np.float32)
        Xz = X

    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed,
                             solver="lbfgs")
    clf.fit(Xz, y)

    # Fold standardisation back into the weights so the probe reads RAW acts:
    #   w_z . ((x - mu)/sd) + b_z  ==  (w_z/sd) . x + (b_z - w_z.(mu/sd))
    if standardize:
        w_z = clf.coef_.reshape(-1).astype(np.float64)
        b_z = float(clf.intercept_.reshape(-1)[0])
        w_raw = w_z / sd
        b_raw = b_z - float((w_z * (mu / sd)).sum())
        clf.coef_ = w_raw.reshape(1, -1).astype(np.float64)
        clf.intercept_ = np.array([b_raw], dtype=np.float64)

    # Attach the unit-norm TCAV-style direction (the arrow the concept lies along).
    w = clf.coef_.reshape(-1).astype(np.float32)
    clf.concept_direction_ = (w / (np.linalg.norm(w) + 1e-8)).astype(np.float32)
    return clf


def probe_readout(clf, acts) -> np.ndarray:
    """Return the probe's SIGNED MARGIN (decision_function) on ``acts`` [n, d].

    The margin ``w.x + b`` is the calibrated, continuous concept readout CFS uses
    (not the 0/1 label): bigger margin = more concept present. We deliberately do
    NOT squash through the sigmoid — the raw margin is monotone in the edit and
    keeps full dynamic range for the Spearman (monotonicity) and Cohen-d
    (sufficiency) statistics, whereas the sigmoid saturates and would flatten the
    effect size.

    Returns
    -------
    np.ndarray [n]  float32 readout, one number per activation row.
    """
    X = _as_np(acts, np.float32)
    return clf.decision_function(X).reshape(-1).astype(np.float32)


# =========================================================================== #
# 2. TCAV-style supervised concept direction (the quality reference).         #
# =========================================================================== #
def supervised_concept_direction(clf):
    """Unit-norm weight vector of a probe = its TCAV concept-activation vector.

    Kim et al. 2017: the direction a linear classifier uses to separate
    concept-present from concept-absent activations IS the concept direction. The
    target probe's direction is the *supervised* steering reference (baseline 5);
    cfs_eval steers along it as the label-expensive quality ceiling.
    """
    if hasattr(clf, "concept_direction_"):
        return np.asarray(clf.concept_direction_, dtype=np.float32)
    w = clf.coef_.reshape(-1).astype(np.float32)
    return (w / (np.linalg.norm(w) + 1e-8)).astype(np.float32)


def probe_accuracy(clf, acts, labels) -> float:
    """Held-in accuracy of a probe (sanity that the concept is linearly readable).

    Doubles as the A4 'well-defined concept' filter signal: a concept whose probe
    cannot beat chance is not a clean knob and should be dropped from selection.
    """
    if isinstance(clf, _ConstantProbe):
        return 0.5
    X = _as_np(acts, np.float32)
    y = _as_np(labels, np.int64).reshape(-1)
    return float(clf.score(X, y))


# =========================================================================== #
# 3. Probe BANK: one probe per (target + off-target) concept.                 #
# =========================================================================== #
class ProbeBank:
    """A collection of trained probes keyed by concept id, plus the target.

    cfs_eval reads the TARGET probe to score monotonicity/sufficiency and watches
    the OFF-TARGET probes for specificity drift. Holding them together (with a
    consistent concept ordering) keeps that controlled comparison clean.
    """

    def __init__(self, probes: dict, target_concept, directions: dict | None = None):
        self.probes = probes                      # concept_id -> fitted probe
        self.target_concept = target_concept
        # Cache each probe's unit direction (the TCAV vectors) for fast reuse.
        self.directions = directions or {
            cid: supervised_concept_direction(clf) for cid, clf in probes.items()
        }

    @property
    def concept_ids(self):
        return list(self.probes.keys())

    def off_target_ids(self, target=None):
        t = self.target_concept if target is None else target
        return [c for c in self.probes if c != t]

    def readout(self, concept_id, acts) -> np.ndarray:
        """Signed-margin readout of one concept's probe on ``acts``."""
        return probe_readout(self.probes[concept_id], acts)

    def target_direction(self, target=None) -> np.ndarray:
        t = self.target_concept if target is None else target
        return np.asarray(self.directions[t], dtype=np.float32)


def build_probe_bank(acts, concept_labels: dict, target_concept,
                     cfg: dict | None = None) -> ProbeBank:
    """Train one probe per concept and bundle them into a :class:`ProbeBank`.

    Parameters
    ----------
    acts : [n, d_in] activations shared by all concepts (one bank, many rulers).
    concept_labels : {concept_id: labels[n]} binary presence per concept.
    target_concept : which concept id is the steering TARGET.
    cfg : optional config; reads ``cfs``/probe knobs (C, max_iter) if present.
    """
    cfs_cfg = (cfg or {}).get("cfs", {}) if isinstance(cfg, dict) else {}
    C = float(cfs_cfg.get("probe_C", 1.0))
    max_iter = int(cfs_cfg.get("probe_max_iter", 2000))
    probes = {
        cid: train_linear_probe(acts, y, C=C, max_iter=max_iter)
        for cid, y in concept_labels.items()
    }
    return ProbeBank(probes, target_concept)


# =========================================================================== #
# 4. A constant fallback probe (used when a concept slice is single-class).    #
# =========================================================================== #
class _ConstantProbe:
    """Mimics the bits of LogisticRegression that probe_readout/accuracy need,
    but always reads a flat margin. Keeps the pipeline robust to degenerate
    concept slices (no crash, no spurious specificity movement)."""

    def __init__(self, d: int, const: float = 0.0):
        self.coef_ = np.zeros((1, d), dtype=np.float64)
        self.intercept_ = np.array([0.0], dtype=np.float64)
        self.const = float(const)
        self.concept_direction_ = np.zeros(d, dtype=np.float32)

    def decision_function(self, X):
        X = np.asarray(X)
        return np.full((X.shape[0],), self.const, dtype=np.float32)


# =========================================================================== #
# 5. Self-test: train probes on fabricated real-SHAPED CLIP activations.      #
# =========================================================================== #
def _smoke():
    """Fabricate real-shaped activations with planted concept directions, train a
    probe bank, and verify the rulers separate present/absent and recover the
    TCAV directions. CPU-only, no open_clip, no downloads."""
    rng = np.random.default_rng(0)
    d = 64                              # tiny stand-in for CLIP width (1024)
    n = 1200
    n_concepts = 4
    target = 0

    # Plant n_concepts orthogonal-ish concept directions; each item randomly has
    # a subset present, injected along its direction over a Gaussian background.
    dirs = rng.standard_normal((n_concepts, d)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    acts = rng.standard_normal((n, d)).astype(np.float32) * 0.5
    labels = (rng.random((n, n_concepts)) < 0.5).astype(np.int64)
    for c in range(n_concepts):
        acts += (labels[:, c:c + 1] * 3.0) * dirs[c][None, :]
    acts = acts.astype(np.float16)     # exercise the float16 cache path

    bank = build_probe_bank(
        acts,
        {c: labels[:, c] for c in range(n_concepts)},
        target_concept=target,
    )

    accs = {c: probe_accuracy(bank.probes[c], acts, labels[:, c])
            for c in range(n_concepts)}
    # Cosine of recovered TCAV direction vs the true planted direction.
    cos_tgt = float(abs(bank.target_direction() @ dirs[target]))
    rd = bank.readout(target, acts)
    sep = float(rd[labels[:, target] == 1].mean() - rd[labels[:, target] == 0].mean())

    print("[probes smoke] d=%d n=%d concepts=%d" % (d, n, n_concepts))
    print("  per-concept probe accuracy:", {c: round(a, 3) for c, a in accs.items()})
    print("  target TCAV |cos| vs planted dir: %.3f (want >0.6)" % cos_tgt)
    print("  target readout present-minus-absent margin: %.3f (want >0)" % sep)
    ok = (accs[target] > 0.8) and (cos_tgt > 0.6) and (sep > 0.0)
    print("[probes smoke] PASS" if ok else "[probes smoke] CHECK FAILED")
    return ok


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="FAITH-SAE real-run linear probes.")
    ap.add_argument("--smoke", action="store_true",
                    help="Run the tiny CPU self-test on fabricated activations.")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    else:
        # REAL PATH: probes are trained from cfs_eval/concept_select on cached
        # CLIP activations with real concept labels; there is no standalone real
        # entry point here. Run with --smoke for the offline check.
        print("probes.py: import this module (train_linear_probe / build_probe_bank). "
              "Use --smoke for the offline self-test.")
