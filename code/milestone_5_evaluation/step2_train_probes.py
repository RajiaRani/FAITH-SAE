"""step2_train_probes.py — train one LINEAR PROBE per concept (+ estimate U_r).

==============================================================================
WHAT THIS STEP DOES (in one sentence)
==============================================================================
It trains a small linear "ruler" (a LogisticRegression probe) for EACH concept,
so later we can read every concept's level off any activation; it also PCAs the
bank to get the on-manifold subspace U_r that the on-manifold steerer needs.

==============================================================================
TEACH-FROM-ZERO: every term used below, defined before it is used
==============================================================================

A PROBE = A RULER THAT READS ONE CONCEPT OFF AN ACTIVATION
  A "probe" is a tiny model that looks at one activation (64 numbers) and reports
  ONE number: "how much of concept C is in here?" Think of it as a RULER built
  for a single concept — hold it up to any activation and it reads that concept's
  level, ignoring the others. We train one ruler per concept.
  Tiny number: a probe for "stripey" might output 0.95 ("very stripey") on a
  zebra-patch activation and 0.04 ("not stripey") on a sky-patch activation.

LINEAR PROBE, concretely (LogisticRegression)
  "Linear" means the ruler is just a weighted sum of the 64 activation numbers
  plus a bias, squashed into a 0..1 probability:
      p = sigmoid( w[0]*a[0] + w[1]*a[1] + ... + w[63]*a[63] + b ).
  LEARNING the probe = finding the weights w (a 64-number arrow) and bias b that
  best separate "concept present" (label 1) from "concept absent" (label 0) in
  the labelled bank from step1. scikit-learn's LogisticRegression does this fit.
  - sigmoid(x) = 1/(1+e^-x): a soft 0/1 switch. Tiny number: sigmoid(0)=0.5,
    sigmoid(3)=0.95, sigmoid(-3)=0.05.
  - the learned weight vector w POINTS in the activation-space direction the
    concept lives along — so it is itself a "concept direction" discovered from
    LABELS. (That is exactly what makes it the TCAV-style supervised reference in
    step3: a label-trained concept direction.)

ACCURACY (how we trust a ruler)
  After training we check the probe on HELD-OUT activations it never saw: what
  fraction does it label correctly? A good ruler scores near 1.0; a useless one
  scores ~0.5 (a coin flip). We print this so you can see the rulers are real.
  Tiny number: 190 correct out of 200 held-out items = 0.95 accuracy.

WHY WE NEED ONE RULER PER CONCEPT
  SPECIFICITY (step3) asks: when I steer the TARGET concept, do the OFF-TARGET
  concepts stay flat? To answer that you must be able to READ every off-target
  concept at every knob setting — that is what the per-concept rulers are for.

U_r — THE ON-MANIFOLD SUBSPACE (recap from milestone 4, reused here)
  Real activations live on a thin SHEET inside the 64-dim space. We estimate that
  sheet with PCA (Principal Component Analysis: "find the directions the cloud
  spreads out most") and keep the top `manifold_rank` directions as the COLUMNS
  of a [dim, r] matrix U_r. The on-manifold steerer projects its edit onto this
  sheet (a <- a + s*(P_M*d), P_M = U_r U_r^T) so the edit stays realistic.
  Milestone 4 teaches PCA/U_r/P_M from zero; here we just rebuild U_r so this
  milestone runs standalone.

==============================================================================
RUN
==============================================================================
    /usr/bin/python3 step2_train_probes.py
Reads  outputs/probe_acts.npy, outputs/probe_labels.npy  (from step1).
Writes outputs/probe_weights.npy  [n_concepts, dim]  one ruler-direction per concept
       outputs/probe_bias.npy     [n_concepts]       one bias per ruler
       outputs/U_r.npy            [dim, r]           the on-manifold subspace basis
"""
from __future__ import annotations

import numpy as np

from _common import banner, load_cfg, outpath


def train_probes(acts: np.ndarray, labels: np.ndarray, seed: int):
    """Train one LogisticRegression probe per concept; return (W, b, accs).

    W [n_concepts, dim] = each row is the learned weight vector (the concept's
    ruler-direction). b [n_concepts] = each probe's bias. accs = held-out
    accuracy per concept (how trustworthy each ruler is).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    n_c = labels.shape[1]
    dim = acts.shape[1]
    W = np.zeros((n_c, dim), dtype=np.float32)
    b = np.zeros((n_c,), dtype=np.float32)
    accs = []
    for c in range(n_c):
        y = labels[:, c].astype(int)
        # split into train / held-out test so accuracy is honest (not memorized).
        Xtr, Xte, ytr, yte = train_test_split(
            acts, y, test_size=0.2, random_state=seed + c, stratify=y)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, ytr)
        W[c] = clf.coef_[0].astype(np.float32)      # the ruler-direction
        b[c] = float(clf.intercept_[0])
        accs.append(float(clf.score(Xte, yte)))     # held-out accuracy
    return W, b, accs


def estimate_U_r(acts: np.ndarray, r: int) -> np.ndarray:
    """PCA the bank; return U_r [dim, r] (top-r principal directions as columns).

    Same construction milestone 4 explains in full: PCA finds the directions the
    activation cloud spreads out most; the top-r of them are our estimate of the
    on-manifold sheet. We transpose components_ so directions are COLUMNS.
    """
    from sklearn.decomposition import PCA
    dim = acts.shape[1]
    r = min(int(r), dim)
    pca = PCA(n_components=r, svd_solver="full")
    pca.fit(acts)
    U_r = pca.components_[:r].T.copy()              # [dim, r], columns = top-r PCs
    return U_r.astype(np.float32)


def worked_probe_example() -> None:
    """Print a tiny by-hand 'probe = weighted sum + sigmoid' example."""
    banner("TINY WORKED EXAMPLE: a linear probe is a weighted sum, then a sigmoid")

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    w = np.array([2.0, -1.0])          # learned ruler-direction (2-D for clarity)
    bias = -0.5
    a_has = np.array([1.5, 0.2])       # an activation that HAS the concept
    a_not = np.array([-0.3, 1.0])      # one that does NOT
    z_has = float(w @ a_has + bias)
    z_not = float(w @ a_not + bias)
    print(f"  ruler weights w = {tuple(w)}, bias = {bias}")
    print(f"  present activation a = {tuple(a_has)} -> w.a+b = {z_has:+.2f} "
          f"-> p = sigmoid = {sigmoid(z_has):.2f}  (>0.5 -> reads 'present')")
    print(f"  absent  activation a = {tuple(a_not)} -> w.a+b = {z_not:+.2f} "
          f"-> p = sigmoid = {sigmoid(z_not):.2f}  (<0.5 -> reads 'absent')")
    print("  >>> SPECIFICITY (step3) holds these rulers up to OFF-TARGET concepts")
    print("      while we steer the TARGET, and checks the off-target reads stay flat.")


def main() -> None:
    cfg = load_cfg()
    seed = int(cfg["seed"])
    r = int(cfg["manifold_rank"])
    banner("STEP 2 — train one LINEAR PROBE (ruler) per concept + estimate U_r")

    acts = np.load(outpath("probe_acts.npy"))
    labels = np.load(outpath("probe_labels.npy"))
    print(f"  loaded acts {acts.shape}, labels {labels.shape} "
          f"(rerun step1 if these are missing)")

    print("\n  training one LogisticRegression probe per concept ...")
    W, b, accs = train_probes(acts, labels, seed)
    for c, acc in enumerate(accs):
        tag = "TARGET " if c == int(cfg["target_concept"]) else "off-tgt"
        print(f"    concept {c} [{tag}] held-out accuracy = {acc:.3f}  "
              f"(~1.0 => a real, trustworthy ruler)")
    print(f"  mean probe accuracy = {np.mean(accs):.3f}  "
          f"(rulers must be accurate before we can read concepts off activations)")

    print(f"\n  estimating the on-manifold subspace U_r (top r = {r} PCs by PCA) ...")
    U_r = estimate_U_r(acts, r)
    P = U_r @ U_r.T
    print(f"  U_r shape = {U_r.shape}  (each COLUMN is one estimated sheet direction)")
    print(f"  projection check trace(P_M) = {np.trace(P):.2f}  (should equal r = {r})")

    np.save(outpath("probe_weights.npy"), W)
    np.save(outpath("probe_bias.npy"), b)
    np.save(outpath("U_r.npy"), U_r)
    print(f"\n  saved -> {outpath('probe_weights.npy')}  (one ruler-direction per concept)")
    print(f"  saved -> {outpath('probe_bias.npy')}     (one bias per ruler)")
    print(f"  saved -> {outpath('U_r.npy')}           (the on-manifold subspace basis)")

    worked_probe_example()
    print("\nSTEP 2 done. Next: step3 MEASURES the three CFS components per method.")


# REAL RUN (M5): train the probes on REAL CLIP ViT-B/16 activations with real
# concept labels (the same LogisticRegression call). Estimate U_r ONCE from the
# large real ImageNet-val activation bank (reuse the M4-cached U_r.npy). The
# TARGET probe's weight vector becomes the TCAV-style supervised reference
# direction in step3 — exactly as here.
if __name__ == "__main__":
    main()
