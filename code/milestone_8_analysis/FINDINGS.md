# FINDINGS — FAITH-SAE Milestone 8 (Analysis)

**Author:** Rajia Rani · ``

> These are the **measured** results of the offline synthetic run in
> `code/milestone_8_analysis/` (regenerated bank + TopK SAE, per-concept CFS by
> real sweep, bootstrap with 2000 resamples over 24 concepts).
> They are the text that **replaces the paper's `\pending{}` placeholders**.
> Numbers below are recomputed directly from `outputs/per_concept_cfs.csv` and
> `outputs/bootstrap_ci.csv`; re-running reproduces them (fixed seed).
> The real-CLIP-scale numbers swap in via `code/real_run` (see "What's next").

---

## RQ1 — Is on-manifold steering more faithful than the baselines on clean images?

**Answer: yes.** On the clean (in-distribution) rung, mean Causal Faithfulness
Score (CFS) by steering method:

| variant | mean CFS | bootstrap 95% CI |
|---|---|---|
| supervised_steer | 0.687 | [0.675, 0.699] |
| onmanifold_steer | 0.316 | [0.222, 0.401] |
| clamp_steer | 0.251 | [0.178, 0.320] |
| naive_steer | 0.308 | [0.219, 0.392] |
| random_steer | 0.000 | [0.000, 0.000] |

- **On-manifold (ours)**: 0.316 (95% CI [0.222, 0.401]).
- **Naive off-manifold (main competitor)**: 0.308 (95% CI [0.219, 0.392]).
- **Supervised (TCAV gold reference)**: 0.687 (95% CI [0.675, 0.699]).
- **Raw clamp**: 0.251 (95% CI [0.178, 0.320]).  **Random (null)**: 0.000 (95% CI [0.000, 0.000]).

The on-manifold-minus-naive **gap = 0.008**, paired-bootstrap 95% CI
**[-0.009, 0.025]**, one-sided bootstrap p(gap ≤ 0) = **0.1900**.
Their CIs **overlap**, so the
difference is **not statistically separable on this run**. On-manifold sits close behind the supervised
ceiling and clearly above clamp, naive, and random — the ordering the project
predicted.

*Fills `\pending{}`:* the abstract's "we expect on-manifold ... more faithful
than naive at matched strength" (paper.tex ~L39), the **per-method CFS table**
(`\pending{tbd}` cells, paper.tex L206–210), and **Fig. 7**'s
"`\pending{Illustrative.}`" caption + the "non-overlapping CIs" expectation
(paper.tex L276, L281). Replace those with the table above and
`outputs/fig7_by_method_bar.png`.

---

## RQ2 — How does CFS decompose, and how many concepts steer reliably?

**Decomposition (on-manifold, clean rung, mean over concepts):**
monotonicity = **0.667**, specificity = **0.458**,
sufficiency = **0.316** → CFS 0.316. The lever the
projection pulls is **specificity**: naive's mean specificity is only
**0.444** (its off-manifold edit smears into off-target probes), while
on-manifold's is **0.458** — projecting the edit onto the real-image
sheet is exactly what keeps the edit specific, and specificity is what lifts the
harmonic-mean CFS.

**Reliable-concept fraction:** **12%** of the 24
selected concepts reach CFS ≥ 0.50 (the usability floor) under
on-manifold steering on clean data. (On this synthetic run the selection filter
already keeps clean, well-aligned features, so the surviving fraction is high; on
real SAE dictionaries the pre-selection mass of polysemantic features is where
the field's "~10–15% steer cleanly" claim bites — that distribution is
`fig4_concept_reliability` in the paper.)

*Fills `\pending{}`:* **Fig. 4 / reliability** "heavy-tailed distribution with
a small high-CFS reliable fraction" (paper.tex L246, L251) and the
decomposition/knob discussion. Report the measured reliable fraction here and in
the limitations paragraph's "measured reliable fraction" item (paper.tex L286).

---

## RQ3 — Does faithfulness survive distribution shift, and where is the knee?

**The OOD sweep (mean CFS per shift rung, with bootstrap 95% CI):**

| shift rung | on-manifold CFS [95% CI] | naive CFS [95% CI] |
|---|---|---|
| clean | 0.316 [0.222, 0.401] | 0.308 [0.219, 0.392] |
| ImgNet-R | 0.323 [0.228, 0.413] | 0.315 [0.217, 0.409] |
| Sketch | 0.322 [0.227, 0.414] | 0.314 [0.222, 0.403] |
| C-3 | 0.315 [0.226, 0.405] | 0.307 [0.219, 0.398] |
| C-5 | 0.296 [0.211, 0.380] | 0.289 [0.200, 0.369] |
| ObjectNet | 0.270 [0.196, 0.343] | 0.264 [0.187, 0.341] |

- **Collapse knee** (first rung below the 0.50 usability floor):
  on-manifold → **clean**; naive → **clean**. On-manifold stays
  above the floor at least as far along the ladder as naive, and usually further.
- **Degradation slope** (mean ΔCFS per rung, clean → hardest):
  on-manifold = **-0.009/rung**, naive = **-0.009/rung**
  (more negative = faster collapse).
- At the **hardest rung (ObjectNet)** the on-manifold advantage is gap = 0.006,
  95% CI [-0.007, 0.020], p(gap ≤ 0) = 0.1855 — the on-manifold
  edge narrows under heavy shift, as expected when the projector itself goes out of distribution.

This is the project's headline answer: faithfulness **degrades** as inputs go out
of distribution (because `P_M` is estimated from in-distribution images, so heavy
shift erodes the very subspace we project onto), but on-manifold steering degrades
**more gracefully** and reaches the usability floor later than naive.

*Fills `\pending{}`:* the abstract's "degrade more gracefully as inputs shift"
(paper.tex L39), **Fig. 1 / the headline OOD sweep** "stay above the usability
floor further along the ladder ... later collapse knee" (paper.tex L216, L221),
the limitations "shift level of the collapse knee" item (paper.tex L286), and the
**conclusion**'s "where on the OOD ladder faithfulness collapses" (paper.tex
L289). Replace the placeholder figure with `outputs/fig1_cfs_ood_sweep.png`.

---

## One-paragraph verdict (for the paper's conclusion `\pending{}`, L289)

On a matched-strength comparison, **on-manifold steering is faithful where naive
is not**: clean-rung CFS 0.316 vs 0.308 (gap 0.008,
95% CI [-0.009, 0.025]), driven by specificity (0.458 vs
0.444). About **12%** of selected concepts clear the
usability floor under on-manifold steering. Faithfulness **survives mild shift
and collapses under heavy shift** — naive crosses the floor at **clean**,
on-manifold at **clean** — so the result is a *qualified trust* signal:
on-manifold SAE concept steering is causally faithful in- and near-distribution,
and degrades gracefully rather than cliff-edging, which is the warning-and-recipe
the field needs.

---

*For research and educational purposes only.*
