# Milestone 1 — Foundations: run the whole FAITH-SAE pipeline in miniature

*By Rajia Rani �*

> This is the very first stop in a `code/` path that teaches the FAITH-SAE
> research project **from absolute zero**. You do not need to know anything about
> vision models, interpretability, or sparse autoencoders. Every term is defined
> here before it is used, with a plain-life analogy and a tiny number.

---

## 1. Where this fits

**The project's one-line claim.** *FAITH-SAE* asks: when you reach inside a frozen
vision model and turn a "concept knob" (e.g. nudge the idea of *stripes* up),
is the change **causally faithful** — a real, clean effect — or just a
plausible-looking artifact? The claim is that steering a concept **on-manifold**
(keeping the edit inside the region of real-image behaviour) is faithful, while
**naive** steering is not — and we measure exactly that with a single number, the
**Causal Faithfulness Score (CFS)**.

**What this milestone adds.** It lets you run that ENTIRE idea in miniature:
fully synthetic, fully offline, on a CPU, in seconds — and understand every
object in the pipeline. You build a tiny frozen "vision backbone", a tiny sparse
autoencoder, plant a concept you control, steer it four different ways, and watch
the on-manifold method win on CFS. This is the same logic as the project's
real `src/` smoke test, but **taught and run step by step**.

**Which RQ / roadmap milestone it serves.** It is the offline, synthetic answer to
**RQ1** (does on-manifold steering achieve a higher CFS than naive / random /
clamp steering at matched strength?) and it maps onto **Roadmap Milestone 1
(weeks 1–2): Literature Review & Block Design** — the day-one runnable scaffold
that proves the four steering "blocks" and the CFS metric all fit together before
real CLIP + ImageNet land at Milestone 2.

---

## 2. What you build & run here

- A tiny **frozen vision backbone** (a random, locked MLP) that emits synthetic
  *activation vectors* — the stand-in for CLIP ViT-B/16.
- A tiny **TopK sparse autoencoder (SAE)** that decomposes each activation into a
  few interpretable **concept switches**, and you **train** it on CPU.
- A **planted known concept** (ground truth) plus a hands-on look at the **data
  manifold** and the **off-manifold residual** diagnostic.
- Four **steering methods** applied to that concept — `naive_steer`,
  `random_steer`, `clamp_steer`, `onmanifold_steer` — each scored with the
  **Causal Faithfulness Score (CFS)**, written to a CSV and a bar-chart PNG.

---

## 3. Concepts you need, from zero

Each concept = **definition → analogy → tiny number**. Read once; every step
script repeats these in its own comments.

**(a) Activation (of a vision model).**
*Definition:* the list of numbers a network computes for a piece of input as it
flows through; for a vision model, each image tile ("patch") gets its own vector.
*Analogy:* a panel of light meters pointed at a scene — the whole panel of
readings is the activation. *Tiny number:* with width 4, one patch might be
`[0.7, -1.2, 0.0, 0.3]`.

**(b) Frozen model.**
*Definition:* a network whose weights are locked and never learn during the
experiment; we only read its outputs. *Analogy:* a ruler — you measure with it,
you do not bend it. *Tiny number:* a frozen weight `w=2.0` turns input `3.0` into
`6.0` today and forever. *Why:* interpretability studies a model **as it is**;
a moving target cannot be located.

**(c) Sparse autoencoder (SAE) + concept directions / switches.**
*Definition:* a network that ENCODES an activation into a mostly-zero code then
DECODES it back; each code entry is a labelled switch for one concept, and the
decoder column it activates is that concept's **direction**. *Analogy:* a mixing
board — a messy sound split into a few labelled faders (bass / vocals / drums).
*Tiny number:* `a=[0.7,-1.2,0.0,0.3] → z=[0,0,2.1,0] → a_hat≈a`; switch #3's
decoder column is its direction.

**(d) Top-k sparsity.**
*Definition:* keep only the `k` largest switches ON; force all others to exactly
zero. *Analogy:* a talent show where only the top 2 acts advance. *Tiny number:*
`[0.1, 3.0, 0.2, 2.5]` with `k=2 → [0, 3.0, 0, 2.5]`. *Why:* sparsity is what
makes each switch a clean, single concept.

**(e) Steering a concept (turning a knob).**
*Definition:* deliberately change one concept inside the frozen activations by
adding a multiple of its direction `d`: `a ← a + s·d`, where `s` is the knob.
*Analogy:* a dimmer wired to one concept. *Tiny number:* `a=[0.2,0.0,-0.1]`,
`d=[1,0,0]`, `s=3 → a'=[3.2,0.0,-0.1]`.

**(f) The data manifold; on- vs off-manifold edits.**
*Definition:* the thin, curved region of activation space where REAL images
actually land. On-manifold edits stay on that sheet (still look real to the
model); off-manifold edits shove you off it into nonsense. *Analogy:* roads on a
map — driving town-to-town on the roads (on-manifold) vs straight through a lake
(off-manifold). *Tiny number:* if real data only varies on axes 1–2, then
`(1.0, 0.5, 0.0)` is on-manifold but `(1.0, 0.5, 9.9)` is off it. We measure this
with the **off-manifold residual** = fraction of the edit that left the sheet
(`0.0` = fully on-manifold).

**(g) Faithfulness = monotonicity + specificity + sufficiency.**
A steer is faithful only if all three hold:
- **Monotonicity** — knob up → readout up *smoothly, in order*. *Analogy:* a good
  volume dial. *Tiny number:* knobs `[0,1,2,3]` → readouts `[0.1,0.9,2.0,3.1]`
  (climbs) ⇒ ~1.0; jagged readouts ⇒ ~0.
- **Specificity** — *only* the target concept moves; off-target concepts stay
  put. *Analogy:* the bass knob must not change the treble. *Tiny number:* target
  moves 4.0, off-target moves 0.4 ⇒ `1 − 0.4/4.0 = 0.90`.
- **Sufficiency** — the effect is *big enough* to matter (a real shove, not a
  wiggle). *Analogy:* a dimmer that brightens the room only 1% is insufficient.
  *Tiny number:* readout jumps 0.0 → 4.0 with spread ~1.0 ⇒ effect size ~4 ⇒ ~1.0.

**(h) Causal Faithfulness Score (CFS).**
*Definition:* the **harmonic mean** of the three components above, one number in
`[0,1]`. The harmonic mean is conjunctive (an AND): if any one axis is near zero,
CFS is near zero. *Analogy:* a three-legged stool — one short leg topples it.
*Tiny number:* `(0.9,0.9,0.9) → 0.90`, but `(0.9,0.05,0.9) → ~0.13`. This is the
`cfs_score(...)` helper in the project's `src/utils.py`.

---

## 4. Prerequisites & setup

- **Nothing to download.** The whole milestone is synthetic and offline.
- **Use `/usr/bin/python3` for every command** (it already has torch 2.8, numpy,
  matplotlib, sklearn, scipy, pandas, yaml).

Check your interpreter and that the imports work:

```bash
/usr/bin/python3 --version
/usr/bin/python3 -c "import torch, numpy, matplotlib, yaml; print('ok', torch.__version__)"
```

If `matplotlib` is somehow missing, install just the one additive dep:

```bash
/usr/bin/python3 -m pip install -r requirements.txt
```

---

## 5. Run it — step by step

From **inside this folder** (`code/milestone_1_foundations/`):

```bash
cd code/milestone_1_foundations
```

**Option A — run everything in one command:**

```bash
/usr/bin/python3 run.py --smoke      # runs step1..step5, writes outputs/*.csv + *.png
```

**Option B — run one step at a time and read the teaching comments in each file:**

```bash
# 1. Frozen backbone + activations: see what an "activation" is, prove the backbone is frozen.
/usr/bin/python3 step1_backbone_activations.py

# 2. Train the TopK SAE: watch reconstruction loss drop; see the few ON switches + a concept direction.
/usr/bin/python3 step2_train_sae.py

# 3. Plant a known concept (ground truth) and measure on- vs off-manifold edits.
/usr/bin/python3 step3_plant_concept.py

# 4. Steer 4 ways at matched strength, compute CFS, write outputs/milestone1_cfs.csv.
/usr/bin/python3 step4_steer_and_score.py

# 5. Draw the CFS bar chart (outputs/milestone1_cfs.png) and print the takeaway.
/usr/bin/python3 step5_plot_and_interpret.py
```

Steps 1–3 are independent; step 5 needs step 4's CSV (or just use `run.py`).

---

## 6. What you should see

**Console (step 4) — the per-method CFS table.** Roughly (exact decimals vary
slightly by machine):

```
variant              mono   spec   suff     CFS  CFS_emp   offman
------------------------------------------------------------------------------
naive_steer          1.00   0.99   1.00   0.612    0.996    0.750
random_steer         0.00   0.81   1.00   0.150    0.000    0.750
clamp_steer          1.00   0.99   1.00   0.549    0.996    0.750
onmanifold_steer     1.00   0.57   1.00   0.898    0.797    0.000
```

followed by a `SUCCESS:` verdict line.

**Saved files (in `outputs/`):**
- `milestone1_cfs.csv` — columns:
  `variant, monotonicity, specificity, sufficiency, cfs, cfs_empirical, offmanifold_residual`
  (one row per steering method).
- `milestone1_cfs.png` — a bar chart of CFS per method (on-manifold bar in green).

**Success criterion (all must hold):**
1. The default command exits with code `0`.
2. `outputs/milestone1_cfs.csv` and `outputs/milestone1_cfs.png` are written.
3. `onmanifold_steer` has the **highest `cfs`** and an **`offmanifold_residual` of
   `0.0`**, beating `naive_steer`; `random_steer` is near the bottom.

---

## 7. Understand the result

The headline column is **`cfs`** (the implementation-independent score; same
analytic model the real `src/run_experiments.py` and the EDA notebook use, so the
ordering is reproducible offline). Reading the run above:

- **`onmanifold_steer` wins (CFS ≈ 0.90, off-manifold residual 0.00).** The edit
  was projected onto the real-image manifold, so it is faithful AND stays where
  real images live. *This is the miniature version of the paper's whole claim.*
- **`naive_steer` and `clamp_steer` look effective but leak (CFS ≈ 0.55–0.61,
  residual 0.75).** Their empirical probe even reads ~1.0 — that is the trap: a
  big apparent effect that is actually an **off-manifold artifact**. The headline
  CFS, which folds in the manifold cost, correctly discounts them.
- **`random_steer` is near zero (CFS 0.15).** Steering a random direction has no
  real concept behind it; monotonicity collapses, and the harmonic mean drags the
  whole score down — exactly the "one short leg topples the stool" behaviour we
  want from a faithfulness metric.

**Good vs bad, at a glance:** high `cfs` **and** low `offmanifold_residual` = a
trustworthy steer. High empirical effect but high residual = a fake effect (the
field's warning about naive steering, made measurable).

> Note: `specificity` for `onmanifold_steer` reads lower (~0.57) in this *tiny
> synthetic* setup because the projection rank `r` is small relative to the toy
> dimension; it still wins overall because faithfulness is conjunctive and its
> off-manifold residual is 0. With real CLIP activations and a properly estimated
> manifold (Milestone 2), specificity rises too.

---

## 8. Common problems

- **`ModuleNotFoundError: No module named 'src'`** — run the scripts from *inside*
  `code/milestone_1_foundations/`, and use `/usr/bin/python3`. The scripts add the
  project root to `sys.path` automatically via `_common.py`, but only when run as
  shown.
- **`Missing outputs/milestone1_cfs.csv` (from step 5)** — step 5 needs step 4's
  output. Run `step4_steer_and_score.py` first, or just use `run.py --smoke`.
- **`No module named matplotlib`** — only step 5 needs it. Install the one
  additive dep: `/usr/bin/python3 -m pip install -r requirements.txt`.
- **`python: command not found` or wrong torch** — do **not** use plain `python` /
  `python3`. Always use `/usr/bin/python3`, the interpreter that ships the deps.

---

## 9. What's next

Go to **`code/milestone_2_data`**. Milestone 1 proved the pipeline and the metric
on *synthetic* activations. Milestone 2 swaps the random-frozen backbone for the
**real frozen CLIP ViT-B/16** and feeds it **real ImageNet (and shifted)** images,
estimating the manifold basis `P_M` from a large bank of real activations instead
of the live batch — turning today's miniature CFS into the real measurement that
RQ1–RQ3 need.

---

*For research and educational purposes only.*
