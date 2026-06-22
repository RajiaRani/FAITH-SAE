# Milestone 3 — Baseline: train the TopK SAE and measure naive-steering CFS

**FAITH-SAE** · author: **Rajia Rani** ()
*For research and educational purposes only.*

> This page assumes you know **nothing**. Every term is defined from zero — a
> plain definition, an everyday analogy, and a tiny number you can check by hand —
> *before* it is used. Read top to bottom; do not skip the "Concepts from zero"
> section.

---

## 1. Where this fits

The whole FAITH-SAE project asks one question: when we find a "concept" inside a
frozen vision model and **steer** it (push it up or down), is that edit a **real,
faithful causal lever** — or a convincing-looking fake? And does it stay faithful
when the input images get weird (out of distribution)?

To answer that we need, in order:

| Milestone | What it builds | Status |
|---|---|---|
| M1 foundations | the smallest runnable pieces (tensors, a toy SAE forward pass) | (earlier) |
| M2 data pipeline | the **activation bank** (synthetic now; real CLIP later) | (earlier) |
| **M3 baseline (you are here)** | **train the SAE, pick clean concepts, measure the BASELINE steering score** | **this folder** |
| M4 method | the proposed **on-manifold** steering that must **beat** this baseline | next |

**This milestone produces the number to beat.** We (A) **train** a TopK Sparse
Autoencoder on activations and pick a few clean concepts, then (B) run the
**baseline steerer** (`naive_steer`) and measure its **Causal Faithfulness Score
(CFS)**. Milestone 4's whole job is to score higher than the CFS you compute here.

This folder is **self-contained**: it regenerates its own synthetic activation
bank, so it does **not** depend on milestone 2's outputs.

---

## 2. What you build & run

Two parts, three small scripts, one optional driver:

- **Part A — train + select (`step1`, `step2`)**
  - `step1_train_sae.py` — make a synthetic activation bank, **train a TopK SAE**
    on it, save the trained model and a **reconstruction-loss curve PNG**.
  - `step2_select_concepts.py` — score every learned feature for "cleanliness"
    and keep the few **testable** concepts (the field says only ~10–15% steer
    reliably, so we *select*).
- **Part B — baseline steer + score (`step3`)**
  - `step3_naive_steer_cfs.py` — run the **baseline** steerer `naive_steer`
    (just add `s·direction`) on each selected concept and compute its **CFS**.
    Writes `outputs/baseline_cfs.csv`.
- **Driver** — `run_all.py` runs all three in order (the default entry point).

Everything reuses the project's real code in `../../src/` (the `TopKSAE` model,
the `naive_steer` steerer from `STEER_REGISTRY`, and the `cfs_score` helper) — we
do **not** re-implement them here.

---

## 3. Concepts from zero

Read these once; the scripts repeat the same definitions in their comments.

**Activation.**
*Def:* the list of numbers a neural network produces inside itself when it looks
at one input. *Analogy:* the network's private "notes" about what it sees.
*Number:* a length-4 activation might be `[0.2, -1.1, 0.0, 3.4]`. Our bank holds
many such vectors, each of length `dim = 64`.

**Patch / token.**
*Def:* a vision transformer chops an image into a grid of small squares
("patches") and makes one activation vector per patch. *Analogy:* reading a
picture tile-by-tile instead of all at once. *Number:* `n_patches = 16` is a 4×4
grid → 16 vectors per image.

**Autoencoder.**
*Def:* a network that learns to **copy** its input to its output through a
restricted middle, so the middle becomes a clean re-description of the input.
*Analogy:* describe a photo in a few words ("beach, sunset, dog") and have a
friend redraw it; if the redraw matches, your few words captured the photo.
*Number:* in → middle code → out, where in and out are both length 64.

**Encoder / decoder.**
*Def:* the **encoder** turns an activation (64 numbers) into a longer **feature
code** (256 numbers, mostly zero); the **decoder** turns that code back into a
reconstructed activation (64 numbers). *Number:* encoder `64 → 256`, decoder
`256 → 64`.

**Feature / concept direction.**
*Def:* each of the 256 features stands for "a concept." The decoder is a `64×256`
matrix; its **column j** is a length-64 vector — the **direction** that feature j
paints into the activation when it turns on. We call that column the **concept
direction** `d_j`. *Analogy:* a paint tube; turning feature j up squirts paint of
colour `d_j` onto the activation. *Number:* if column 5 is `[0,1,0,…]`, turning
feature 5 up adds to the 2nd coordinate of the activation.

**Reconstruction loss (MSE).**
*Def:* "how wrong is the copy?" — **M**ean **S**quared **E**rror: subtract the
copy from the original, square each coordinate, average. *Analogy:* grading the
friend's redraw by average squared mismatch. *Number:* original `[1, 2]`, copy
`[1.5, 2]` → errors `[0.5, 0]` → squares `[0.25, 0]` → MSE `= 0.125`. Training
shrinks this; **lower = better copy**.

**Sparsity, and why it helps interpretability.**
*Def:* "sparse" = mostly zeros; we force only a few features nonzero per
activation. *Why:* if every feature lit up for everything, each would be a vague
blur used for many things ("polysemantic"). Forcing few-active pushes each
feature to mean **one** clean thing ("monosemantic"), so a human can name it.
*Analogy:* a tidy toolbox with one labelled tool per slot beats a junk drawer
where every tool is half of three others. *Number:* of 256 features, keep 8 on,
zero 248.

**Top-k operation.**
*Def:* the exact rule that enforces sparsity — keep the **k largest** feature
values, set the rest to 0. *Number:* with k=2 on `[5, 1, 4, 0, 2]`, the two
largest are 5 and 4, so the code becomes `[5, 0, 4, 0, 0]`.

**Dictionary learning (one line).**
Learning a small set of reusable "atoms" (the decoder columns) so any activation
is a short combination of a few atoms — that is all an SAE is.

**Loss curve.**
*Def:* a plot of reconstruction loss (y) vs training step (x). *How to read it:*
a healthy curve **starts high and falls**, steeply at first, then flattens as the
SAE runs out of easy wins. A flat-from-the-start or rising curve means something
is wrong (see §8).

**Steering.**
*Def:* deliberately **editing** an activation to turn a concept up/down, then
seeing what changes. *Analogy:* the bass knob on a stereo — push it, listen.

**Steering strength `s`.**
*Def:* **how hard** you push. We move the activation by `s` units along the
concept's (unit-length) direction `d`. *Number:* `a=[1,0]`, `d=[0,1]`, `s=3` →
`a' = a + s·d = [1, 3]`. The whole study uses the **same** `s` for every method
("matched strength"), so differences come from the *method*, not the push.

**Baseline — and why `naive_steer` (off-manifold) is the *weak* one.**
*Def of baseline:* the simplest sensible thing you compare against — the bar to
clear. Here it is `naive_steer`: `a' = a + s·d`, just add the direction.
*Why weak:* real activations don't fill the whole 64-D space; they cluster near a
thin, curved surface called the **manifold** (think: the skin of a balloon inside
a room). Adding a raw direction usually shoves the point **off** that surface,
into a region the model never sees in real life ("off-manifold"). The readout
still moves (so it *looks* like an effect), but the edit is unrealistic and
**leaks** into unrelated concepts. That is why naive steering earns only a
**mediocre** CFS — and exactly what milestone 4 fixes by projecting the edit back
onto the manifold. (`naive_steer` is the no-projection special case of the M4
method.)

**Causal Faithfulness Score (CFS) — the headline number, in `[0,1]`.**
*Def:* is this edit a real, clean causal lever? It is the **harmonic mean** (an
"AND" — all three must be high) of:
- **Monotonicity** — turn the knob up, does the target readout rise **smoothly,
  in order**? (Spearman rank correlation of knob vs readout; 1.0 = perfectly
  ordered.)
- **Specificity** — does **only** the target move? We watch unrelated
  ("off-target") concept readouts; if they barely drift, specificity is high.
  (`1 − worst off-target drift`.)
- **Sufficiency** — is the effect **big enough** to matter? (A standardized effect
  size, Cohen's-d style, at full knob vs none.)
*Number:* CFS of `(0.9, 0.1, 0.9)` ≈ **0.23** — the 0.1 specificity poisons the
whole thing. That conjunctive behaviour is the point: faithful means **all three
at once**.

---

## 4. Prereqs & setup

You need the project's interpreter **`/usr/bin/python3`** (do **not** use plain
`python` / `python3`). The offline default downloads **nothing**.

Check the four libraries this milestone uses (all already in the base env):

```bash
/usr/bin/python3 -c "import torch, yaml, numpy, matplotlib; print('ok')"
```

If that prints `ok`, you are ready. If anything is missing:

```bash
/usr/bin/python3 -m pip install -r code/milestone_3_baseline/requirements.txt
```

(`requirements.txt` here is **additive** — on top of the repo-root one.)

---

## 5. Run it step-by-step

From this folder (`code/milestone_3_baseline/`). The one-shot driver:

```bash
/usr/bin/python3 run_all.py
```

…or run the three steps individually to watch each stage:

1. **Train the SAE.** *Why:* an untrained SAE reconstructs garbage; training
   minimises reconstruction MSE so the decoder columns become meaningful concept
   directions.
   ```bash
   /usr/bin/python3 step1_train_sae.py
   ```
   Writes `outputs/sae_topk.pt` (the trained model) and
   `outputs/sae_loss_curve.png` (the loss curve).

2. **Select clean concepts.** *Why:* most learned features are messy; we keep
   only the few that look like crisp, nameable concepts, so we don't waste the
   steering test on junk. This is the paper's "~10–15% steer reliably" selection.
   ```bash
   /usr/bin/python3 step2_select_concepts.py
   ```
   Writes `outputs/selected_concepts.csv`.

3. **Baseline steer + CFS.** *Why:* this produces the **number M4 must beat** —
   the faithfulness of plain off-manifold steering.
   ```bash
   /usr/bin/python3 step3_naive_steer_cfs.py
   ```
   Writes `outputs/baseline_cfs.csv`.

Each step is independently runnable; steps 2 and 3 read the checkpoint/CSV that
the earlier step wrote (they will tell you to run the prior step if it's missing).

---

## 6. Expected output

After `run_all.py` (exit code **0**), `outputs/` contains:

| File | What it is | Success looks like |
|---|---|---|
| `sae_topk.pt` | the trained TopK SAE checkpoint | exists; final recon MSE well below the step-0 value |
| `sae_loss_curve.png` | reconstruction-loss curve | starts high (~1.2), **falls** and flattens (~0.1) |
| `selected_concepts.csv` | the chosen testable concepts | 5 rows, each with a positive `alignment` to a planted concept |
| `baseline_cfs.csv` | the **baseline** CFS table | one row per concept + a `MEAN` row |

**Reference run** (seed 0, defaults). Your numbers should be very close:

- `step1`: reconstruction MSE falls from **≈1.26 → ≈0.10** over 400 steps; average
  active features stays at **8** (= `topk_k`).
- `step2`: 5 concepts kept, `cleanliness ≈ 0.36–0.43`, `alignment ≈ 0.52–0.72`.
- `step3`: **mean `naive_steer` CFS ≈ 0.77**, with monotonicity ≈ 1.0,
  specificity ≈ 0.71, sufficiency ≈ 0.68, off-manifold residual ≈ 0.38.

**Success criterion.** The driver exits **0**; `baseline_cfs.csv` exists with a
`MEAN` row whose `cfs` is a finite number in `[0,1]`, monotonicity is high
(steering *does* move the target), and specificity is clearly **below** 1.0 and
the off-manifold residual is clearly **above** 0 — i.e. the baseline is real but
**imperfect**, leaving room for M4 to improve.

---

## 7. Understand the result

**What a good SAE reconstruction looks like.** The loss curve should drop quickly
then flatten — the SAE learned to rebuild activations from just 8 active features.
Here MSE lands near **0.10**, far below the **≈1.26** it started at, because our
synthetic activations genuinely live in a low-dimensional subspace (the
"manifold") that a small dictionary can capture. If the curve never fell, the
decoder columns would be meaningless and steering would be nonsense.

**Why the selected concepts are "clean."** Each kept feature (a) actually fires on
a fair share of inputs (not dead), (b) fires decisively when it does, and (c)
points along one of the planted ground-truth directions (`alignment` up to 0.72).
Those three together are our offline stand-in for "a human could name this
concept."

**Why the baseline CFS is mediocre (≈0.77, not ≈1.0).** Look at the components:
- **Monotonicity ≈ 1.0** — pushing harder *does* move the target readout
  smoothly. Naive steering is good at *producing an effect*.
- **Specificity ≈ 0.71** (not 1.0) — steering one concept also nudges *unrelated*
  off-target probes. That's **leakage**: because the planted concepts are
  correlated and the raw edit ignores the manifold, the push spills into
  neighbours.
- **Sufficiency ≈ 0.68** — the effect is decent but not overwhelming.
- **Off-manifold residual ≈ 0.38** — a chunk of the edit lands **outside** the
  top-`r` real-data subspace. This is the smoking gun: the edit partly leaves the
  region the model actually uses.

Because CFS is a conjunctive (harmonic) mean, the imperfect specificity and
sufficiency pull the score down to ≈0.77. **That is the bar.** Milestone 4's
on-manifold method projects the edit back onto the manifold, which should *raise
specificity*, *drive the off-manifold residual toward 0*, and so *push CFS higher*
— and, crucially, hold up better as images shift out of distribution (RQ3).

---

## 8. Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `command not found: /usr/bin/python3` | wrong interpreter path on your OS | use the absolute path to your Python 3 with torch installed; do **not** use a different one than the rest of the repo |
| `ModuleNotFoundError: No module named 'src'` | ran from the wrong directory | run from inside `code/milestone_3_baseline/`; each script auto-adds the repo root to `sys.path`, but the working dir still matters for `outputs/` |
| `ModuleNotFoundError: torch / matplotlib / yaml` | base env not active | `/usr/bin/python3 -m pip install -r requirements.txt` |
| `FileNotFoundError: ... sae_topk.pt` | ran `step2`/`step3` before `step1` | run `step1_train_sae.py` first (or just use `run_all.py`) |
| loss curve is flat or rises | learning rate too high/low, or seed unlucky | lower `lr` in `config.yaml` (try `0.001`) or raise `steps` |
| CFS comes out exactly 0 | one component collapsed to 0 (harmonic mean → 0) | check `baseline_cfs.csv`: a 0 in any component means that axis failed; with defaults this shouldn't happen |
| numbers differ slightly from the reference | different seed / library version | small drift is fine; the *story* (CFS < 1, specificity < 1, residual > 0) must hold |

---

## 9. What's next → `milestone_4_method`

You now have the **baseline**: a trained TopK SAE, a handful of clean concepts,
and the mediocre CFS of naive off-manifold steering (≈0.77, residual ≈0.38).

Milestone 4 swaps `naive_steer` for **`onmanifold_steer`** (ours): it projects the
raw edit onto the top-`r` real-image subspace (`a ← a + s·(P_M·d)`), keeping the
edit on the manifold. The expectation: **higher specificity, off-manifold residual
near 0, and a higher CFS than the number in `outputs/baseline_cfs.csv`** — plus
graceful degradation under distribution shift. That comparison, at matched
strength, is the project's headline result (RQ1).

---

### REAL RUN (M3)

The offline default trains the SAE on a **synthetic** activation bank so it runs
today on CPU with no downloads. To train on **real CLIP activations** instead:

1. In milestone 2's real path, dump a bank of **CLIP ViT-B/16 patch
   activations** over ImageNet (uncomment `open_clip_torch` + `datasets` in
   `requirements.txt`).
2. Replace `make_activation_bank(cfg)` in `step1_train_sae.py` with a loader for
   that real bank (set `dim: 768`, the CLIP width, in `config.yaml`).
3. Replace the planted-concept `alignment` term in `step2_select_concepts.py`'s
   cleanliness score with a real interpretability signal (e.g. a learned linear
   probe / TCAV-style label agreement), since real banks have no planted answer.
4. Re-run the same three steps. Everything else — the TopK SAE, `naive_steer`,
   and `cfs_score` — is unchanged; only the data source and the readout probe
   move from synthetic to real.
