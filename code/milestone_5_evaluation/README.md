# Milestone 5 — The Evaluation: the full Causal Faithfulness Score (CFS)

**FAITH-SAE** · author: **Rajia Rani** · ``

> Read this top to bottom. It assumes you know **nothing** about probes, rank
> correlation, effect sizes, or harmonic means — every term is defined from zero
> with a tiny number before it is used. By the end you will have **measured** (not
> looked up) the project's faithfulness metric end to end, and seen which steering
> methods are *causally real* and which are a *mirage*.

---

## 1. Where this fits

The whole FAITH-SAE project asks one question:

> When you reach inside a frozen vision model and **steer a concept** ("make this
> image more *dog*"), is the change you cause **causally real**, or just a
> plausible-looking artifact?

The project's answer has two halves:

- **A method** that makes the steer real — *on-manifold steering* (milestone 4).
- **A measuring stick** that proves it — the *Causal Faithfulness Score*, CFS
  (this milestone).

Milestone-by-milestone, the `code/` path is:

| Milestone | What it teaches / builds |
|---|---|
| `milestone_1_foundations` | the synthetic SAE pipeline, the four steerer names, what CFS is |
| `milestone_2_data` | the activation data (synthetic bank now; real CLIP later) |
| `milestone_3_baseline` | **naive** off-manifold steering `a' = a + s·d` — the competitor |
| `milestone_4_method` | the **proposed method**: on-manifold steering `a' = a + s·(P_M·d)` |
| **`milestone_5_evaluation` (you are here)** | the **full CFS metric**, every component *MEASURED* from the data, across 5 steering methods |
| `milestone_6_headline_experiment` | re-measures CFS along the out-of-distribution (OOD) ladder to find the collapse knee |

Milestones 1–4 *built* the pieces. This milestone is where the **measuring stick
itself is constructed and read**. Earlier milestones quoted a CFS number; here we
turn every part of it into a quantity computed from data:

> **CFS = harmonic mean of three measured components:**
> **monotonicity** (does turning the steering knob up move the concept up, in
> order?), **specificity** (did *only* the target concept move, or did it smear?),
> and **sufficiency** (was the effect *big enough* to matter?).

We compute all three for **five** steering methods on the same data with the same
knob settings — a matched-strength controlled experiment — and rank them.

> **Which research question?** This milestone answers **RQ2 — "Can a single
> scalar faithfulness score, built only from measurable internal signals, separate
> a causally-real steer from a mirage?"** The headline figure shows it can: the
> conjunctive harmonic mean drives any method with a weak component to ~0.

---

## 2. What you build & run

You will run a 4-step offline pipeline (no downloads, CPU only) that:

1. **Builds a multi-concept activation bank** living on a known 8-D sheet, with
   **one TARGET concept** to steer and **three OFF-TARGET concepts** that should
   stay put, and trains the project's real TopK SAE on it (`step1`).
2. **Trains one linear probe (a "ruler") per concept** so we can read every
   concept's level off any activation, and PCAs the bank into the on-manifold
   subspace **U_r** (`step2`).
3. **MEASURES the three CFS components** — monotonicity, specificity, sufficiency
   — for all five steering methods by sweeping the knob, then combines them with
   the project's harmonic-mean `cfs_score` (`step3`).
4. **Draws** `outputs/cfs_breakdown.png` — grouped bars, four per method, so you
   can see *which component each method wins or loses* (`step4`).

The five methods compared (the four runtime steerers come from the project's
`STEER_REGISTRY`; `supervised_steer` is the label-trained reference step3 builds):

| name | what it steers along | role |
|---|---|---|
| `supervised_steer` | the **target probe's** weight vector (a direction learned from labels) | **TCAV-style reference / quality ceiling** |
| `onmanifold_steer` | the SAE edit **projected onto the sheet** `a' = a + s·(P_M·d)` | **ours (proposed method)** |
| `clamp_steer` | clamp the SAE feature to magnitude `s`, no projection | off-manifold variant |
| `naive_steer` | the whole raw SAE edit `a' = a + s·d` | **milestone-3 baseline / competitor** |
| `random_steer` | a fixed **random** direction | null / sanity floor |

Nothing here re-implements the metric: the steerers come from `src.model`'s
`build_steer`, and the combine rule comes from `src.utils.cfs_score`. This
milestone *drives* the project's real code and **measures** the result.

---

## 3. Concepts from zero

Read this once slowly. Every later step refers back to these.

### 3.1 Activation
A list of numbers a neural network produces inside itself while looking at an
input — the model's private "notes" about one image-patch. Real CLIP ViT-B/16
notes are **768** numbers long; we use **64** so it runs instantly on a laptop.
One activation = one **point** in a 64-dimensional space (a list like
`[0.3, -1.2, ...]`, 64 entries).

### 3.2 A concept, and its concept readout
A **concept** is a human-meaningful property an image can have ("has stripes",
"is a dog"). Inside the activation space a concept is a fixed **direction** `d` (a
unit-length 64-number arrow). An image that *has* the concept has its activation
pushed a little along `d`.

A **concept readout** is the *single number* that says "how much of this concept
is in this activation". The bare-bones version is the **dot product** with the
direction: `readout(a) = <a, d> = a[0]·d[0] + a[1]·d[1] + ...`. Big readout = lots
of the concept; small readout = little.
- *Analogy:* shining a flashlight (`d`) onto an activation and reading how bright
  the spot is.
- *Tiny number:* with `d = (1, 0)`, an activation `a = (3.0, 0.2)` gives readout
  `3.0·1 + 0.2·0 = 3.0` (concept present); `a = (0.1, -0.4)` gives `0.1` (absent).

When we **steer the concept up**, this readout should *rise*. Watching it rise is
the whole monotonicity test (3.4).

### 3.3 Rank vs value
The **value** of a number is its size; its **rank** is its position when you sort
the list (1st-smallest, 2nd-smallest, ...). Many faithfulness questions only care
about the *order*, not the exact sizes.
- *Tiny number:* the values `[0.1, 0.5, 0.9]` and `[2, 40, 41]` have the **same
  ranks** `[1, 2, 3]` — both rise in the same order, even though the gaps differ
  wildly. A test built on ranks treats these two as identical.

### 3.4 Spearman rank correlation — "do they rise together, in order?"
**Spearman correlation** is the ordinary correlation computed on the *ranks* of
the values, not the raw values. It asks only "when one goes up a step, does the
other go up a step?", ignoring *by how much*. It ranges **−1 to +1**:
**+1** = perfectly same order, **0** = no order relation, **−1** = perfectly
reversed order.

*3-point worked example* — knobs `[0, 1, 2]` (ranks `1, 2, 3`):

| readout | ranks | Spearman | meaning |
|---|---|---|---|
| `[0.1, 0.5, 0.9]` | `1, 2, 3` | **+1.0** | rises in perfect order (monotone up) |
| `[0.1, 0.9, 0.5]` | `1, 3, 2` | **+0.5** | one swap — mostly up, one zig-zag |
| `[0.9, 0.5, 0.1]` | `3, 2, 1` | **−1.0** | falls in perfect order (monotone *down*) |

We use `scipy.stats.spearmanr` and **clip negatives to 0**: a steer that moves the
concept the *wrong* way is not faithful, so it scores 0, not −1. This is the
**monotonicity** component.

### 3.5 Linear probe / logistic regression — "a ruler that reads one concept"
A **probe** is a tiny model that looks at one activation (64 numbers) and reports
*one* number: "how much of concept C is in here?" Think of it as a **ruler built
for a single concept** — hold it up to any activation and it reads that concept's
level, ignoring the others. We train **one ruler per concept**.

A **linear** probe (`LogisticRegression`) is just a weighted sum of the 64
activation numbers plus a bias, squashed into a 0..1 probability:
`p = sigmoid( w·a + b )`, where `sigmoid(x) = 1/(1+e^-x)` is a soft on/off switch.
**Learning** the probe = finding the weights `w` (a 64-number arrow) and bias `b`
that best separate "concept present" (label 1) from "concept absent" (label 0) in
the labelled bank. scikit-learn does this fit.
- *Tiny number:* ruler `w = (2.0, −1.0)`, bias `−0.5`.
  Present `a = (1.5, 0.2)` → `w·a+b = +2.30` → `sigmoid = 0.91` (reads "present").
  Absent `a = (−0.3, 1.0)` → `w·a+b = −2.10` → `sigmoid = 0.11` (reads "absent").
- The learned weight vector `w` *points* in the direction the concept lives along
  — so a label-trained probe is itself a "concept direction". That is why the
  **target probe's `w`** is exactly the **TCAV-style supervised reference** (3.9).

We check each ruler on **held-out** activations it never saw; a good ruler scores
near **1.0**, a useless one near 0.5 (a coin flip).

### 3.6 Off-target drift — "did anything else move?"
When we steer the TARGET concept, we hold up the **off-target** rulers and watch
their readings. **Off-target drift** = how far an off-target reading moved between
the lowest and the highest knob setting. A *specific* steer moves the target a lot
and the off-targets not at all; an *entangled* (off-manifold) steer smears, moving
off-targets too.

**Specificity** turns drift into a [0,1] score:
`specificity = 1 − (mean off-target drift / target move)`, clipped to [0,1].
- *Tiny number:* target read moved by 6.0; off-target reads drifted by 0.3 and 0.9
  → mean drift 0.6 → `0.6 / 6.0 = 0.10` → `specificity = 1 − 0.10 = 0.90`.
- 0 drift → specificity 1.0 (perfectly specific). Drift as big as the target move
  → specificity 0.0 (smears everywhere).

### 3.7 Cohen's d / effect size — "how big was the move, in std-dev units?"
**Cohen's d** is the gap between two group means measured in **standard-deviation
units**: `d = (mean_after − mean_before) / pooled_std`. It answers "how many
standard deviations did the readout move?" — a unit-free **effect size**, so it is
comparable across concepts and backbones.
- *Analogy:* "two towns differ in height by 1.5 std-devs" is more telling than
  "3 cm" — it accounts for how spread-out heights already are.
- *Tiny number:* readout at knob 0 has mean 0.0 (std 1.0); at full knob mean 4.0
  (std 1.0) → pooled std 1.0 → `d = (4.0 − 0.0)/1.0 = 4.0` (a huge, ample effect).

We map d to a [0,1] **sufficiency** score: `sufficiency = min(d / cohen_d_ample,
1.0)`, with `cohen_d_ample = 4.0` (a 4-std-dev move counts as fully sufficient).

### 3.8 Harmonic mean — why one weak component tanks the whole score
The three components are combined with the **harmonic mean** (the reciprocal of
the average of reciprocals), not a plain average. The harmonic mean is
**conjunctive**: if *any* one component is near zero, the result is near zero.
Faithfulness needs **all three at once** — monotone **AND** specific **AND**
sufficient. A plain average would let one strong axis hide a failure in another;
the harmonic mean refuses to.
- *Tiny number:* `HM(0.9, 0.9, 0.9) = 0.90`, but `HM(0.9, 0.9, 0.05) = 0.13` —
  one weak axis tanks the whole score. (Plain average of the second triple is
  0.62, which would hide the failure.)

We call the project's `src.utils.cfs_score` so the EDA notebook and every
milestone use exactly one scoring rule.

### 3.9 TCAV — the supervised reference, in one line
**TCAV (Testing with Concept Activation Vectors, Kim et al. 2017)** = use a linear
concept direction *learned from labelled examples* to probe or steer a concept.
Here `supervised_steer` steers along the **target probe's weight vector** (a
label-trained direction) — the strong, label-expensive reference a good
*unsupervised* SAE direction should aspire to approach.

---

## 4. Prerequisites & setup

Everything runs **offline on CPU. No GPU. Nothing to download** — the "real
images" are a synthetic activation bank generated locally in `step1`. Use
**`/usr/bin/python3`** for every command (plain `python`/`python3` may point
elsewhere).

```bash
# from this folder: code/milestone_5_evaluation/
cd /Users/abroadhub/Desktop/Research/25_Rajia_Rani_FAITH_SAE/code/milestone_5_evaluation

# Check your interpreter has everything (all preinstalled on the provided box):
/usr/bin/python3 - <<'PY'
for m in ("torch","numpy","sklearn","scipy","pandas","matplotlib","yaml"):
    mod = __import__(m); print("OK", m, getattr(mod, "__version__", "?"))
PY
```

If any line says it is missing (only on a fresh machine), install the project base
then this milestone's additive line:

```bash
/usr/bin/python3 -m pip install -r ../../requirements.txt   # torch, pyyaml, numpy
/usr/bin/python3 -m pip install -r requirements.txt         # scikit-learn, scipy, pandas, matplotlib
```

---

## 5. Run it step-by-step

The fastest path is the whole pipeline in one command (≈6 seconds):

```bash
/usr/bin/python3 run_all.py
```

To learn what each stage does, run them one at a time (each reads the previous
step's files from `outputs/` and writes its own — they are independent on disk):

1. **`/usr/bin/python3 step1_build_bank.py`**
   *Why:* manufacture the multi-concept "real-image" bank on a known 8-D sheet
   (1 target + 3 off-target concepts), label every item, and train the project's
   TopK SAE on it. Writes `concept_dirs.npy`, `probe_acts.npy`, `probe_labels.npy`,
   `sae_decoder.npy`.

2. **`/usr/bin/python3 step2_train_probes.py`**
   *Why:* train one `LogisticRegression` **ruler per concept** (so we can read any
   concept off any activation) and PCA the bank into the on-manifold subspace
   **U_r**. Writes `probe_weights.npy`, `probe_bias.npy`, `U_r.npy`.

3. **`/usr/bin/python3 step3_measure_cfs.py`**
   *Why:* the heart of the milestone — sweep the steering knob for each of the five
   methods, read the concepts with the rulers, and **MEASURE** monotonicity
   (Spearman), specificity (off-target drift), sufficiency (Cohen's d), then
   combine via `cfs_score`. Writes the headline `outputs/cfs_breakdown.csv`.

4. **`/usr/bin/python3 step4_plot.py`**
   *Why:* render `outputs/cfs_breakdown.png` — four grouped bars per method
   (mono / spec / suff / CFS) so you can see which component each method loses on.

---

## 6. Expected output

After `run_all.py` you get, in `outputs/`:

- **`cfs_breakdown.csv`** — one row per method, columns:
  `variant, monotonicity, specificity, sufficiency, cfs`.
  The **real measured numbers** from this run:

  | variant | monotonicity | specificity | sufficiency | **CFS** |
  |---|---|---|---|---|
  | `supervised_steer` | 1.000 | 0.981 | 0.520 | **0.761** |
  | **`onmanifold_steer`** | **1.000** | **0.472** | **0.180** | **0.346** |
  | `clamp_steer` | 0.000 | 0.000 | 0.030 | **0.000** |
  | `naive_steer` | 1.000 | 0.000 | 0.046 | **0.000** |
  | `random_steer` | 1.000 | 0.029 | 0.042 | **0.051** |

- **`cfs_breakdown.png`** — grouped bars, four per method
  (blue = monotonicity, green = specificity, red = sufficiency, black = CFS),
  sorted best-CFS-first, with `onmanifold_steer` flagged "ours". The black CFS bar
  can never top a method's *weakest* component — the harmonic mean in action.

**Success criterion** (step3 prints `PASS`/`FAIL` for each):

> all CFS in [0,1] (**PASS**), `onmanifold_steer` CFS **>** `naive_steer` CFS
> (0.346 > 0.000, **PASS**), and `onmanifold_steer` is **among the two most
> faithful** methods (**PASS** — it ranks 2nd, just under the supervised ceiling).

---

## 7. Understand the result

Read the components, not just the headline CFS:

- **`supervised_steer` (CFS 0.761)** — the **ceiling**. It steers along the
  label-trained target direction, so it is monotone (1.0), highly specific (0.98),
  and has the largest effect (d ≈ 2.08). This is the TCAV-style reference a good
  unsupervised method should *approach but not need labels to reach*.
- **`onmanifold_steer` (CFS 0.346) — ours, and the key result.** Its profile is
  the **only balanced one besides the supervised ceiling**: it is monotone (1.0),
  *positively* specific (0.47), and has a *real* effect (suff 0.18, d ≈ 0.72). It
  is the **2nd most faithful** method — the best you can get *without labels*.
  Why balanced? Projecting the SAE edit onto the sheet (`P_M·d`) deletes the
  off-sheet part of the entangled (polysemantic) SAE feature, so the edit no longer
  shoves activations into the void where the off-target rulers misread — that is
  why specificity survives where naive's collapses.
- **`naive_steer` (CFS 0.000)** — the **mirage**. It is monotone (1.0) and *looks*
  like it works, but its specificity is **0.0**: the raw SAE edit carries the
  feature's off-sheet part into mid-air, the off-target rulers drift just as far as
  the target, and the conjunctive harmonic mean drives CFS to **0**. A big-looking
  effect that is not causally real — exactly what CFS is built to expose.
- **`clamp_steer` (CFS 0.000)** — fails on **two** axes: monotonicity 0.0 (clamping
  to a fixed magnitude does not move the readout in order as the knob turns) *and*
  specificity 0.0. CFS 0.
- **`random_steer` (CFS 0.051)** — the **floor**. A random direction has no real
  concept; it scores near zero on specificity and sufficiency, so CFS collapses.
  This is the sanity check: the metric does **not** reward "big change" alone.

**Good vs bad, at a glance:** a *good* (faithful) method has **all three bars
tall** → tall CFS (supervised, on-manifold). A *bad* method has one or more bars on
the floor → the harmonic mean zeroes the CFS no matter how tall the others are
(naive, clamp, random). The single design choice — project the edit onto the sheet
— is what moves a method from the "mirage" group toward the "real" group.

> Note: these are **synthetic, illustrative** numbers from a 64-dim toy bank with a
> tiny 60-step SAE; the *ordering and the conjunctive behaviour* are the point, not
> the absolute magnitudes. The `# REAL RUN (M5)` note below explains the real-CLIP
> swap, where the gaps widen.

---

## 8. Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: src` | ran from the wrong folder, or `sys.path` not set | run from `code/milestone_5_evaluation/`; `_common.py` adds the project root automatically |
| `FileNotFoundError: outputs/U_r.npy` (in step3) | ran a later step before an earlier one | run `step1` → `step2` → `step3` → `step4` in order, or just `run_all.py` |
| `No module named sklearn` / `scipy` / `matplotlib` | fresh machine | `/usr/bin/python3 -m pip install -r requirements.txt` |
| figure opens a window / crashes on a headless box | matplotlib tried to use a display | prefix with `MPLBACKEND=Agg` (step4 also forces `Agg` itself) |
| a probe accuracy prints ~0.5 | that concept's signal is too weak in the bank | raise `concept_strength` in `config.yaml` (the rulers must be accurate before CFS is meaningful) |

---

## 9. What's next → `milestone_6_headline_experiment`

You have now **measured** the full CFS on clean, in-distribution data and seen it
separate the real steer from the mirage. Milestone 6 turns the dial: it re-runs
this exact measurement at every rung of the **out-of-distribution (OOD) ladder**
(clean → ImageNet-R → ImageNet-Sketch → ImageNet-C severity 1–5 → ObjectNet) to
trace the **CFS-vs-shift curve** and find the **collapse knee** — the point where
faithfulness finally breaks, and whether on-manifold steering degrades more
gracefully than naive. The probes, the U_r subspace, and the `cfs_score` you built
here are exactly what that shift-loop reuses at every level.

---

### Real-run note (`# REAL RUN (M5)`)
The offline default builds a **synthetic** multi-concept bank and trains a tiny
SAE. For the real study, swap `build_labelled_bank()` for **real CLIP ViT-B/16
patch activations** over ImageNet-val with real concept annotations, train (or
load M4's) SAE, fit the probes on real activations, and reuse the M4-cached
`U_r.npy`. The knob sweep, Spearman monotonicity, off-target specificity, and
Cohen's-d sufficiency are computed **identically** — only the data changes. Each
step's `# REAL RUN (M5):` comment block spells out the swap.

---

*For research and educational purposes only.*
