# Milestone 6 — The Headline Experiment: the OOD Faithfulness Sweep

**FAITH-SAE** · author: **Rajia Rani** · ``

> Read this top to bottom. It assumes you know **nothing** about distribution
> shift, out-of-distribution data, covariate shift, severity dials, or
> degradation curves — every term is defined from zero with an analogy and a tiny
> number before it is used. By the end you will have run the **headline
> experiment of the whole project** and seen, with your own eyes, the shape of the
> curve that answers the paper's third research question: **does steering
> faithfulness survive distribution shift, and where does it collapse?**

---

## 1. Where this fits

The whole FAITH-SAE project asks one question:

> When you reach inside a frozen vision model and **steer a concept** ("make this
> image more *dog*"), is the change you cause **causally real**, or just a
> plausible-looking artifact — and **does that realness survive when the test
> images stop looking like the training images**?

The project answers it in three moves, one per research question (RQ):

- **RQ1 — the method.** On clean images, *on-manifold* steering is more faithful
  than naive steering (milestone 4).
- **RQ2 — the knobs.** Faithfulness decomposes into monotonicity × specificity ×
  sufficiency, and depends on steering strength and projection rank (milestone 5).
- **RQ3 — the sweep (this milestone, the HEADLINE).** Push the test images
  **out of distribution** and watch the Causal Faithfulness Score (CFS) fall.
  **The shape of that fall is the paper's answer.**

Milestone-by-milestone, the `code/` path is:

| Milestone | What it teaches / builds |
|---|---|
| `milestone_1_foundations` | the synthetic SAE pipeline, the four steerer names, what CFS is |
| `milestone_2_data` | the activation data (synthetic bank now; real CLIP later) |
| `milestone_3_baseline` | **naive** off-manifold steering `a' = a + s·d` — the competitor |
| `milestone_4_method` | the **method**: on-manifold steering `a' = a + s·(P_M·Δ)`, vs naive, on clean data |
| `milestone_5_evaluation` | the full **CFS metric MEASURED** (monotonicity × specificity × sufficiency) on clean data |
| **`milestone_6_headline_experiment` (you are here)** | the **OOD sweep**: re-measure CFS at every rung of the distribution-shift ladder for on-manifold vs naive — the **headline curve `fig1_cfs_ood_sweep.png`** |
| `milestone_7_ablations` | turns the design knobs (SAE type, `k`, rank `r`, …) and re-checks |

This milestone reuses everything you built before — the synthetic activation
**sheet**, the frozen **clean subspace `U_r`**, the **CFS** scoring rule — and adds
the one new ingredient that makes it the headline: a **ladder of distribution
shifts**, from clean photos all the way out to wrecked corruptions and odd
real-world poses. We measure CFS at every rung and plot the answer.

> This folder is **self-contained**: it regenerates its own clean bank, its own
> frozen `U_r`, and its own probes, so you can run it standalone without having
> run milestones 1–5 first.

---

## 2. What you build & run

You will run a 4-step offline pipeline (no downloads, CPU only) that:

1. **Builds a CLEAN, in-distribution activation bank** (`step1`) that lives on a
   thin 8-dimensional **sheet** inside a 64-dimensional space, with several
   concepts planted in (one TARGET we steer + five OFF-TARGET concepts that should
   not move).
2. **Estimates and FREEZES the clean sheet `U_r`** with PCA, and **trains the
   concept probes** (the "rulers" that read concepts back), both on clean data
   only (`step2`). Freezing them on clean is the crux of why a steer can collapse
   under shift.
3. **Sweeps the OOD ladder** (`step3`): for each rung it *corrupts* the clean bank
   to that severity, then — for **both** on-manifold and naive steering and for
   **every** concept — **MEASURES** the CFS by turning the steering knob and
   reading the probes. It bootstraps a confidence interval over concepts, finds
   the **collapse knee**, and writes `outputs/ood_cfs_sweep.csv`.
4. **Draws the HEADLINE curve** `outputs/fig1_cfs_ood_sweep.png` (`step4`): CFS vs
   shift severity, one line per method, with confidence bands, the usability
   floor, and each method's collapse knee marked.

The two methods we contrast (names fixed by the design brief, registered in
`src/blocks/__init__.py`):

| name | edit it applies | role |
|---|---|---|
| `onmanifold_steer` | `a' = a + s·(P_M·d)` (edit projected onto the frozen clean sheet) | **ours (proposed method)** |
| `naive_steer` | `a' = a + s·d` (the whole raw edit, off-sheet sliver and all) | **baseline / main competitor** |

We **reuse** the project's real code via `sys.path`: the steerers come from the
`STEER_REGISTRY`, the CFS combiner from `src.utils.cfs_score`, and the off-manifold
diagnostic from `src.utils.onmanifold_projection_residual`. Nothing here
re-implements the method — this milestone *drives* it across the shift ladder.

**Every number in the headline CSV/PNG is MEASURED from corrupted data — nothing
is a looked-up placeholder.** Measuring how CFS falls IS the experiment.

---

## 3. Concepts from zero

Read this once slowly. Every later step refers back to these. (Activations, the
sheet/manifold, PCA, `U_r`, `P_M`, and the off-manifold residual were defined from
zero in milestone 4's README; here we define the **shift** vocabulary the headline
needs.)

### 3.1 Distribution shift
**Definition.** "Distribution shift" means the test images differ from the images
the model was effectively built on. The model still *sees* an image, but it is a
**different kind** of image than it is used to.
- *Analogy:* a chef who trained only on fresh ingredients, suddenly handed
  freeze-dried ones. Same dish in principle, but the inputs are off, so the
  results get unreliable.
- *Tiny number:* if 100% of training photos were sharp colour photographs and the
  test set is 100% pencil sketches, that is a **total** input-distribution shift.

### 3.2 In-distribution vs out-of-distribution (OOD)
**Definition.**
- **In-distribution (clean):** the test images look like the build images; their
  activations sit **on** the clean sheet. This is the reference rung.
- **Out-of-distribution (OOD):** the test images differ in a way the model never
  trained on (a sketch, an art rendition, a blurry/noisy corruption, an odd
  real-world pose). Their activations drift **off** the clean sheet.
- *Analogy:* a swimmer who only ever practised in a calm pool (in-distribution),
  now in choppy ocean surf (OOD) — same skill, much harder water.
- *Tiny number:* on clean data ~96% of an activation's length lies on the 8-D
  sheet; by the worst rung the code measures **~92% of it OFF** the sheet
  (`offsheet_energy` climbs 0.25 → 0.92 in the CSV).

### 3.3 The shift ladder (the three flavours, named after real benchmarks)
We simulate the real ImageNet shift ladder offline. The rungs, in order:
- **clean** — in-distribution reference (ImageNet-val).
- **ImageNet-R (rendition):** an art/cartoon/sculpture of the object — same thing,
  very different "look". *Tiny example:* a *cartoon* dog instead of a *photo* dog.
- **ImageNet-Sketch (texture removed):** a pencil drawing — the shape is there but
  the colour/texture cues are gone.
- **ImageNet-C severity 1…5 (corruption dial):** the **same** photo, degraded —
  blur, noise, fog, JPEG — with a **severity dial** from 1 (barely) to 5 (wrecked).
- **ObjectNet:** real photos in unusual poses / backgrounds — the hardest
  real-world shift.

### 3.4 Covariate shift
**Definition.** The precise name for "the **inputs** changed but the **meaning** of
the concept did not." A dog is still a dog in a sketch; only its pixels (and so its
activations) moved. That is exactly our setting — the concept label is unchanged,
the activation has drifted.
- *Analogy:* the same word spoken in a heavy accent. The word (the concept) is
  identical; only the sound (the input) shifted.
- *Tiny number:* the target concept's planted meaning (its on-sheet read direction)
  is **fixed** across all 9 rungs; only the activation it lives in is corrupted.

### 3.5 Why a fixed `U_r` degrades under shift (the headline mechanism)
**Definition.** On-manifold steering projects the edit onto `U_r`, the clean sheet
basis — but `U_r` is **estimated on clean data and then frozen** (you can only
measure the manifold on the data you have). Once shifted activations have drifted
**off** the clean sheet, `U_r` describes the **wrong** sheet for them: the
projection `P_M = U_r U_rᵀ` points the edit at where the clean manifold *used to
be*, not where the shifted activations actually are. The clean-trained probe also
reads through ever-more off-sheet junk. Both effects drag CFS down.
- *Analogy:* a map of a river drawn last summer. Use it after the river has shifted
  its course (the shift) and your "project onto the river" routine now aims you at
  dry land. The map (`U_r`) didn't change; the world did.
- *Tiny number:* on clean data the projection captures the concept almost perfectly
  (CFS ≈ 0.91); by the worst rung, with ~92% of the activation off the clean sheet,
  CFS has fallen to ≈ 0.50 — the projection is aiming at the old riverbed.

  The question RQ3 answers is the **shape** of that fall, and whether **naive**
  (which never projected at all, so its edit is *doubly* off-manifold under shift)
  falls **at least as fast**.

### 3.6 The severity dial (ImageNet-C 1–5)
**Definition.** ImageNet-C corruptions come with a 1-to-5 **severity dial**: same
corruption type, turned up. Severity 1 is barely noticeable; severity 5 is severe.
It gives a *smooth, graded* x-axis (not just clean-vs-broken), so we can see CFS
slide rung by rung rather than jump.
- *Analogy:* a dimmer switch, not an on/off switch — you watch the room darken
  gradually.
- *Tiny number:* our offline dial grows the corruption strength monotonically:
  `gauss` (random noise) climbs 0.60 → 0.95 → 1.40 → 1.95 → 2.60 across C-1…C-5,
  and the off-sheet "style" push climbs with it.

### 3.7 Reading a degradation curve, and the "knee"
**Definition.** A **degradation curve** plots a quality number (here CFS) as
conditions get harder (here further OOD). Read it left-to-right:
- **height** = how faithful the steer is at that rung (higher = better);
- **slope** = how fast faithfulness is falling (steeper down = more fragile);
- **the KNEE** = the first rung where the curve dives **below the usability floor**
  — the rung where the concept stops being trustworthy.
- *Analogy:* a phone battery curve. It coasts flat, then there is an elbow (the
  knee) where it nose-dives to dead. The knee is the moment that matters.
- *Tiny number:* with a floor of 0.50, a curve reading
  `0.91, 0.89, 0.86, 0.84, 0.76, 0.68, 0.58, 0.50, 0.51` has its knee at the rung
  where it first touches 0.50 — here ImageNet-C severity 5.

### 3.8 A bootstrap confidence interval (over concepts)
**Definition.** We only measure a handful of concepts, so one mean CFS could be
luck. The **bootstrap** asks "how much would the mean wobble if we'd drawn a
different handful?" by **resampling** the per-concept CFS list **with replacement**
many times and reading the spread of the resampled means. The central band (e.g.
5th–95th percentile) is the shaded confidence band on the curve.
- *Analogy:* re-rolling the same bag of dice many times to see how much the average
  jiggles — that jiggle is your uncertainty.
- *Tiny number:* per-concept CFS `[0.8, 0.7, 0.9]`; one resample `[0.8, 0.8, 0.7]`
  → mean 0.767; another `[0.9, 0.9, 0.8]` → mean 0.867. Do it 400× → the band.

### 3.9 Robustness
**Definition.** "Robustness" = how well a quality survives getting harder
conditions. A robust steer keeps a **high curve** that crosses the floor **late**
(or never). The headline comparison is exactly this: does the on-manifold curve
sit **above** the naive curve and/or keep its knee **further right**?
- *Analogy:* an all-terrain tyre vs a racing slick — both grip on a dry track, but
  only one keeps gripping in the rain. Robustness is the rain performance.
- *Tiny number:* across the whole ladder the on-manifold curve averages CFS ≈ 0.73
  vs naive ≈ 0.66 — a +0.07 robustness margin that holds at **every** rung.

---

## 4. Prereqs & setup

Everything runs **offline on CPU**. Use **`/usr/bin/python3`** for every command
(plain `python`/`python3` may point elsewhere).

```bash
# from this folder: code/milestone_6_headline_experiment/
cd /Users/abroadhub/Desktop/Research/25_Rajia_Rani_FAITH_SAE/code/milestone_6_headline_experiment

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

There is **nothing to download** and **no GPU** is used: the OOD ladder is
simulated by progressively corrupting a synthetic clean activation bank generated
locally in `step1`. We set `MPLBACKEND=Agg` so plotting never tries to open a
window (it just writes the PNG).

---

## 5. Run it step-by-step

The fastest path is the whole pipeline in one command:

```bash
MPLBACKEND=Agg /usr/bin/python3 run_all.py
```

To learn what each stage does, run them one at a time (each reads the previous
step's files from `outputs/` and writes its own — they are independent on disk):

1. **`MPLBACKEND=Agg /usr/bin/python3 step1_build_clean_bank.py`**
   *Why:* manufacture the CLEAN, in-distribution activation bank on a known 8-D
   sheet, plant the target + off-target concepts (a genuine **on-sheet** read
   direction plus an **off-sheet sliver** for the SAE steering direction), and save
   the shared off-sheet "style" subspace the shift will later flood. Writes
   `outputs/clean_acts.npy`, `labels.npy`, `read_dirs.npy`, `concept_dirs.npy`,
   `sheet_basis.npy`, `style_basis.npy`.

2. **`MPLBACKEND=Agg /usr/bin/python3 step2_estimate_clean_subspace.py`**
   *Why:* on clean data only, PCA the bank to estimate and **FREEZE** the clean
   sheet `U_r` (build `P_M`, verify `trace(P_M)=r` and 100% recovery of the planted
   sheet), and train the per-concept **probes** (rulers). Writes `outputs/U_r.npy`,
   `probe_weights.npy`, `probe_bias.npy`.

3. **`MPLBACKEND=Agg /usr/bin/python3 step3_sweep_ood_cfs.py`**
   *Why:* the core measurement. For each rung of the shift ladder, corrupt the
   clean bank to that severity, then **measure** CFS (monotonicity via Spearman,
   specificity via off-target drift, sufficiency via Cohen's d) for **both** methods
   over **all** concepts, bootstrap a CI, find the collapse knee. Writes the
   headline `outputs/ood_cfs_sweep.csv` and prints the success checks.

4. **`MPLBACKEND=Agg /usr/bin/python3 step4_plot_headline.py`**
   *Why:* render the headline `outputs/fig1_cfs_ood_sweep.png` — CFS vs shift
   severity for both methods, CI bands, the usability floor, knees marked.

---

## 6. Expected output

After `run_all.py` you get, in `outputs/`:

- **`ood_cfs_sweep.csv`** — one row per `(shift_level, variant)` with columns
  `shift_level, severity_index, variant, cfs, cfs_ci_lo, cfs_ci_hi, monotonicity,
  specificity, sufficiency, offsheet_energy, offmanifold_residual`.
  Approximate values (synthetic, illustrative — yours match within seed noise):

  | shift_level | sev | on-manifold CFS | naive CFS | offsheet energy |
  |---|---|---|---|---|
  | clean | 0 | **0.91** | 0.81 | 0.25 |
  | imagenet_r | 1 | 0.89 | 0.80 | 0.55 |
  | imagenet_sketch | 2 | 0.86 | 0.78 | 0.68 |
  | imagenet_c_sev1 | 3 | 0.84 | 0.76 | 0.70 |
  | imagenet_c_sev2 | 4 | 0.76 | 0.70 | 0.81 |
  | imagenet_c_sev3 | 5 | 0.68 | 0.62 | 0.87 |
  | imagenet_c_sev4 | 6 | 0.58 | 0.54 | 0.90 |
  | imagenet_c_sev5 | 7 | **0.50** | **0.46** | 0.92 |
  | objectnet | 8 | 0.51 | 0.48 | 0.92 |

- **`fig1_cfs_ood_sweep.png`** — the HEADLINE curve: two descending lines
  (green = on-manifold, red = naive), each with its 90% confidence band; the
  dashed **usability floor = 0.50**; and each method's **collapse knee** marked
  (here at ImageNet-C severity 5). The on-manifold line sits **above** naive at
  every rung. The `offsheet_energy` column rising 0.25 → 0.92 is the activations
  visibly leaving the clean sheet — the mechanism in numbers.

**Success criterion** (the run prints `PASS`/`FAIL` for each):

> 1. **All CFS in [0, 1]** across every rung and method (a sane metric), **and**
> 2. **naive collapses at least as fast as on-manifold** — naive's collapse knee is
>    at the **same or an earlier** severity than on-manifold's (and on-manifold's
>    mean CFS across the ladder is **≥** naive's).

---

## 7. Understand the result — does faithfulness survive shift?

The curve answers RQ3 directly, and **either** answer is publishable:

- **If the curves stay above the floor far out** → vision-SAE concept steering is
  **trustworthy under shift** (a green light for the field: interpret away).
- **If the curves dive below the floor early** → clean-data faithfulness **does not
  transfer OOD** (a warning to the field: a steer that looks faithful on ImageNet-val
  may be a mirage on a sketch).

The offline run lands in between, which is the honest and interesting case:
**faithfulness largely survives mild–moderate shift (renditions, sketch, light
corruption) but collapses at the most severe corruptions and the hardest real-world
shift** — both curves cross the usability floor around ImageNet-C severity 5. That
is a real, falsifiable claim with a clearly marked knee.

The **method contribution holds across the whole ladder**: the on-manifold curve
sits **above** the naive curve at **every** rung, and its knee comes **no earlier**.
Why? On clean data, on-manifold trims the SAE direction's off-sheet **sliver** and
puts its full strength on the genuine on-sheet concept, so it is **more specific**
(off-target probes barely move: specificity ≈ 0.80 vs naive's ≈ 0.66). Under shift,
the off-sheet directions the naive edit injects into are **exactly** the ones the
shift floods with junk (they share the "style" subspace) — so naive's wasted
off-sheet strength **collides** with the shift and its readout corrupts faster.
On-manifold put nothing there, so it degrades more gracefully. The single change —
`a' = a + s·(P_M·Δ)` instead of `a' = a + s·Δ` — buys a robustness margin that the
off-manifold residual (0.00 for on-manifold vs ~0.63 for naive in the CSV) proves
is real, not luck.

---

## 8. Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: src` | ran from the wrong folder, or `sys.path` not set | run from `code/milestone_6_headline_experiment/`; `_common.py` adds the project root automatically |
| `FileNotFoundError: outputs/U_r.npy` (in step3) | ran step3 before step2 | run `step1` → `step2` → `step3` in order, or just `run_all.py` |
| a plot window tries to open / the run hangs | matplotlib chose a GUI backend | prefix commands with `MPLBACKEND=Agg` (the steps also force `Agg` internally) |
| `No module named sklearn`/`scipy`/`pandas`/`matplotlib` | fresh machine | `/usr/bin/python3 -m pip install -r requirements.txt` |
| `command not found: /usr/bin/python3` | non-macOS / different layout | use the interpreter that has torch+sklearn+scipy+pandas; the contract assumes `/usr/bin/python3` |
| a CFS prints as exactly 0.0 | one component (often monotonicity) hit 0 — the harmonic mean is conjunctive | expected at extreme shift; if it happens on **clean**, lower `steer_strength` or raise `concept_strength` so the knob moves the readout |
| both curves identical | `concept_offsheet_frac` set to 0 (no sliver to trim) → on-manifold = naive | keep `concept_offsheet_frac > 0` (default 0.45): with no off-sheet sliver there is nothing for projection to fix |
| no visible knee (both stay above the floor) | the ladder's worst rungs aren't harsh enough | raise the `gauss`/`style` of the high-severity rungs in `config.yaml`, or raise `cfs_floor` |

---

## 9. What's next → `milestone_7_ablations`

You measured the headline OOD curve with the **default** design knobs. Milestone 7
turns those knobs and re-checks the answer: SAE type (TopK vs L1), the sparsity
level `k`, the **manifold-projection rank `r`** (the core A3 knee — too small
over-constrains the edit and the effect dies; too large lets it drift off-manifold
and on-manifold degenerates into naive), the concept-selection threshold (the
field's "only ~10–15% of features steer reliably" claim), and the backbone layer /
token choice. Each ablation re-runs a slimmed version of this very sweep, so the
robustness margin you just measured is shown to be a property of the **method**,
not of one lucky knob setting. The frozen clean `U_r` and the CFS scoring rule you
used here carry straight over.

---

### Real-run note (`# REAL RUN (M6)`)
The offline default **simulates** the OOD ladder by progressively corrupting a
synthetic clean activation bank (growing Gaussian noise + an off-sheet "style"
rotation, one setting per rung). For the real study, replace the simulated ladder
with **real shifted activations**: run a frozen **CLIP ViT-B/16** over each real
dataset — **ImageNet-R, ImageNet-Sketch, ImageNet-C at severities 1–5, and
ObjectNet** — cache the patch activations per rung, and feed each cached bank in
where `step3` uses `shifted`. **Everything else is identical:** `U_r` and the
probes stay **frozen on clean ImageNet-val**, the knob sweep / Spearman
monotonicity / off-target specificity / Cohen's-d sufficiency measurement is
unchanged, the bootstrap CI is over the real concept set, and the same
`step4_plot_headline.py` renders the paper's `fig1_cfs_ood_sweep.png`. Each step's
`# REAL RUN (M6):` comment block spells out the swap.

---

*For research and educational purposes only.*
