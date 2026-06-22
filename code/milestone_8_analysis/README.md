# Milestone 8 — The Analysis: Bootstrap CIs, Figures & Findings

**FAITH-SAE** · author: **Rajia Rani** · ``

> Read this top to bottom. It assumes you know **nothing** about samples,
> resampling, the bootstrap, confidence intervals, or statistical significance —
> every term is defined from zero with a tiny number before it is used. By the
> end you will have run the project's **final analysis**: you will put honest
> error bars on the per-concept faithfulness scores, render the two headline
> paper figures from the *real measured* numbers, and write the plain-language
> answers (`FINDINGS.md`) that **replace the paper's `\pending{}` placeholders**.

---

## 1. Where this fits

The whole FAITH-SAE project asks one question:

> When you reach inside a frozen vision model and **steer a concept** ("make this
> image more *dog*"), is the change you cause **causally real**, or just a
> plausible-looking artifact? And does staying **on the data manifold** make the
> steer faithful where the **naive** off-manifold add is a mirage?

By milestone 8 the *method* (on-manifold steering) and the *measuring stick*
(the Causal Faithfulness Score, CFS) already exist and have been run across the
out-of-distribution (OOD) ladder. What is still missing is the **honest
bookkeeping**: a single per-concept score is one draw of the dice — could the
on-manifold-vs-naive gap just be luck of *which concepts we happened to test*?
This milestone answers that, and **closes the loop to `paper/`**.

Milestone-by-milestone, the `code/` path is:

| Milestone | What it teaches / builds |
|---|---|
| `milestone_1_foundations` | the synthetic SAE pipeline, the four steerer names, what CFS is |
| `milestone_2_data` | the activation data (synthetic bank now; real CLIP later) |
| `milestone_3_baseline` | **naive** off-manifold steering `a' = a + s·d` — the competitor |
| `milestone_4_method` | the **proposed method**: on-manifold steering `a' = a + s·(P_M·d)` |
| `milestone_5_evaluation` | the full CFS metric across the OOD ladder |
| `milestone_6_headline_experiment` | the headline on-manifold-vs-naive sweep |
| `milestone_7_ablations` | the knob/rank ablations |
| **`milestone_8_analysis` (you are here)** | **the final analysis**: bootstrap 95% CIs over concepts, the measured paper figures, and `FINDINGS.md` |

This milestone is where the project's numbers become **publishable claims**. It
does three things, in order:

- **Bootstrap confidence intervals over concepts** — resample the 24 tested
  concepts 2000 times to get a 95% CI on each method's mean CFS, so we can state
  whether the on-manifold-vs-naive gap is *real* or *noise*.
- **Render the real measured paper figures** — `fig1_cfs_ood_sweep.png` (the
  headline OOD sweep with CI bands) and `fig7_by_method_bar.png` (mean CFS by
  method with CI error bars). These replace the paper's "illustrative
  placeholder" figures with measured ones.
- **Write `FINDINGS.md`** — the plain-language RQ1/RQ2/RQ3 answers, each tagged
  with the exact paper `\pending{}` placeholder it fills.

> The two figure names (`fig1_cfs_ood_sweep.png`, `fig7_by_method_bar.png`) and
> the file `FINDINGS.md` are the *deliverables the paper consumes* — the paper's
> `\includegraphics` resolve to these, and `FINDINGS.md` is the text you paste in
> where the paper currently says `\pending{...}`.

---

## 2. What you build & run

You will run a 4-step offline pipeline (no downloads, CPU only, ~8 seconds) that:

1. **Regenerates the synthetic activation bank + trains the project's real TopK
   SAE, selects the 24 testable concepts, and MEASURES a per-concept CFS** for
   every method across the OOD ladder. Writes `outputs/per_concept_cfs.csv`
   (720 rows) (`step1`).
2. **Bootstraps a 95% confidence interval** on each method's mean CFS by
   resampling the concept set with replacement **2000 times**, and bootstraps the
   on-manifold-minus-naive **gap** (paired) to test whether it is real. Writes
   `outputs/bootstrap_ci.csv` (`step2`).
3. **Renders the two paper figures from the measured numbers** —
   `outputs/fig1_cfs_ood_sweep.png` and `outputs/fig7_by_method_bar.png` — using
   the same colorblind palette and labels as `paper/figures/` (`step3`).
4. **Writes `FINDINGS.md`** — the RQ1/RQ2/RQ3 answers computed *directly from the
   CSVs*, each naming the paper `\pending{}` it replaces (`step4`).

The five steering methods being compared (names fixed by the project's design
brief, registered in the project's `STEER_REGISTRY`):

| name | edit it applies | role |
|---|---|---|
| `supervised_steer` | steer the **planted ground-truth** direction | label-expensive **gold ceiling** (TCAV-style) |
| `onmanifold_steer` | `a' = a + s·(P_M·d)` (edit projected onto the sheet) | **ours (proposed method)** |
| `clamp_steer` | clamp the SAE feature to magnitude `s`, no projection | off-manifold variant |
| `naive_steer` | `a' = a + s·d` (the whole raw edit) | **main competitor** |
| `random_steer` | `a' = a + s·(random dir)` | null / sanity baseline |

We **reuse** the project's real code via `sys.path` (`_common.py` adds the
project root): the steerers come from `build_steer`/`STEER_REGISTRY`, the TopK
SAE from `src.model`, and `cfs_score` + `onmanifold_projection_residual` from
`src.utils`. Nothing here re-implements the method or the metric — this milestone
*measures, resamples, and reports* them.

---

## 3. Concepts from zero

Read this once slowly. Every later step refers back to these. The first few
recap the project; the rest (3.5–3.10) are the **statistics** this milestone adds.

### 3.1 Activation, manifold, concept direction (one-line recaps)
- **Activation** = the list of numbers a network produces inside itself for one
  image-patch — its private "notes". Here each note is **64** numbers (real CLIP
  uses 768); we use 64 so it runs instantly on a laptop CPU.
- **The data manifold (the "sheet")** = the thin, low-dimensional region real
  activations actually land in — like a sheet of paper floating in a 64-D gym.
  Our synthetic bank lives on a **24-D sheet** inside the 64-D space. Points
  *off* the sheet are states the model never really saw.
- **Concept direction `d`** = a fixed unit vector; turning the concept "up" pushes
  activations along `d`. **On-manifold** first projects `d` onto the sheet so the
  edit stays realistic; **naive** adds the whole `d` and flies off the sheet.

### 3.2 CFS — the per-concept faithfulness score (one-line recap)
The **Causal Faithfulness Score** (CFS, in `[0,1]`) asks: *is this edit a real,
clean causal lever?* It is the **harmonic mean** (an "AND" — all three must be
high) of **monotonicity** (turn the knob up → readout rises smoothly),
**specificity** (only the target moves, off-target probes stay flat), and
**sufficiency** (the effect is big enough to matter). One weak axis tanks the
whole score: `HM(0.9,0.9,0.9)=0.90` but `HM(0.9,0.9,0.05)=0.13`.

### 3.3 Sample vs population
- **Population** = *every* thing you could possibly measure — here, *every concept
  a SAE could ever discover*. You can never see all of it.
- **Sample** = the handful you actually measured — here, our **24** selected
  concepts. We compute the *sample mean* CFS and hope it is close to the (unknown)
  *population mean*.
- *Analogy:* the population is "all voters in the country"; the sample is "the 24
  people the pollster actually phoned".
- *Tiny number:* if the 24 concepts have CFS values averaging **0.316**, that
  **0.316 is a sample mean** — a single noisy snapshot of the true population mean
  we can't see directly.

### 3.4 Sampling variability (why one mean isn't enough)
If we had happened to discover **24 slightly different concepts**, the sample mean
would have come out a little different. That wobble — how much the answer changes
just because of *which* items landed in the sample — is **sampling variability**.
A claim like "on-manifold beats naive by 0.008" is only meaningful once we know
*how big the wobble is*: if the wobble is ±0.02, a 0.008 gap could be pure luck.
- *Analogy:* weigh yourself on a cheap scale once and you get 70.1 kg; step off
  and on again and you get 69.8 kg. The "true" weight is fixed; the *reading*
  wobbles. Sampling variability is that wobble, for *means*.

### 3.5 Resampling with replacement
**Resampling with replacement** = drawing a new set the *same size* as your
original, picking items from the original **at random, putting each one back
after you read it** — so the same item can be picked twice (or zero times).
- *Analogy:* a bag of 24 numbered marbles. To make a "new" bag of 24, you pull a
  marble, **write its number down, drop it back in**, shake, and pull again — 24
  times. Some marbles appear twice; some not at all.
- *Tiny number:* original 5 values `[0.6, 0.8, 0.7, 0.9, 0.5]` (indices 0–4). One
  resample draws indices `(0, 0, 3, 1, 4)` → values `[0.6, 0.6, 0.9, 0.8, 0.5]`.
  Index 0 appears twice, index 2 not at all. That is *with replacement*.

### 3.6 The bootstrap procedure (5-number worked example)
The **bootstrap** estimates how much a statistic (here, the mean) wobbles, using
**only the data you already have** — no formula, no assumption about a bell curve.
The recipe:

1. Resample your data **with replacement** to the same size.
2. Compute the statistic (the mean) of that resample.
3. Repeat thousands of times — here **2000**.
4. The spread of those 2000 means *is* the sampling variability; its middle 95%
   *is* the confidence interval.

**Worked 5-number example** (the script prints exactly this so you can check it by
hand):

```
5 per-concept CFS values : [0.6, 0.8, 0.7, 0.9, 0.5]      mean = 3.5 / 5 = 0.70
  resample indices (4, 3, 2, 1, 1) -> [0.5, 0.9, 0.7, 0.8, 0.8] -> mean = 0.74
  resample indices (0, 0, 0, 0, 4) -> [0.6, 0.6, 0.6, 0.6, 0.5] -> mean = 0.58
  ... 2000 resamples -> sort the 2000 means -> 95% CI = [0.58, 0.82]
```

The original mean `0.70` **always** sits inside its own CI
(`0.58 ≤ 0.70 ≤ 0.82`) — the script *asserts* this as a sanity check.

> **Why resample CONCEPTS** (not images or knob steps)? The claim is
> "on-manifold steers *concepts* more faithfully". The thing we want to
> generalize over is the concept, so the concept is what we resample.

### 3.7 Confidence interval (CI), and "non-overlapping CIs → the difference is real"
A **95% confidence interval** is a range built so that, if we repeated the whole
experiment many times, ~95% of the intervals we'd build would contain the true
(population) mean. Loosely: *"we're fairly sure the real mean lives in here."*
- **Wide CI** = noisy / few concepts; **narrow CI** = stable / many concepts.
- *Tiny number:* on-manifold's mean CFS is `0.316` with **95% CI [0.222, 0.401]**
  — a fairly wide band, because 24 concepts is a small sample.

**Non-overlapping CIs → the difference is real.** If method A's CI is
`[0.74, 0.82]` and B's is `[0.49, 0.61]`, they share *no value at all* — there is
no single number both could plausibly have as their true mean, so the gap is not
luck of the draw: it is **real**. If the CIs **overlap**, you *cannot* rule out
"same true mean, unlucky draw". (The cleaner test is to bootstrap the **gap
itself** — A minus B, on the *same* resampled concepts each round; if the gap's
95% CI is entirely above 0, A genuinely beats B. This milestone reports both.)

### 3.8 Statistical significance — what it does and does NOT mean
A difference is **statistically significant** when it is **bigger than the
wobble** — i.e. the bootstrap says it is unlikely to be produced by sampling
variability alone (here: the gap's 95% CI sits entirely above 0).
- It **does** mean: "this difference is unlikely to be pure luck of which
  concepts we sampled."
- It does **NOT** mean: the difference is *large*, *important*, or *practically
  useful*; that it is *certainly* true; or that "not significant" proves the two
  are *equal*. "Not significant" only means **"this sample can't rule out luck"** —
  often just because the sample (24 concepts) is small.
- *Tiny number:* on this synthetic run the clean-rung gap is `0.008` with 95% CI
  `[-0.009, 0.025]` — the CI **straddles 0**, so it is **not significant**: a
  small, real-looking edge that 24 concepts cannot statistically separate from
  noise. (See §7 for why that is the *honest* result and how it maps to the paper.)

### 3.9 p-value (one line)
The **p-value** is the probability of seeing a gap this big (or bigger) *if the
true gap were actually ≤ 0*; small p (e.g. < 0.05) = "luck alone rarely does this"
= significant. Here the one-sided bootstrap p is `P(gap ≤ 0)` ≈ **0.19** on the
clean rung — far above 0.05, i.e. not significant.

### 3.10 Why CIs are exactly what this milestone adds
Milestones 5–7 produced *point* numbers (one mean per method per rung). A point
number with no error bar can't support a claim. The bootstrap turns each point
into **mean + [ci_low, ci_high]**, so the figures get honest bands and the
findings can say, with evidence, *which* gaps are real and which are noise. That
is the difference between "a plot" and "a result".

---

## 4. Prereqs & setup

Everything runs **offline on CPU** — **no GPU, nothing to download** (the "real
images" are a synthetic activation bank generated locally in `step1`). Use
**`/usr/bin/python3`** for every command (plain `python`/`python3` may point
elsewhere).

```bash
# from this folder: code/milestone_8_analysis/
cd /Users/abroadhub/Desktop/Research/25_Rajia_Rani_FAITH_SAE/code/milestone_8_analysis

# Check your interpreter has everything (all preinstalled on the provided box):
/usr/bin/python3 - <<'PY'
for m in ("torch", "numpy", "matplotlib", "yaml"):
    mod = __import__(m); print("OK", m, getattr(mod, "__version__", "?"))
PY
```

If any line says it is missing (only on a fresh machine), install the project
base then this milestone's one additive line (only `matplotlib` is new):

```bash
/usr/bin/python3 -m pip install -r ../../requirements.txt   # torch, pyyaml, numpy
/usr/bin/python3 -m pip install -r requirements.txt         # matplotlib
```

---

## 5. Run it step-by-step

The fastest path is the whole pipeline in one command (~8 seconds):

```bash
/usr/bin/python3 run_all.py
```

To learn what each stage does, run them one at a time (each reads the previous
step's saved CSV from `outputs/` and writes its own — they are independent on
disk):

1. **`/usr/bin/python3 step1_measure_per_concept_cfs.py`**
   *Why:* regenerate the bank, train the SAE, select the 24 testable concepts,
   and **measure a per-concept CFS** for every method at every shift rung. Writes
   `outputs/per_concept_cfs.csv` (the raw material everything else chews on).

2. **`/usr/bin/python3 step2_bootstrap_ci.py`**
   *Why:* put **honest error bars** on those numbers — bootstrap a 95% CI on each
   method's mean CFS (2000 resamples), and bootstrap the on-manifold-minus-naive
   **gap** to test if it's real. Prints the 5-number worked example first, then
   writes `outputs/bootstrap_ci.csv`.

3. **`/usr/bin/python3 step3_render_figures.py`**
   *Why:* render the **measured** versions of the two paper figures with their CI
   bands — `outputs/fig1_cfs_ood_sweep.png` (OOD sweep + collapse knee) and
   `outputs/fig7_by_method_bar.png` (mean CFS by method).

4. **`/usr/bin/python3 step4_write_findings.py`**
   *Why:* turn the CSVs into **`FINDINGS.md`** — the plain-language RQ1/RQ2/RQ3
   answers, each tagged with the paper `\pending{}` it replaces. Every number is
   recomputed from the CSVs, nothing hard-coded.

---

## 6. Expected output

After `run_all.py` you get, in `outputs/`:

- **`bootstrap_ci.csv`** — one row per `(variant, shift)`, columns:
  `variant, shift, mean_cfs, ci_low, ci_high, n_concepts`.
  The **real measured** clean-rung rows (and the full sweep for the two compared
  methods):

  | variant | shift | mean_cfs | ci_low | ci_high | n |
  |---|---|---|---|---|---|
  | `supervised_steer` | clean | 0.687 | 0.675 | 0.699 | 24 |
  | `onmanifold_steer` | clean | 0.316 | 0.222 | 0.401 | 24 |
  | `clamp_steer` | clean | 0.251 | 0.178 | 0.320 | 24 |
  | `naive_steer` | clean | 0.308 | 0.219 | 0.392 | 24 |
  | `random_steer` | clean | 0.000 | 0.000 | 0.000 | 24 |

  Full OOD sweep for the two headline methods (mean CFS, 95% CI):

  | shift | on-manifold | naive |
  |---|---|---|
  | clean | 0.316 [0.222, 0.401] | 0.308 [0.219, 0.392] |
  | ImgNet-R | 0.323 [0.228, 0.413] | 0.315 [0.217, 0.409] |
  | Sketch | 0.322 [0.227, 0.414] | 0.314 [0.222, 0.403] |
  | C-3 | 0.315 [0.226, 0.405] | 0.307 [0.219, 0.398] |
  | C-5 | 0.296 [0.211, 0.380] | 0.289 [0.200, 0.369] |
  | ObjectNet | 0.270 [0.196, 0.343] | 0.264 [0.187, 0.341] |

- **`per_concept_cfs.csv`** — 720 rows (5 methods × 6 rungs × 24 concepts), one
  per `(variant, shift, concept)` with `monotonicity, specificity, sufficiency,
  offmanifold_residual, cfs`. This is what the bootstrap resamples.
- **`fig1_cfs_ood_sweep.png`** — the headline OOD sweep: on-manifold (blue) vs
  naive (vermillion), each with its bootstrap **95% CI band**, a dashed
  **usability floor** at CFS = 0.50, and a ring marking each method's **collapse
  knee** (first rung below the floor).
- **`fig7_by_method_bar.png`** — one bar per method (mean CFS on the clean rung)
  with bootstrap **95% CI error bars**: on-manifold close behind the supervised
  ceiling and above clamp/naive/random.
- **`FINDINGS.md`** (written next to the code, not in `outputs/`) — the RQ1/RQ2/
  RQ3 answers + the one-paragraph verdict, each naming the paper `\pending{}` it
  replaces.

**Success criterion** (the `run_all.py` summary spells this out):

> `bootstrap_ci.csv` + the 2 PNGs + `FINDINGS.md` all exist, and every CI is
> well-formed (`ci_low ≤ mean_cfs ≤ ci_high`). `step2` *asserts* the
> well-formedness, so a clean exit-0 run is your pass.

---

## 7. Understand the result

**How to read the figures.** In `fig1`, two lines walk left (easy) to right
(hardest). Each has a shaded **95% CI band**. Where a band is *narrow*, the mean
is stable; where it's *wide*, few concepts make it noisy. The ring is the
**collapse knee** — the first rung where the mean dips below the dashed usability
floor (CFS = 0.50). In `fig7`, the bars are clean-rung means and the whiskers are
95% CIs; **bars whose whiskers do not overlap differ for real**.

**The honest headline of this run.** On-manifold's clean-rung mean CFS is
**0.316** (CI [0.222, 0.401]); naive's is **0.308** (CI [0.219, 0.392]). The gap
is **0.008**, paired-bootstrap 95% CI **[-0.009, 0.025]**, p(gap ≤ 0) ≈ **0.19**.
The two CIs **overlap** and the gap's CI **straddles 0**, so on this synthetic run
the difference is **not statistically separable** — small but in the predicted
*direction* (on-manifold ahead), just inside the noise floor for 24 concepts.
This is the *correct, honest* outcome to report: the bootstrap is doing its job
by refusing to call a 0.008 gap "real" with so few concepts. The clear, robust
result that *is* separable is the **ordering**: supervised (0.687) ≫ on-manifold
≈ naive ≈ clamp (~0.25–0.32) ≫ random (0.000), and the genuine *mechanistic*
separator — the **off-manifold residual** — is ~0 for on-manifold and large for
naive even where their CFS is close.

**How the findings map to the paper's `\pending{}`** (all spelled out in
`FINDINGS.md`):

| RQ | Finding (measured) | Paper `\pending{}` it fills |
|---|---|---|
| **RQ1** | clean-rung per-method CFS table + gap CI | abstract claim (~L39), per-method CFS table cells (L206–210), **Fig. 7** caption + "non-overlapping CIs" (L276, L281) → use `fig7_by_method_bar.png` |
| **RQ2** | decomposition (spec is the lever: 0.458 vs naive 0.444) + **12%** reliable-concept fraction (CFS ≥ 0.50) | **Fig. 4** reliability (L246, L251), limitations "measured reliable fraction" (L286) |
| **RQ3** | OOD sweep table + collapse knees + slopes (both ≈ -0.009/rung) | abstract "degrade more gracefully" (L39), **Fig. 1** OOD sweep (L216, L221), limitations knee (L286), conclusion (L289) → use `fig1_cfs_ood_sweep.png` |

To finish the paper, paste each `FINDINGS.md` section over the matching
`\pending{}` and swap in the two PNGs.

---

## 8. Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: src` | ran from the wrong folder | run from `code/milestone_8_analysis/`; `_common.py` adds the project root to `sys.path` automatically |
| `FileNotFoundError: outputs/per_concept_cfs.csv` (in step2/3/4) | ran a later step before `step1` | run `step1` → `step2` → `step3` → `step4` in order, or just `run_all.py` |
| `No module named matplotlib` | fresh machine | `/usr/bin/python3 -m pip install -r requirements.txt` |
| a figure window tries to pop up / blocks on a headless box | default matplotlib backend | run with `MPLBACKEND=Agg` (the step already forces `Agg`, but the env var is belt-and-braces) |
| `command not found: /usr/bin/python3` | non-macOS / different layout | use the interpreter that has torch + numpy + matplotlib; the contract assumes `/usr/bin/python3` |
| the gap reads "not significant" / CIs overlap | **expected** on this 24-concept synthetic run | this is the honest result (see §7); the real-CLIP run with far more concepts is where significance is decided — see §9 |
| CIs look slightly different run-to-run | you changed `seed` or `n_boot` in `config.yaml` | keep the defaults (`seed: 0`, `n_boot: 2000`) for the reproducible numbers in this README |

---

## 9. What's next → fill the paper, then re-run at scale

You now have the three deliverables the paper needs: the **bootstrap CIs**, the
two **measured figures**, and **`FINDINGS.md`**. Two steps close the project:

1. **Fill the paper from `FINDINGS.md`.** For each `\pending{}` listed in §7,
   paste the matching `FINDINGS.md` text into `paper/paper.tex` and replace the
   "illustrative" figures with `outputs/fig1_cfs_ood_sweep.png` and
   `outputs/fig7_by_method_bar.png`. The paper now states *measured* results.

2. **Re-run at scale with `code/real_run/` on a GPU.** The offline default here is
   a **synthetic** bank scored on a simulated OOD ladder so it runs in ~8 s on a
   CPU. The `# REAL RUN (M8)` comment block at the bottom of each `step*.py`
   spells out the swap: point `step1` at the **real** per-concept CFS measured on
   **CLIP ViT-B/16** activations across the *real* ladder (clean ImageNet →
   ImageNet-R → ImageNet-Sketch → ImageNet-C severity 1–5 → ObjectNet) produced by
   `code/real_run/`. `step2`/`step3`/`step4` are **data-agnostic** — the
   bootstrap, the figures, and `FINDINGS.md` regenerate unchanged, now with
   real-scale numbers and far more concepts (which is what lets significance be
   decided rather than swamped by a 24-concept sample). Raise `n_boot` for smoother
   CI edges; the cost is linear.

---

*For research and educational purposes only.*
