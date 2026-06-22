# Milestone 2 — The Data Pipeline & EDA

### FAITH-SAE · taught from absolute zero · author: Rajia Rani

> **What you will be able to do after this folder:** explain — to anyone, with
> no prior knowledge — exactly *what data* the FAITH-SAE project studies, *where
> it comes from*, and *what it looks like*; and run a complete, offline pipeline
> that **manufactures activations shaped exactly like a real CLIP ViT-B/16 vision
> model would produce**, saves them, and inspects them with proper EDA.

This README assumes you know **nothing**. Every term is defined the first time it
appears: a plain definition, an everyday analogy, and a tiny number you can check
by hand. Read it top to bottom.

---

## 1. Where this fits

FAITH-SAE asks one question (see `../../DESIGN_BRIEF.md`):

> *Are the "concept directions" a Sparse Autoencoder finds inside a frozen vision
> model **causally faithful** — and do they stay faithful when the input images
> get weird (out of distribution)?*

To even begin, you need the **raw material** the whole project chews on:
**activations** — the internal numbers a vision model computes while looking at an
image. Everything downstream is built on these numbers:

```
        THIS MILESTONE (M2)                       LATER MILESTONES
  ┌───────────────────────────────┐     ┌──────────────────────────────────┐
  │ images  ->  CLIP ViT-B/16  ->  │     │ M3: train a Sparse Autoencoder   │
  │ ACTIVATIONS (patch tokens)  -> │ --> │     on these activations; find    │
  │ save them; LOOK at them (EDA) │     │     concept directions; steer them │
  └───────────────────────────────┘     │ M4: measure faithfulness (CFS)    │
                                          │     across the OOD ladder         │
                                          └──────────────────────────────────┘
```

- **M1 (foundations)** set up the project skeleton and the synthetic
  planted-concept smoke test.
- **M2 (this folder)** builds the *real data pipeline*: how activations are
  produced and extracted, what the OOD ("harder images") ladder is, and a full
  EDA. Because this laptop has **no GPU, no CLIP weights, and no image datasets**,
  the default path **manufactures a synthetic activation bank shaped exactly like
  CLIP ViT-B/16 patch tokens** and runs the whole EDA on it. The real CLIP path
  is written out and clearly marked `# REAL RUN (M2):` for when you have the
  hardware and data.
- **M3 (baseline)** is next: it consumes `outputs/activations.npz` to train the
  Sparse Autoencoder. See [§9](#9-whats-next--milestone_3_baseline).

---

## 2. What you build & run here

Two small, heavily-commented scripts:

| Script | What it does | Output |
|---|---|---|
| `step1_build_synthetic_bank.py` | Manufactures a bank of activations shaped **exactly** like CLIP ViT-B/16 patch tokens (`200 images × 197 tokens × 768 dims`), with a realistic low-rank "manifold", a couple of planted "concept" directions, and an OOD-shift knob. | `outputs/activations.npz` |
| `step2_eda.py` | Loads the bank and runs **Exploratory Data Analysis**: per-dimension mean/variance, sparsity, token-count check, a 2-D PCA scatter, plus a clean-vs-OOD comparison. | `outputs/eda_summary.csv`, `outputs/eda_overview.png`, `outputs/ood_shift.png` |

Plus `config.yaml` (every number, explained) and `requirements.txt` (additive).

You also get, written but **not run offline**, the real pipeline:
`build_real_clip_bank()` in `step1`, gated behind `# REAL RUN (M2):`. It shows
exactly how to load the frozen CLIP model and extract patch-token activations
from real images. **Do not run it now** — it would download multi-GB weights and
datasets.

---

## 3. Concepts from zero

Read this section once and the code reads itself. Each concept: **definition →
analogy → tiny number.**

### 3.1 Image
- **Definition.** A digital image is a grid of tiny colored dots called *pixels*.
- **Analogy.** A mosaic: step back and you see a cat; step close and it is just
  colored tiles.
- **Tiny number.** A `224×224` color image (CLIP's input size) is
  `224 × 224 × 3 = 150,528` numbers (the `3` is Red, Green, Blue brightness per
  pixel).

### 3.2 Pixel tensor
- **Definition.** A **tensor** is just a box of numbers with a shape. An image
  becomes a pixel tensor of shape `[3, 224, 224]` (3 color channels, 224 rows,
  224 columns), each number a brightness in `[0, 1]`.
- **Analogy.** A spreadsheet, but 3-D: three stacked sheets (R, G, B), each
  `224×224`.
- **Tiny number.** The top-left pixel might be `R=0.50, G=0.20, B=0.10` — a dark
  reddish-brown dot. Three numbers.

### 3.3 A vision backbone (the model that "looks")
- **Definition.** A **backbone** is a big trained neural network that turns a
  pixel tensor into a compact, meaningful list of numbers (an *embedding*). Two
  common designs: a **CNN** (Convolutional Neural Network — slides small filters
  over the image to detect edges, then shapes, then objects) and a
  **Transformer** (chops the image into patches and lets every patch "talk to"
  every other patch via *attention*). FAITH-SAE uses a Transformer backbone:
  **CLIP ViT-B/16**.
- **Analogy.** A factory line: raw pixels go in one end; by the far end the
  network has distilled "this is a striped cat on grass" into a few hundred
  numbers.
- **Tiny number.** ViT-B/16's internal width is **768** — every patch is
  described by 768 numbers as it flows through the network.

### 3.4 ViT patches & patch tokens vs the CLS token
- **Definition.** A **ViT** (Vision Transformer) cuts a `224×224` image into a
  grid of `16×16`-pixel **patches**: `224 / 16 = 14`, so `14 × 14 = 196` patches.
  Each patch is turned into a 768-number vector called a **patch token**. The ViT
  also prepends one extra special vector, the **CLS token** ("classification"
  token), meant to summarize the *whole* image. Total tokens: `196 + 1 = 197`.
- **Analogy.** Reading a comic page: each of the 196 panels (patches) is one
  *patch token*; the one-line plot summary at the bottom is the *CLS token*.
- **Tiny number.** One image → `197` tokens × `768` numbers each →
  a `[197, 768]` activation tensor. **FAITH-SAE trains its Sparse Autoencoder on
  the 196 PATCH tokens** (rich, local, many per image), **not** the single CLS
  token — patch tokens give thousands of diverse activation vectors per batch,
  which is what an SAE needs.

### 3.5 An embedding / activation vector
- **Definition.** An **activation** (or **embedding**) is the list of numbers a
  layer of the network outputs for a token. One patch token's activation is a
  point in 768-dimensional space; nearby points "mean" similar things.
- **Analogy.** GPS coordinates for meaning: just as `(lat, lon)` pins a place on
  Earth, a 768-number activation pins a patch's meaning in "concept space".
- **Tiny number.** A 4-dim toy activation might be `[0.9, -0.1, 0.0, 0.3]`. The
  real ones are 768-dim. **These vectors are the data the SAE learns from.**

### 3.6 A "layer"
- **Definition.** A network is a stack of **layers**; each transforms its input
  a bit and passes it on. ViT-B/16 has 12 transformer *blocks* (layers). We tap
  **one** chosen layer and read its activations.
- **Analogy.** An assembly line with 12 stations. Early stations see edges/colors;
  late stations see whole-object concepts. We pick a station to photograph.
- **Tiny number.** We tap layer `-2` (the second-to-last block) by default — late
  enough to be conceptual, not so late it has collapsed to a single label.

### 3.7 "The dataset the SAE learns from"
- **Definition.** Not images — **activations**. We run many images through the
  frozen backbone, collect the patch-token activations, and pile them up. That
  pile (millions of 768-dim vectors) is the SAE's training set.
- **Analogy.** A geologist studies *rock samples*, not the whole mountain. We
  study *activation samples*, not the raw images.
- **Tiny number.** This milestone's bank: `200 images × 197 tokens = 39,400`
  activation vectors of dim 768. A real run uses millions.

### 3.8 Distribution shift / out-of-distribution (OOD)
- **Definition.** A model is trained on images from some *distribution* (e.g.
  ordinary photos). **Distribution shift** means test images come from a
  *different* source; such images are **out-of-distribution (OOD)**. Performance
  and internal activations both change.
- **Analogy.** You learned to read printed books (in-distribution). Handed
  doctor's handwriting (OOD), you struggle — same task, harder, shifted input.
- **The FAITH-SAE OOD ladder** (clean → progressively harder; see [§4](#4-prerequisites--setup)):
  - **ImageNet-val** — ordinary photos (the clean baseline).
  - **ImageNet-R** — *renditions*: art, cartoons, sculptures, toys of the same
    object classes. Tests *style/abstraction* shift.
  - **ImageNet-Sketch** — black-and-white *sketches*: shape kept, **texture and
    color removed**. Tests whether a concept survives without texture cues.
  - **ImageNet-C** — *corruptions*: 15 distortions (blur, noise, fog, JPEG…) each
    at **severity 1–5**. This is a smooth **dial** of OOD-ness — the headline
    stress curve.
  - **ObjectNet** — real photos with controlled odd **pose, background,
    viewpoint** (a chair on its side, a teapot in the bathtub). The hardest,
    most realistic shift.
- **Tiny number.** A "dog" patch token might read `concept=+2.0` on a clean photo,
  `+1.2` on an ImageNet-R cartoon, `+0.4` on a sketch — the readout *decays* as
  the input shifts. Measuring that decay is RQ3.

### 3.9 Variance
- **Definition.** **Variance** of a set of numbers is the average squared
  distance from their mean — how much they wobble. Low variance = nearly
  constant; high variance = spread out.
- **Analogy.** Two archers: one always hits near the same spot (low variance),
  one sprays the target (high variance).
- **Tiny number.** Values `[2, 2, 2]` → mean 2, variance **0**. Values
  `[0, 2, 4]` → mean 2, variance `((−2)²+0²+2²)/3 = 8/3 ≈ 2.67`.

### 3.10 Sparsity
- **Definition.** **Sparsity** = the fraction of numbers that are (near) **zero**.
  "90% sparse" means 9 of 10 numbers are ~0.
- **Analogy.** A mostly-empty checklist with only a couple of boxes ticked.
- **Tiny number.** `[0.01, 2.3, −0.02, 0.0, −1.7]` with threshold `0.1`: three of
  five are below `0.1` in magnitude → sparsity `= 3/5 = 0.60`. (Sparse
  Autoencoders, next milestone, are *built* to make representations sparse — so
  measuring the data's natural sparsity is a useful baseline.)

### 3.11 PCA (Principal Component Analysis)
- **Definition.** PCA finds the directions in your data along which the points
  are most **spread out** (highest variance). The 1st principal component (PC1)
  is the single most-stretched direction; PC2 is the next, perpendicular to PC1;
  etc. Keeping the **top 2** lets us draw a flat scatter of 768-dim data.
- **Analogy.** A 3-D object casts a 2-D shadow. PCA picks the *camera angle* that
  makes the shadow as informative (spread-out) as possible.
- **Tiny number.** Points hugging the line `y = x` in 2-D have nearly all their
  variance along one diagonal direction — PCA reports PC1 ≈ that diagonal holding,
  say, **98%** of the variance, PC2 only 2%. The data is "really 1-D in disguise."
- **Why we care.** Real activations live on a thin **manifold** (a low-dimensional
  curved sheet) inside the 768-dim box. PCA *reveals* that: in our bank the top
  **32** components hold ~**98%** of the variance, proving the 768 numbers are
  really ~32 numbers in disguise. That low-rank structure is exactly what the
  "on-manifold steering" method ([DESIGN_BRIEF §3](../../DESIGN_BRIEF.md))
  projects onto.

### 3.12 The "manifold" (used throughout FAITH-SAE)
- **Definition.** A **manifold** is a low-dimensional surface that the real data
  lives near, even though it sits inside a high-dimensional space.
- **Analogy.** A sheet of paper (2-D) crumpled inside a room (3-D): every ink dot
  is *on the paper*, never floating freely in the room.
- **Tiny number.** Our 768-dim activations cluster near a ~32-dim sheet:
  "on-manifold" = on the paper; an OOD image pushes the activation **off** the
  paper, which the EDA measures as a rising *off-manifold residual* (`0.14`
  clean → `0.82` shifted).

---

## 4. Prerequisites & setup

### The default offline path needs **nothing extra**
This machine's system Python already has everything:

```bash
/usr/bin/python3 -c "import numpy, matplotlib, sklearn, pandas, yaml, torch; print('OK')"
```

If that prints `OK`, you are ready. **Use `/usr/bin/python3` for every command in
this milestone** (not plain `python`/`python3`).

> The additive `requirements.txt` lists the same libraries for completeness; you
> do **not** need to `pip install` anything for the offline run.

### The REAL RUN path (do NOT run today)
Only when you have a GPU + datasets and want true CLIP activations:

```bash
# REAL RUN (M2): install the real backbone + dataset loaders (NOT for offline).
/usr/bin/python3 -m pip install "open_clip_torch>=2.24" "datasets>=2.19" "pillow>=9.0"
```

**Datasets to download (sizes from `../../ROADMAP.pdf` M2 table):**

| Dataset | Size | How to obtain |
|---|---|---|
| **ImageNet-val** | 50k images (~6.7 GB) | image-net.org (registration) or HF `datasets` `imagenet-1k` (validation split). |
| **ImageNet-R** | 30k images (~2 GB) | `github.com/hendrycks/imagenet-r` → `imagenet-r.tar`. |
| **ImageNet-Sketch** | 50k images (~6 GB) | `github.com/HaohanWang/ImageNet-Sketch` (Google-Drive link in the repo). |
| **ImageNet-C** | 15 corruptions × 5 severities (~70 GB total) | `github.com/hendrycks/robustness` / Zenodo tars `blur.tar`, `noise.tar`, … |
| **ObjectNet** | 50k images (~28 GB) | `objectnet.dev` → `objectnet-1.0.zip` (license click-through). |

You do **not** need any of these to complete this milestone. They are the
shopping list for the real run.

---

## 5. Run it step-by-step

Every command uses `/usr/bin/python3`. From inside this folder:

```bash
cd /Users/abroadhub/Desktop/Research/25_Rajia_Rani_FAITH_SAE/code/milestone_2_data
```

**Step 1 — build the activation bank.** *Why:* downstream code needs real-shaped
data; this manufactures it deterministically (fixed seed) so results reproduce.

```bash
/usr/bin/python3 step1_build_synthetic_bank.py
```

This writes `outputs/activations.npz` (~110 MB) holding the `[200, 197, 768]`
activations, the ground-truth concept directions/labels, and the manifold basis.

**Step 2 — run the EDA.** *Why:* you never train on data you have not inspected;
this prints the summary stats and writes the figures, then builds an OOD-shifted
twin so you *see* what distribution shift does.

```bash
/usr/bin/python3 step2_eda.py
```

This writes `outputs/eda_summary.csv`, `outputs/eda_overview.png`,
`outputs/ood_shift.png`, and prints a PASS/FAIL success block.

**(Optional) re-run with different settings.** *Why:* to feel the knobs. Edit
`config.yaml` — e.g. raise `ood_shift` to `0.5` to make the *default* bank itself
shifted, or change `manifold_rank` — then re-run both steps.

---

## 6. Expected output

After step 1, `outputs/activations.npz` exists (~110 MB). After step 2 you get a
printout like this (numbers reproduce with `seed: 0`):

```
PER-DIMENSION SUMMARY (over all image x token rows):
  mean of per-dim means     : +0.0010  (healthy ~ 0 => activations are centered)
  per-dim variance: min 0.0434  median 0.1231  max 0.4601
SPARSITY (|value| < 0.1): 22.4% of values ~ 0
TOKENS PER IMAGE: min 197  max 197  (expect 197 = 196 patch + 1 CLS)
...
PCA: PC1 holds 23.9% of variance, PC2 11.7% (top-2 = 35.6%).
...
OFF-MANIFOLD RESIDUAL (energy outside the clean manifold):
  clean bank : 0.140
  OOD bank   : 0.824   <- distribution shift pushes mass OFF the manifold
...
SUCCESS CRITERION:
  [PASS] shape matches CLIP ViT-B/16 (197 tokens x 768 dims)
  [PASS] activations centered (|mean| < 0.25)
  [PASS] low-rank manifold (top-32 PCs hold 98.2% of variance, want >80%)
  [PASS] OOD shift increases off-manifold residual (0.140 -> 0.824)
STEP 2 complete.
```

**Artifacts written to `outputs/`:**

| File | What it is |
|---|---|
| `activations.npz` | the activation bank (`acts`, `concept_dirs`, `concept_labels`, `basis`, `meta`). |
| `eda_summary.csv` | every dimension's mean & variance, sorted by variance (the manifold's strong directions on top). |
| `eda_overview.png` | 4 panels — (A) per-dim variance histogram, (B) all activation values, (C) tokens-per-image check, (D) 2-D PCA scatter colored by concept. |
| `ood_shift.png` | clean vs OOD-shifted activation clouds on the *same* PCA axes, with off-manifold residuals. |

**Success criterion:** step 2 exits `0` and all four `[PASS]` lines show. (Exit
`2` with a `[FAIL]` means a knob in `config.yaml` was pushed out of range — see
[§8](#8-common-problems).)

---

## 7. Understand the result

**What *healthy* activations look like (and why these pass):**
- **Centered** — per-dimension means ≈ 0. A dimension that is always-on carries
  no information; real activations hover around zero.
- **Low-rank manifold** — the top **32** principal components capture ~**98%** of
  the variance, even though there are 768 dimensions. The data is "really 32-D in
  disguise." Note this does **not** show up in *raw* per-dimension variance (each
  raw dim is a *mixture* of all manifold directions, so raw variances look flat);
  it shows up in **PCA/component** space. That is why panel (A)'s histogram is a
  smooth hump, but the PCA spectrum is steep.
- **Some natural sparsity** — ~22% of values sit near zero. The SAE's job (next
  milestone) is to *increase* sparsity drastically while staying reconstructive.
- **Consistent token count** — every image yields exactly 197 tokens (panel C is
  a flat bar). If this ever varied, batching would break.
- **Concept structure** — in panel (D) the points where concept #0 is present
  (red) lean differently from where it is absent (gray). They are not perfectly
  separated, and that is *realistic*: a concept is a single diffuse direction, and
  PC1/PC2 capture the *strongest* manifold directions, which need not be the
  concept's direction. (Finding the concept's own direction is precisely the
  SAE's job in M3.)

**What OOD *shift* does to them (the headline intuition):**
- The shifted cloud (`ood_shift.png`, right panel) is **displaced and more
  spread** on the *same* clean PCA axes — the data moved.
- The **off-manifold residual jumps from `0.140` to `0.824`**: most of the
  shifted energy now lives **off** the clean manifold (off the paper). This is
  the single most important fingerprint of distribution shift — and exactly why
  on-manifold steering (which *projects edits back onto the clean manifold*) is
  expected to degrade under OOD. The CFS-vs-shift curve in M4 measures how far
  faithfulness survives this push off the manifold.

---

## 8. Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `command not found: /usr/bin/python3` | wrong interpreter path on your OS | use the path that has the libraries; verify with the §4 import check. |
| `ModuleNotFoundError: No module named 'src'` | launched from the wrong directory | run from inside `milestone_2_data/`; the scripts add the project root to `sys.path` automatically, but the working directory must let them find `../../src`. |
| `ERROR: outputs/activations.npz not found` | ran step 2 before step 1 | run `step1_build_synthetic_bank.py` first. |
| Step 2 exits `2`, a `[FAIL]` line | a `config.yaml` knob is extreme (e.g. `noise_scale` too high drowns the manifold, or `ood_demo_shift = 0`) | restore defaults: `manifold_scale: 5.0`, `noise_scale: 0.05`, `ood_demo_shift: 0.8`. |
| `activations.npz` too big for git | 200×197×768 float32 ≈ 110 MB | it stays in `outputs/` (git-ignored region); commit only `.gitkeep`. Lower `n_images` to shrink it. |
| Figures look empty / window error | a display backend tried to open | the scripts force the headless `Agg` backend; do not change that. |
| `ImportError: open_clip` | you tried the REAL RUN path offline | the real path is gated behind `# REAL RUN (M2):` and intentionally disabled — leave it for when you have a GPU + datasets. |

---

## 9. What's next → `milestone_3_baseline`

You now have the data and you understand it. Milestone 3 **consumes
`outputs/activations.npz`** to:
1. **Train a TopK Sparse Autoencoder** on the 196 patch-token activations — learn
   an overcomplete dictionary of sparse "concept" features.
2. **Recover the planted concept directions** (the `concept_dirs` saved here are
   the ground truth to check against) and build the held-out readout probe.
3. **Estimate the top-`r` real-image PCA subspace** (the `basis` saved here is the
   exact analog) used by the on-manifold projection `P_M`.
4. Wire the four steering variants (`naive_steer`, `random_steer`, `clamp_steer`,
   `onmanifold_steer`) and the `faithfulness(variant, cfg)` dispatcher, then check
   `onmanifold_steer` beats `naive`/`random` on the synthetic CFS.

The bridge is the `.npz`: M2 produces it, M3 trains on it. Then M4 runs the OOD
ladder and draws the CFS-vs-shift curve that answers the paper's question.

---

### The `# REAL RUN (M2)` step left for the learner
The only real-data piece intentionally **not** run here is switching
`step1.build_real_clip_bank()` from its `NotImplementedError` stub to the live
CLIP pipeline: `pip install open_clip_torch`, download a dataset from [§4](#4-prerequisites--setup),
load the frozen `ViT-B-16`, register a forward hook on the chosen layer, run
images through it, and keep the 196 **patch** tokens. The body is written out and
commented inside `step1_build_synthetic_bank.py` directly under the
`# REAL RUN (M2):` marker — uncomment it and fill `real_run.imagenet_val_dir` in
`config.yaml`.

---

*For research and educational purposes only. Author: Rajia Rani
().*
