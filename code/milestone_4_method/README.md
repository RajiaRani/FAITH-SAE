# Milestone 4 — The Method: On-Manifold Steering

**FAITH-SAE** · author: **Rajia Rani** · ``

> Read this top to bottom. It assumes you know **nothing** about steering,
> manifolds, PCA, or projections — every term is defined from zero with a tiny
> number before it is used. By the end you will have run the project's
> **proposed method** (on-manifold steering) and seen, with your own eyes, that
> it is *faithful* where the naive baseline is a *mirage*.

---

## 1. Where this fits

The whole FAITH-SAE project asks one question:

> When you reach inside a frozen vision model and **steer a concept** ("make this
> image more *dog*"), is the change you cause **causally real**, or just a
> plausible-looking artifact?

The project's answer has two halves:

- **A method** that makes the steer real — *on-manifold steering* (this milestone).
- **A measuring stick** that proves it — the *Causal Faithfulness Score*, CFS
  (milestone 5).

Milestone-by-milestone, the `code/` path is:

| Milestone | What it teaches / builds |
|---|---|
| `milestone_1_foundations` | the synthetic SAE pipeline, the four steerer names, what CFS is |
| `milestone_2_data` | the activation data (synthetic bank now; real CLIP later) |
| `milestone_3_*` (naive baseline) | **naive** off-manifold steering `a' = a + s·d` — the competitor |
| **`milestone_4_method` (you are here)** | the **proposed method**: on-manifold steering `a' = a + s·(P_M·Δ)`, compared side-by-side with the naive baseline |
| `milestone_5_evaluation` | the full CFS metric across the out-of-distribution (OOD) ladder |

This milestone is where the project's **headline idea** appears: instead of
adding the raw edit Δ (which flies *off* the thin sheet of activations the model
actually understands), we first **project Δ onto that sheet** with a matrix
`P_M`, so the steered activation stays realistic. Then we measure — with the same
strength for every method — that this projection is exactly what makes the steer
faithful.

> The naive baseline this milestone compares against is **milestone 3**'s
> `naive_steer`. This folder re-derives it from zero too, so you can run
> milestone 4 standalone without having run milestone 3 first.

---

## 2. What you build & run

You will run a 5-step offline pipeline (no downloads, CPU only) that:

1. **Regenerates a synthetic "real-image" activation bank** that deliberately
   lives on a thin **8-dimensional sheet** inside a 64-dimensional space
   (`step1`).
2. **Estimates that sheet with PCA**, keeping the top-`r` directions as the
   columns of a matrix **U_r**, and builds the projection **P_M = U_r U_rᵀ**
   (`step2`).
3. **Runs all four steering methods** on one raw edit Δ, and measures each
   method's **off-manifold residual** — how much of its edit leaves the sheet
   (`step3`).
4. **Scores the Causal Faithfulness Score (CFS)** per method and writes the
   headline table `outputs/method_compare.csv` (`step4`).
5. **Draws** `outputs/method_compare.png` — a bar chart of CFS plus a scatter of
   residual-vs-CFS showing on-manifold in the "faithful AND on-manifold" corner
   (`step5`).

The four methods (names fixed by the project's design brief, registered in
`src/blocks/__init__.py`):

| name | edit it applies | role |
|---|---|---|
| `naive_steer` | `a' = a + s·d` (the whole raw edit) | **milestone-3 baseline / main competitor** |
| `random_steer` | `a' = a + s·(random dir)` | null / sanity baseline |
| `clamp_steer` | clamp the SAE feature to magnitude `s`, no projection | off-manifold variant |
| `onmanifold_steer` | `a' = a + s·(P_M·d)` (edit projected onto the sheet) | **ours (proposed method)** |

We **reuse** the project's real code via `sys.path`: the steerers come from the
`STEER_REGISTRY`, the residual from `src.utils.onmanifold_projection_residual`,
and the score from `src.utils.cfs_score` (through the shared `faithfulness`
dispatcher). Nothing here re-implements the method — this milestone *drives* it.

---

## 3. Concepts from zero

Read this once slowly. Every later step refers back to these.

### 3.1 Activation
A list of numbers a neural network produces inside itself while looking at an
input — the model's private "notes" about one image-patch. Real CLIP ViT-B/16
notes are **768** numbers long; we use **64** so it runs instantly on a laptop.
One activation = one **point** in a 64-dimensional space (a list like
`[0.3, -1.2, ...]`, 64 entries).

### 3.2 The big space, and the data manifold (the "sheet")
- **The big space** = all possible 64-number lists. Picture a gym, but with 64
  directions instead of 3. Every activation is a dot somewhere in that gym.
- **The data manifold ("the sheet")** = the thin, curved region the model
  *actually* lands in when you feed it **real** images. It is like a **sheet of
  paper floating in the gym**: the paper is (say) 8-dimensional even though the
  gym is 64-dimensional. The model was effectively trained *on the paper*, so it
  only behaves sensibly for activations **on** the paper. Points **off** the
  paper (mid-air) are activations the model has never really seen — it does not
  handle them reliably.
  - *Analogy:* handwritten digits. The set of all 28×28 pixel grids is enormous,
    but real handwriting occupies a tiny curved sheet inside it; random pixel
    static (off the sheet) is not a digit.
  - *Tiny number:* if the sheet is 8-D inside a 64-D gym, then **56 of the 64**
    directions are "off the sheet" — real activations barely spread along them.

### 3.3 A subspace
A flat slice through the big space that passes through the origin: a line, a
plane, or a higher-dimensional "flat". To first approximation our sheet **is** a
subspace — the span of a few directions.
- *Analogy:* in a 3-D room, the floor is a 2-D subspace; a single light beam is a
  1-D subspace.
- *Tiny number:* the span of 16 chosen directions in 64-D is a **16-dimensional**
  subspace; a point in it needs only 16 coordinates, not 64.

### 3.4 PCA — "find the main directions"
**Principal Component Analysis** takes a cloud of points and finds the direction
it spreads out **most** (PC1), then the next-most-spread direction perpendicular
to it (PC2), and so on. Each PC has a **variance** = how much the cloud spreads
along it. The first few PCs trace the sheet; the rest have near-zero variance.
- *Analogy:* a thin frisbee floating in a room. PC1 and PC2 lie in its flat face
  (lots of spread); PC3 is its thin axis (almost none). Keep PC1+PC2 → you keep
  the frisbee; drop PC3 → you drop only the thinness.
- *Tiny number:* if PC1 explains 60% of the spread, PC2 30%, PC3 8%, rest 2%,
  then the top 3 PCs capture **98%** — the sheet is ~3-D.

### 3.5 U_r — the estimated sheet basis
Stack the top-`r` principal components as the **columns** of a `[64, r]` matrix
**U_r**. Its columns are **orthonormal** (mutually perpendicular, each length 1).
U_r **is** our estimate of the sheet's directions. In real life this is all you
get — the *true* sheet is unknown.

### 3.6 Projection, and the matrix P_M = U_r U_rᵀ
**Projecting** a vector `v` onto a subspace = finding the **closest** point to `v`
that lies *in* the subspace (drop a perpendicular onto it).
- coordinates of `v` inside the subspace: `c = U_rᵀ v`  (just `r` numbers)
- rebuild that point in the big space: `v_proj = U_r c = U_r (U_rᵀ v)`
- so one matrix does both at once: **`P_M = U_r U_rᵀ`**, and `v_proj = P_M v`.
- Properties you can rely on: `P_M` is symmetric, `P_M·P_M = P_M` (projecting
  twice changes nothing), and `trace(P_M) = r`.

### 3.7 The on/off-manifold idea — a tiny 2-D worked example
Shrink the gym to **2-D** (an x-axis and a y-axis). Let the "sheet" be the
**x-axis line** (a 1-D sheet inside a 2-D space). Projecting onto it keeps the
x-part and zeroes the y-part: `project((a,b)) = (a, 0)`.

| | vector | result | on the sheet? |
|---|---|---|---|
| point | `P = (2.0, 0.0)` | — | yes (on the x-axis) |
| edit **along** the sheet | `Δ_on = (1.0, 0.0)` | `P+Δ = (3.0, 0.0)` | **yes** — realistic |
| edit **off** the sheet | `Δ_off = (0.0, 1.0)` | `P+Δ = (2.0, 1.0)` | **no** — floats into air |

**Off-manifold residual** `= ‖Δ − P_M·Δ‖ / ‖Δ‖`:
- `Δ_on`: `‖(1,0)−(1,0)‖ / ‖(1,0)‖ = 0/1 = 0.0` → fully on the sheet.
- `Δ_off`: `‖(0,1)−(0,0)‖ / ‖(0,1)‖ = 1/1 = 1.0` → fully off the sheet.

That single `0.0`-vs-`1.0` number is **exactly** what this milestone measures,
just in 64-D. (`step1` prints this example so you can check it by hand.)

### 3.8 The raw edit Δ, and the two ways to apply it
To steer a concept up, we add a direction Δ to the activation. Here Δ is the SAE
**concept direction** `d` (the decoder column for the feature we picked).
- **naive** adds the *whole* Δ: `a' = a + s·d`. If Δ has a big off-sheet part, the
  steered activation flies off the manifold.
- **on-manifold** adds only the on-sheet part: `a' = a + s·(P_M·d)`. The edit
  stays realistic. *(Tiny 2-D number: raw Δ = (0.6, 0.8). naive applies all of it
  → residual 0.8; on-manifold applies (0.6, 0) → residual 0.0.)*

### 3.9 The rank `r` knob
`r = manifold_rank` = how many PCs we keep = the dimension of the sheet we trust.
- `r` **too small** → we throw away real sheet directions; the edit can't move
  the concept (the effect dies; over-constrained).
- `r` **about right** → we keep the whole sheet, nothing more; edits stay
  realistic and still move the concept.
- `r → 64` → we keep everything; `P_M` becomes the identity `I`; projecting does
  nothing; **on-manifold degenerates into naive** (design brief §14: naive is the
  `r → ∞`, `P_M = I` case).

### 3.10 Off-manifold residual = the diagnostic
The off-manifold residual is the fraction of an edit that **leaves the sheet**. We
compute it with the project's own helper `onmanifold_projection_residual(edit,
U_r)`. naive/clamp/random edits have a **large** residual; on-manifold's is
**≈ 0** because `P_M·(P_M·d) = P_M·d` (projecting an already-projected edit leaves
nothing off-sheet).

### 3.11 Why staying on-manifold makes the steer *causally real* (not a mirage)
A model only "understands" activations on its sheet. Push an activation off the
sheet and any downstream readout that moves is reacting to a state the model was
never trained on — a **mirage**: the number changed, but not for a reason the
model genuinely represents, and the change does not survive scrutiny (it smears
into unrelated concepts, isn't decodable, breaks under distribution shift). Keep
the edit on the sheet and the same readout move reflects a **real** internal
concept — it is *monotone*, *specific*, and *sufficient*. That trio is the CFS.

---

## 4. Prereqs & setup

Everything runs **offline on CPU**. Use **`/usr/bin/python3`** for every command
(plain `python`/`python3` may point elsewhere).

```bash
# from this folder: code/milestone_4_method/
cd /Users/abroadhub/Desktop/Research/25_Rajia_Rani_FAITH_SAE/code/milestone_4_method

# 1. Check your interpreter has everything (all preinstalled on the provided box):
/usr/bin/python3 - <<'PY'
for m in ("torch","numpy","sklearn","matplotlib","yaml"):
    mod = __import__(m); print("OK", m, getattr(mod, "__version__", "?"))
PY
```

If any line says it is missing (only on a fresh machine), install the project
base then this milestone's additive line:

```bash
/usr/bin/python3 -m pip install -r ../../requirements.txt   # torch, pyyaml, numpy
/usr/bin/python3 -m pip install -r requirements.txt         # scikit-learn, matplotlib
```

There is **nothing to download**: the "real images" are a synthetic activation
bank generated locally in `step1`.

---

## 5. Run it step-by-step

The fastest path is the whole pipeline in one command:

```bash
/usr/bin/python3 run_all.py
```

To learn what each stage does, run them one at a time (each reads the previous
step's file from `outputs/` and writes its own — they are independent on disk):

1. **`/usr/bin/python3 step1_build_bank.py`**
   *Why:* manufacture the "real-image" activation bank that lives on a known 8-D
   sheet, and print the 2-D on/off-manifold example. Writes `outputs/real_bank.npy`.

2. **`/usr/bin/python3 step2_estimate_subspace.py`**
   *Why:* discover the sheet with **PCA**, keep the top-`r` directions as **U_r**,
   build **P_M = U_r U_rᵀ**, and verify it (variance explained, `trace(P_M)=r`,
   100% recovery of the planted sheet). Writes `outputs/U_r.npy`.

3. **`/usr/bin/python3 step3_steer_and_residual.py`**
   *Why:* run all four steerers on one raw edit Δ using the **fixed** real-image
   basis U_r, and measure each method's **off-manifold residual**. Writes
   `outputs/residuals.csv`. (naive ≈ 0.93, on-manifold ≈ 0.00.)

4. **`/usr/bin/python3 step4_score_cfs.py`**
   *Why:* compute the **CFS** per method (reusing the project's `cfs_score` via the
   shared `faithfulness` dispatcher) and join it with the measured residuals.
   Writes the headline `outputs/method_compare.csv`.

5. **`/usr/bin/python3 step5_plot.py`**
   *Why:* render `outputs/method_compare.png` — a CFS bar chart plus a
   residual-vs-CFS scatter with on-manifold in the top-left "faithful AND
   on-manifold" corner.

---

## 6. Expected output

After `run_all.py` you get, in `outputs/`:

- **`method_compare.csv`** — one row per method:
  `variant, monotonicity, specificity, sufficiency, offmanifold_residual, cfs`.
  Approximate values (synthetic, illustrative):

  | variant | off-manifold residual | CFS |
  |---|---|---|
  | `naive_steer` | ~0.93 | ~0.61 |
  | `random_steer` | ~0.89 | ~0.15 |
  | `clamp_steer` | ~0.93 | ~0.55 |
  | **`onmanifold_steer`** | **~0.00** | **~0.90** |

- **`method_compare.png`** — left: CFS bars (on-manifold tallest of the four);
  right: residual-vs-CFS scatter (on-manifold top-left, naive/clamp lower-right,
  random bottom), with an arrow showing the projection dragging naive up into the
  faithful corner.

**Success criterion** (the run prints `PASS`/`FAIL` for both):

> `onmanifold_steer` CFS **>** `naive_steer` CFS, **and** `onmanifold_steer`
> off-manifold residual **≈ 0** (and `<` `naive_steer`'s).

---

## 7. Understand the result — why projection fixes faithfulness

The naive edit Δ is a near-random 64-D direction, so **most of its length points
off the 8-D sheet** — its off-manifold residual is ~0.93. Adding it pushes the
activation into "mid-air", a state the frozen model never learned. Any readout
that moves there is a **mirage**: it smears into off-target concepts
(low *specificity*) and would not survive distribution shift. That is why naive's
CFS is mediocre (~0.61) even though its apparent "effect" looks large.

On-manifold steering first hits Δ with `P_M`, **deleting the off-sheet part**
(residual → 0.00) and keeping only the directions the model actually uses on real
images. The edit is now realistic: it moves the target concept (monotone,
sufficient) **without** smearing into others (specific) — so all three CFS
ingredients are high at once, and CFS jumps to ~0.90. The single change —
`a' = a + s·(P_M·Δ)` instead of `a' = a + s·Δ` — is the entire contribution, and
the off-manifold residual is the diagnostic that proves it.

`random_steer` is the floor: a random direction has no real concept, so its
monotonicity collapses and the harmonic mean drags CFS to ~0.15 — a sanity check
that the metric is not just rewarding "big change".

---

## 8. Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: src` | ran from the wrong folder, or `sys.path` not set | run from `code/milestone_4_method/`; `_common.py` adds the project root automatically |
| `FileNotFoundError: outputs/U_r.npy` (in step3) | ran step3 before step2 | run `step1` → `step2` → `step3` in order, or just `run_all.py` |
| `command not found: /usr/bin/python3` | non-macOS / different layout | use the interpreter that has torch+sklearn; the contract assumes `/usr/bin/python3` |
| `No module named sklearn` / `matplotlib` | fresh machine | `/usr/bin/python3 -m pip install -r requirements.txt` |
| on-manifold residual **not** ~0 | `manifold_rank` set to 0, or U_r is empty | keep `manifold_rank ≥ true_manifold_rank` (default 16 ≥ 8) |
| on-manifold CFS ≈ naive CFS | `manifold_rank` raised toward `dim` (64) | that is the **expected** degeneration (`P_M → I` = naive); lower `r` back to 16 |
| effect "dies", on-manifold CFS drops | `manifold_rank` set far **below** `true_manifold_rank` | raise `r`; too few directions over-constrains the edit (ablation A3 knee) |

---

## 9. What's next → `milestone_5_evaluation`

You proved on-manifold steering is faithful **on clean, in-distribution data**.
Milestone 5 turns the dial: it measures the **full CFS** (monotonicity ×
specificity × sufficiency, harmonic mean) **empirically** by sweeping the knob,
and then re-measures it as inputs get harder along the **OOD ladder**
(clean → ImageNet-R → ImageNet-Sketch → ImageNet-C severity 1–5 → ObjectNet) to
find the **collapse knee** — where faithfulness finally breaks, and whether
on-manifold degrades more gracefully than naive. The U_r you estimated here is
the fixed real-image subspace that milestone 5 reuses at every shift level.

---

### Real-run note (`# REAL RUN (M4)`)
The offline default regenerates a **synthetic** low-rank bank and PCAs it. For the
real study, estimate **U_r once** from a **large real CLIP ViT-B/16 activation
bank** over ImageNet-val (hundreds of thousands of patch activations), cache
`U_r.npy`, and reuse it for every steerer, concept, and OOD shift level. The
planted-sheet self-grade is dropped (no ground-truth sheet exists in real life);
the empirical CFS probe (`src.evaluate.cfs_probe`) replaces the analytic
dispatcher. Each step's `# REAL RUN (M4):` comment block spells out the swap.

---

*For research and educational purposes only.*
