# Milestone 7 — The Ablations: A1–A5 (turn one knob, find what matters)

**FAITH-SAE** · author: **Rajia Rani** · ``

> Read this top to bottom. It assumes you know **nothing** about ablations,
> sparse autoencoders, sparsity, projection rank, concept selection, or ViT
> layers/tokens — every term is defined from zero, with an analogy and a tiny
> number, before it is used. By the end you will have run **five controlled
> experiments** that each turn exactly ONE knob and **measure** how the Causal
> Faithfulness Score (CFS) responds — isolating, one cause at a time, what makes
> a steered concept faithful.

---

## 1. Where this fits

The whole FAITH-SAE project asks one question:

> When you reach inside a frozen vision model and **steer a concept** ("make this
> image more *dog*"), is the change you cause **causally real**, or just a
> plausible-looking artifact?

Earlier milestones built the answer:

| Milestone | What it teaches / builds |
|---|---|
| `milestone_3_*` (naive baseline) | **naive** off-manifold steering `a' = a + s·d` — the competitor |
| `milestone_4_method` | the **proposed method**: on-manifold steering `a' = a + s·(P_M·Δ)` |
| `milestone_5_evaluation` | the full CFS metric across the out-of-distribution (OOD) ladder |
| `milestone_6_headline_experiment` | the headline CFS-vs-shift result |
| **`milestone_7_ablations` (you are here)** | **why** it works — five ablations that each turn ONE design knob and measure how CFS moves |
| `milestone_8_analysis` | reads these ablations + the headline into the paper's conclusions |

A headline number ("on-manifold steering is more faithful") is not the same as
**understanding**. Maybe it was the SAE type. Maybe the sparsity. Maybe the
projection rank. Maybe which concepts you kept, or which layer you read. An
**ablation** answers each "maybe" by changing exactly that one thing and watching
CFS — so you learn the **cause**, not just the outcome. This milestone runs the
five ablations the design brief calls **A1–A5**.

---

## 2. What you build & run — the five ablations

You will run five offline ablation steps (no downloads, CPU only) plus a plot.
Each step takes a **fixed baseline rig** (a labelled synthetic activation bank +
linear-probe "rulers" + the on-manifold subspace + an empirical CFS measurement),
turns **one knob** across a small sweep, **re-measures CFS for every value**, and
records a row per `(ablation, knob_value, steerer)`.

| Step | Ablation | The ONE knob it turns | What it isolates | Diagnostic logged |
|---|---|---|---|---|
| `step1_a1_sae_type` | **A1** | SAE type — **TopK vs L1** | which sparsity recipe steers more faithfully | reconstruction MSE |
| `step2_a2_topk_k` | **A2** | TopK **`k`** (sparsity level) | the interpretability ↔ faithfulness sweet spot | reconstruction MSE |
| `step3_a3_proj_rank` | **A3** | projection **rank `r`** (core) | the CFS **knee**: over- vs under-constrained | off-manifold residual |
| `step4_a4_select_threshold` | **A4** | concept-selection **threshold** | the reliable-concept fraction (~10–15% tail) | kept fraction |
| `step5_a5_layer_token` | **A5** | backbone **layer** & **patch/CLS** token | where to attach the SAE | reconstruction MSE |
| `step6_plot` | — | — | renders the 5-panel figure | — |

We **reuse** the project's real code via `sys.path`: the SAE comes from
`src.model` (`TopKSAE`), the steerers from the `STEER_REGISTRY` in
`src/blocks/__init__.py`, the score from `src.utils.cfs_score`, and the manifold
diagnostic from `src.utils.onmanifold_projection_residual`. Nothing here fakes a
number — every CFS is **computed from the data** (a Spearman rank correlation, a
sklearn probe, a Cohen's-d effect size).

The two steerers each panel compares (names fixed by the design brief):

| name | edit it applies | role here |
|---|---|---|
| `onmanifold_steer` | `a' = a + s·(P_M·d)` | **ours** — the green series, on top |
| `naive_steer` | `a' = a + s·d` | off-manifold **reference** — the red series below |

---

## 3. Concepts from zero

Read this once slowly. Every step refers back to these.

### 3.1 What an ablation IS (controlled variable)
An **ablation** = turn exactly ONE knob, freeze everything else, and measure what
changes. It is the scientific "controlled experiment". The knob you turn is the
**independent variable**; everything you hold fixed are the **controlled
variables**; the thing you watch (here CFS) is the **dependent variable**.
- *Analogy:* a recipe. To learn what the salt does, cook the dish twice — once with
  salt, once without — keeping the oven, time, and every other ingredient
  identical. The taste difference is the salt's effect, full stop.
- *Tiny number:* baseline CFS = 0.66 with `sae_type=topk`. Switch ONLY `sae_type`
  to `l1` → CFS = 0.55. The 0.11 drop is attributable to the SAE type, nothing
  else.

### 3.2 Confound — why "hold all else fixed" matters
A **confound** is a SECOND thing that changed at the same time and could explain
the result instead. If, while switching topk→l1, you ALSO doubled the bank size, a
CFS change could be the bank, not the SAE type. The fixed baseline removes
confounds: only the named knob differs between two runs, so any CFS change is
caused by that knob.
- *Tiny number:* if two things change and CFS moves by 0.11, you cannot split the
  0.11 between them. Change one and the whole 0.11 is assignable.

### 3.3 Sparse Autoencoder (SAE) — the thing A1/A2 tune
A small network that re-expresses one activation as a SHORT list of **concept
switches** (features), only a FEW of which are ON for any input, then rebuilds the
activation from them. "Sparse" = few switches on at once. The decoder columns are
the concept **directions** we steer.
- *Tiny number:* dictionary size 128, but only 8 switches on at a time → "sparse".

### 3.4 The A1 knob — TopK vs L1 sparsity
Two ways to force "few switches on":
- **TopK SAE:** after the encoder, KEEP only the `k` largest feature values per
  item and zero the rest — a HARD cap of exactly `k` switches on (Gao et al. 2024).
- **L1 SAE:** do NOT cap the count; ADD a penalty `l1_coeff · mean(|features|)` to
  the training loss, gently pushing most feature values toward zero — a SOFT
  sparsity (the classic "vanilla" SAE; Cunningham et al. 2023).
- *Analogy:* TopK is "you may bring exactly 8 items through customs"; L1 is "you
  pay a tax per item, so you naturally bring few".
- *Tiny number:* with `k=8`, at most 8/128 features are ever on. With L1 and
  `l1_coeff=0.001`, the count is whatever the penalty settles on (often a few).

### 3.5 The A2 knob — the value of `k`, and over- vs under-sparsity
`k` = the maximum number of switches allowed ON at once.
- **Under-sparse (`k` TOO BIG):** too many switches → features share the work, each
  smears across several concepts (**polysemantic**), the steering direction is
  muddier → specificity drops → CFS can sag. Reconstruction is GOOD (lots of
  capacity) but faithfulness suffers.
- **Over-sparse (`k` TOO SMALL):** too few switches → the SAE can't represent the
  activation; the target concept may not even get a clean feature → the effect
  weakens. Reconstruction is BAD.
- **Sweet spot:** a middle `k` where each concept gets a clean dedicated feature
  AND reconstruction is decent → CFS peaks.
- *Analogy:* a packing limit. `k=1` forces you to pick the single most important
  item (very selective); `k=32` lets you bring a cluttered bag.
- *Tiny number:* CFS by `k` here reads ≈ 0.59 (k=1), 0.71 (k=4), 0.67 (k=8),
  0.64 (k=16), 0.61 (k=32) — a gentle hump peaking near **k=4**.

### 3.6 The A3 knob — projection rank `r` (the core knob)
On-manifold steering first **projects** the edit onto the thin **sheet** of
directions the model actually uses on real images (milestone 4 teaches this from
zero): `P_M = U_r U_rᵀ` keeps the top-`r` sheet directions. `r` = how many it
keeps = the dimension of the subspace the edit may live in.
- **`r` TOO SMALL (over-constrained):** the projection throws away real sheet
  directions the concept needs → the edit can barely move the concept → the
  **effect dies** → CFS low.
- **`r` ≈ the true sheet rank:** keeps the whole sheet and nothing more → the edit
  moves the concept (monotone, sufficient) WITHOUT smearing off-sheet (specific) →
  CFS **peaks**.
- **`r → dim` (under-constrained):** `P_M → I`, the projection does nothing,
  on-manifold **degenerates into naive** → the edit drifts off-manifold,
  specificity leaks → CFS sags back to the naive level.
- *Analogy:* a stencil with `r` holes. Too few holes and you can't draw the
  concept; the right number and it comes through cleanly; way too many and the
  stencil is gone — you paint anywhere (naive).
- *Tiny number:* true sheet rank is **8** here. Measured on-manifold CFS by `r`:
  0.00 (r=1, dead), 0.22 (r=2), **0.61 (r=4)**, **0.61 (r=8, peak)**, 0.50 (r=16),
  0.50 (r=32), 0.50 (r=64 = naive). The off-manifold residual is **0.00** in the
  sweet spot and rises to **0.39** (= naive) once `r` re-admits the off-sheet leak.

### 3.7 The A4 knob — a concept-selection threshold
An SAE discovers thousands of features; many are **polysemantic** (one feature
mixes several concepts) or junk. The field's finding: only ≈10–15% are
"well-defined" enough to steer reliably. So before steering you **select** the
good ones with a filter.
- **Interpretability score** (per concept, in [0,1]): how monosemantic a concept
  is. Here we compute **distinctness** = `1 − max |cosine overlap| with any other
  concept`. A clean concept points its own way (overlap ≈ 0 → distinctness ≈ 1); a
  polysemantic one shares its direction with a neighbour (overlap high →
  distinctness ≈ 0).
- **Threshold τ** (the knob): KEEP only concepts with score ≥ τ; DROP the rest.
  - *Analogy:* a bouncer's height line. Raise it (bigger τ) → fewer get in, but
    everyone inside is taller (more reliable).
  - As τ rises: the **kept fraction falls** (toward the reliable tail) while the
    **mean CFS of the survivors rises** (the keepers are cleaner).
- *Tiny number:* measured distinctness for the 4 planted concepts:
  `[0.98, 0.71, 0.20, 0.20]`. At τ=0.0 keep 100%, mean CFS ≈ 0.56; at τ=0.8 keep
  25%, mean CFS ≈ 0.64. Stricter filter → fewer but more faithful concepts.

### 3.8 The A5 knob — which layer, and patch vs CLS token
A Vision Transformer cuts an image into a grid of **patches**, turns each into a
token, and keeps one extra **CLS** token (a whole-image summary). Activations
change as the image flows UP the layer stack: **early** layers carry low-level
texture; **late** layers carry abstract, concept-level meaning.
- **Early vs late layer:** early = a THICKER, blurrier concept "sheet" (less clean
  directions); late = a CLEANER, lower-dimensional sheet (crisp concept
  directions) → steering tends to be more faithful late.
- **Patch vs CLS token:** patch = one activation per region (fine-grained, more
  signal); CLS = a single pooled vector (coarse, noisier readout).
- *Analogy:* reading a smudged draft (early) vs a clean final copy (late);
  photographing each room (patch) vs one street photo of the whole house (CLS).
- *Tiny number:* measured on-manifold CFS — `late|patch` ≈ 0.66 (best),
  `early|cls` / `late|cls` ≈ 0.58–0.61 (worst). Late + patch wins.

### 3.9 How to read a "CFS vs knob" curve — the sweet spot and the knee
Plot CFS (y) against the knob (x).
- A **sweet spot** is the PEAK of a hump — the knob value with the best
  faithfulness.
- A **knee** is the ELBOW where the curve bends sharply (extra knob stops buying
  CFS, or even starts hurting). For A3 the sweet spot and knee coincide near the
  true sheet rank: below it the effect dies, above it faithfulness degrades back to
  naive. You pick the smallest knob value that buys you the most CFS.

### 3.10 The three CFS ingredients (recap; measured, not looked up)
CFS = the **harmonic mean** of three numbers in [0,1] (conjunctive — all three
must hold):
- **Monotonicity** = does turning the knob up move the target readout up, in order?
  (Spearman rank correlation between knob and readout.)
- **Specificity** = does ONLY the target move, while OFF-target probes stay put?
  (1 − normalized off-target drift.)
- **Sufficiency** = is the effect BIG enough? (Cohen's-d effect size at full knob.)
A near-zero in any one drags CFS to near zero. Every ablation re-measures all three
with the SAME rig, so the only thing that changes is its own knob.

---

## 4. Prereqs & setup

Everything runs **offline on CPU**. Use **`/usr/bin/python3`** for every command
(plain `python`/`python3` may point elsewhere).

```bash
# from this folder: code/milestone_7_ablations/
cd /Users/abroadhub/Desktop/Research/25_Rajia_Rani_FAITH_SAE/code/milestone_7_ablations

# Check your interpreter has everything (all preinstalled on the provided box):
/usr/bin/python3 - <<'PY'
for m in ("torch","numpy","sklearn","scipy","matplotlib","yaml"):
    mod = __import__(m); print("OK", m, getattr(mod, "__version__", "?"))
PY
```

If any line says it is missing (only on a fresh machine), install the project base
then this milestone's additive line:

```bash
/usr/bin/python3 -m pip install -r ../../requirements.txt   # torch, pyyaml, numpy
/usr/bin/python3 -m pip install -r requirements.txt         # scikit-learn, scipy, matplotlib
```

There is **nothing to download** and **no GPU**: the "real images" are a synthetic
labelled activation bank generated locally inside the shared rig (`_common.py`).

---

## 5. Run it step-by-step

The fastest path is the whole study in one command (clears the CSV, runs A1→A5,
draws the figure):

```bash
MPLBACKEND=Agg /usr/bin/python3 run_all.py
```

To learn what each ablation does, run them one at a time. Each **appends** its
rows to `outputs/ablations.csv` and prints a teaching table; run `run_all.py` (or
delete the CSV) to start clean:

1. **`/usr/bin/python3 step1_a1_sae_type.py`** — A1: TopK vs L1 SAE; CFS + recon MSE
   per type.
2. **`/usr/bin/python3 step2_a2_topk_k.py`** — A2: sweep `k`; find the sparsity
   sweet spot.
3. **`/usr/bin/python3 step3_a3_proj_rank.py`** — A3 (core): sweep `r`; find the
   CFS knee.
4. **`/usr/bin/python3 step4_a4_select_threshold.py`** — A4: sweep τ; the
   reliable-concept fraction.
5. **`/usr/bin/python3 step5_a5_layer_token.py`** — A5: early/late × patch/CLS; CFS
   per attachment.
6. **`/usr/bin/python3 step6_plot.py`** — render the 5-panel
   `outputs/ablations.png`.

(`MPLBACKEND=Agg` makes matplotlib draw to a file without opening a window, so the
plot never blocks.)

---

## 6. Expected output

After `run_all.py` you get, in `outputs/`:

- **`ablations.csv`** — one row per `(ablation, knob_value, steerer)`. Columns:
  `ablation_id, knob_value, variant, cfs, diagnostic, diagnostic_name,
  monotonicity, specificity, sufficiency, offmanifold_residual`. 48 data rows
  (A1:4, A2:12, A3:14, A4:10, A5:8). Approximate on-manifold values (synthetic,
  illustrative — your numbers reproduce exactly with the fixed seed):

  | ablation | knob → CFS (on-manifold) | the diagnostic |
  |---|---|---|
  | **A1** | topk → ~0.66, l1 → ~0.55 | recon MSE: topk ~0.05, l1 ~0.02 |
  | **A2** | k=1 ~0.59 · k=4 ~0.71 (peak) · k=32 ~0.61 | recon MSE falls as k rises |
  | **A3** | r=1 ~0.00 · r=8 ~0.61 (peak) · r=64 ~0.50 (=naive) | off-resid 0.00 at peak → 0.39 at r→dim |
  | **A4** | τ=0.0 ~0.56 · τ=0.8 ~0.64 (rising) | kept fraction 1.00 → 0.25 (falling) |
  | **A5** | late\|patch ~0.66 (best) · late\|cls ~0.57 (worst) | recon MSE lower on the clean late layer |

- **`ablations.png`** — a 5-panel figure (one panel per ablation A1–A5) plus a text
  panel. In every panel the **green** on-manifold series sits **above** the **red**
  naive reference. A2 marks the sparsity sweet spot; A3 marks the knee and the true
  sheet rank; A4 shows the rising-CFS / falling-fraction trade-off on twin axes.

**Success criterion** (the run prints `MILESTONE 7 COMPLETE`):

> All **five** ablations ran, `ablations.csv` has a **row per
> `(ablation, knob_value, steerer)`**, and **every CFS is in [0,1]**.

---

## 7. Understand the result — what each ablation teaches

- **A1 (SAE type).** With everything else fixed, **TopK** steers more faithfully
  than **L1** (CFS ~0.66 vs ~0.55). TopK's hard cap gives crisper, higher-magnitude
  concept directions; L1's soft penalty blurs feature magnitudes. (L1 reconstructs
  the activation a touch better — lower MSE — but reconstruction is not
  faithfulness; A1 shows you must measure CFS, not MSE.)
- **A2 (sparsity `k`).** CFS **humps**: too small `k` starves the effect (no clean
  feature for the concept), too large `k` makes features polysemantic (specificity
  drops). The peak (~`k=4` here) is the sparsity **sweet spot** — the smallest `k`
  that still gives each concept its own clean switch. Reconstruction MSE keeps
  falling with `k`, which is exactly why you cannot pick `k` by reconstruction.
- **A3 (projection rank `r`) — the core knob.** This is the clearest cause. At
  **low `r`** the projection over-constrains the edit and the **effect dies**
  (CFS → 0). Near the **true sheet rank** (8) CFS **peaks** and the off-manifold
  residual is **0** — the edit is realistic and specific. As **`r → dim`** the
  projection vanishes (`P_M → I`), on-manifold **becomes naive**, the off-sheet
  leak returns, specificity falls, and CFS **degrades back to the naive level**
  (residual → 0.39 = naive). The **knee at the sheet rank** is the whole point of
  on-manifold steering: keep the sheet, drop the rest.
- **A4 (selection threshold).** Faithfulness is **not** uniform across concepts —
  the distinct (monosemantic) ones steer well, the polysemantic ones don't. Raising
  the threshold **drops the unreliable concepts**: the kept fraction falls toward
  the field's ~10–15% reliable tail while the mean CFS of the survivors rises. A4
  tells you to **select before you steer**.
- **A5 (layer & token).** Where you attach the SAE matters: the **clean late layer
  with per-patch tokens** is the most faithful place; the pooled **CLS** token and
  the **blurry early** layer are worse (and reconstruct worse). A5 tells you to read
  late-layer patch activations.

Put together: faithfulness comes from a **TopK** SAE at a **moderate `k`**, steered
**on-manifold at `r` ≈ the sheet rank**, on **selected** (monosemantic) concepts,
read from a **late-layer patch** representation. Each ablation isolates one of
those five causes.

---

## 8. Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: src` | ran from the wrong folder, or `sys.path` not set | run from `code/milestone_7_ablations/`; `_common.py` adds the project root automatically |
| `No module named scipy` / `sklearn` / `matplotlib` | fresh machine | `/usr/bin/python3 -m pip install -r requirements.txt` |
| a plot window blocks / `Agg` warning | matplotlib tried to open a display | prefix commands with `MPLBACKEND=Agg` (the steps also force `Agg` internally) |
| `ablations.csv` has duplicated rows | ran a step twice without clearing | run `run_all.py` (it clears the CSV first) or delete `outputs/ablations.csv` |
| A3 curve looks flat | `true_manifold_rank` ≈ `dim`, so every `r` keeps the sheet | keep `true_manifold_rank` (8) well below `dim` (64) so low `r` truncates |
| A4 keeps 100% at every τ | concepts are all equally clean | the planted bank gives concepts varying distinctness; keep `n_concepts ≥ 4` |
| a CFS prints as `0.0` | one ingredient hit ~0 (e.g. r=1 starves the effect) | expected — the harmonic mean is conjunctive; that IS the over-constrained failure |

---

## 9. What's next → `milestone_8_analysis`

You have now **isolated the causes**: which SAE type (A1), which sparsity (A2),
which projection rank (A3), which concepts (A4), and which layer/token (A5) make a
steer faithful. Milestone 8 **reads these ablations together with the headline
OOD result** (milestone 6) into the paper's analysis: the recommended operating
point (TopK, moderate `k`, `r` ≈ sheet rank, selected concepts, late-patch),
the per-concept reliability distribution, and the limits where faithfulness still
breaks. The shared measuring rig in `_common.py` (bank → probes → U_r → CFS) is the
same rig milestone 8 uses to assemble the final tables.

---

### Real-run note (`# REAL RUN (M7)`)
The offline default regenerates a **synthetic** labelled activation bank + a toy
TopK SAE and sweeps each knob on it. For the real study, each step's
`# REAL RUN (M7):` comment block spells out the swap: read a **large real CLIP
ViT-B/16 activation bank** over ImageNet-val (and the OOD shifts), fit the real
SAE(s), estimate `U_r` once from the real bank, replace A4's distinctness proxy
with the SAE auto-interp / activation-coherence score, and read A5's early/late ×
patch/CLS choices from the real backbone. The measuring rig (bank → probes → `U_r`
→ measured CFS) is unchanged; only the activations become real. No knob value or
column changes — the ablation logic ports directly.

---

*For research and educational purposes only.*
