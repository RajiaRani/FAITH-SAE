# DESIGN BRIEF — FAITH-SAE

> Prospectus **#25** (Cluster: Vision Interpretability — Sparse Autoencoders, Causal
> Faithfulness, Distribution Shift). **NOT an inference-engineering project**: the
> factory's "Mechanism A + Mechanism B + combination" framing is adapted to an
> *interpretability* study (a steering **method** A combined with a faithfulness
> **metric + OOD stress** B), and the "efficiency axis" slot is filled by an
> **implementation-independent quality score (CFS)** rather than bytes/token.
> Exemplar to imitate in shape/depth: `../22_Chenchaiah_Mekalathuru_Sliding_Window_MLA/`.
> **Venue: IEEEtran conference** (self-contained `IEEEtran.cls` staged in `paper/`; no
> external author kit). Title in sentence/title case.
> This brief is the **contract**: the roadmap, paper, figures, and code must all agree
> with it (same title, RQs, baselines, benchmarks, variant names, figure filenames).

## 1. Title & slug
- **Title:** *FAITH-SAE: Are Sparse-Autoencoder Concept Directions in Vision Models
  Causally Faithful Under Distribution Shift?*
- **Folder slug:** `FAITH_SAE` (folder `25_Rajia_Rani_FAITH_SAE`)

## 2. One-line claim
Steering a Sparse-Autoencoder concept **on-manifold** (projecting the edit onto the
directions a frozen vision model actually uses on real images) is **causally faithful**
— monotone, specific, and sufficient — where naive off-manifold steering is not, and we
quantify exactly **how far that faithfulness survives as inputs shift out of
distribution** with a single Causal Faithfulness Score (CFS).

## 3. Framing (the A + B + combination)
- **Mechanism A — On-manifold steering (the method):** force a chosen SAE concept up/down
  inside the frozen backbone, but **constrain the edit to the on-manifold subspace** —
  project the raw activation-addition `Δ` onto the top-`r` principal directions of real-
  image activations (`P_M·Δ`), so the steered activation stays in the region the model
  was actually trained on. Strength: edits are realistic, decodable, non-destructive.
  **Weakness alone:** a steering method with *no metric* is unfalsifiable — you cannot
  tell a real causal effect from a plausible-looking artifact.
- **Mechanism B — CFS faithfulness metric + OOD stress (the measuring stick):** a single
  composite score in [0,1] that asks whether an edit is **causally real** (Monotonicity:
  knob up → readout up smoothly), **specific** (only the target concept moves; off-target
  probes stay flat), and **sufficient** (effect size matches the concept's claimed
  meaning) — then re-measures it as test images get harder (clean → renditions → sketch →
  corruption → real-world shift). Strength: makes "faithful" falsifiable and comparable.
  **Weakness alone:** a metric applied to *naive* steering just measures off-manifold
  artifacts — high apparent effect, low real faithfulness.
- **Combination (the unoccupied space):** measure **CFS of on-manifold steering across the
  OOD sweep**. Each mechanism cures the other's blind spot: on-manifold steering gives the
  metric something *real* to measure; the metric proves the on-manifold edit is faithful
  and tracks where it breaks. The CFS-vs-shift curve is the paper's answer.
- **Why the gap is open:** prior work shows SAEs *find* interpretable vision concepts and
  that you *can* steer them, and separately warns that off-manifold steering is
  unreliable — but **no one has a quantitative faithfulness benchmark for vision SAE
  features, and no one has tested faithfulness under distribution shift**. On-manifold
  faithfulness **and** OOD robustness, combined under one frozen backbone, is new.

## 4. Research questions
- **RQ1 (headline).** Does on-manifold projected steering achieve a higher Causal
  Faithfulness Score (CFS) than naive off-manifold activation-addition steering — and than
  random-direction, raw-clamp, and supervised concept-direction (TCAV-style) steering — at
  **matched steering strength** on clean, in-distribution images?
- **RQ2 (core knob).** How does CFS **decompose** into monotonicity / specificity /
  sufficiency, what fraction of discovered concepts steer reliably (the field's "only
  ~10–15%" claim), and how does the faithfulness optimum depend on the two design knobs —
  **steering strength** and **manifold-projection rank `r`**?
- **RQ3 (the sweep / Pareto-knee analog).** Does CFS **survive distribution shift** across
  clean ImageNet → ImageNet-R → ImageNet-Sketch → ImageNet-C (corruption severity dial) →
  ObjectNet, and **where is the collapse knee** — i.e. what is the ΔCFS-per-shift-level
  degradation slope, and does on-manifold steering degrade more gracefully than naive?

## 5. Baselines (all compared at matched steering strength)
- **Supervised concept-direction (TCAV-style) steering** — *quality reference*: the strong,
  label-expensive direction (linear-probe / concept-activation vector), the gold a good
  unsupervised SAE direction should approach.
- **Naive off-manifold activation-addition steering** — the standard `ActAdd`-style edit
  the field admits is unreliable (the main competitor).
- **Random-direction steering** — null/sanity baseline (high "effect", no real concept).
- **Raw clamp steering** — clamp the SAE feature to a fixed magnitude with no projection.
- **On-manifold projected steering (ours)** — `P_M·Δ` constrained to the top-`r` real-image
  subspace.

## 6. Benchmarks
| Benchmark | Family | Stresses |
|---|---|---|
| ImageNet-val | clean in-distribution | quality reference: clean CFS, concept readouts |
| ImageNet-R | rendition / abstraction shift | **does faithfulness survive style/domain shift** |
| ImageNet-Sketch | texture/style removal | **concept readout without texture cues (stress test)** |
| **ImageNet-C** | corruption-severity dial | **graded OOD: the CFS-vs-severity curve (headline stress test)** |
| ObjectNet | real-world pose/background/viewpoint | hardest real-world shift; faithfulness floor |

Stress test = **ImageNet-C** (a continuous severity dial 1–5) gives the smooth CFS-vs-shift
curve; **ImageNet-Sketch / ObjectNet** are the hard discrete endpoints where faithfulness
lives or dies.

## 7. Metrics
- **Faithfulness components (per concept):**
  - **Monotonicity** = Spearman correlation between the steering knob and a held-out concept
    readout (smooth, ordered response).
  - **Specificity** = 1 − normalized drift of **off-target** linear probes (only the target
    concept should move).
  - **Sufficiency** = standardized effect size (Cohen's-d-style) of the readout at full knob
    vs the concept's claimed magnitude.
- **Efficiency axis (the implementation-independent headline quantity):** **CFS — the Causal
  Faithfulness Score ∈ [0,1]**, the composite of the three components above. This is the
  paper's single contributed measuring stick (the analog of "bytes/token"): a backbone-/
  library-independent number per concept and per shift level.
- **OOD degradation slope:** ΔCFS per shift level (clean → R → Sketch → C → ObjectNet) and
  the **collapse knee** (shift level where CFS crosses a usability floor), with **bootstrap
  confidence intervals over concepts**.

## 8. Contributions
1. **CFS + on-manifold steering:** a quantitative **Causal Faithfulness Score** for vision
   SAE features (monotonicity × specificity × sufficiency) **plus** an on-manifold
   projected-steering method (`P_M·Δ`) that keeps edits realistic — a measuring stick and a
   faithful editing method that did not exist together.
2. **The controlled on-manifold-vs-naive study:** at matched steering strength, on-manifold
   steering vs naive/off-manifold, random, raw-clamp, and supervised TCAV-style steering,
   with bootstrap CIs over concepts.
3. **The OOD-faithfulness benchmark and its empirical answer:** the first measurement of
   whether vision-SAE concept faithfulness **survives distribution shift** (clean → R →
   Sketch → C → ObjectNet) — an answer either way (faithfulness holds → trustworthy
   interpretability; or collapses → a warning to the field), plus the per-concept
   reliability distribution.

## 9. Reading list (12 papers → seed the ~24-entry bibliography)
1. Cunningham et al. 2023 — *Sparse Autoencoders Find Highly Interpretable Features* — arXiv:2309.08600
2. Bricken et al. 2023 — *Towards Monosemanticity: Decomposing Language Models with Dictionary Learning* (Anthropic)
3. Gao et al. 2024 — *Scaling and Evaluating Sparse Autoencoders* (TopK SAE) — arXiv:2406.04093
4. Templeton et al. 2024 — *Scaling Monosemanticity* (Anthropic)
5. Turner et al. 2023 — *Activation Addition / ActAdd: Steering Language Models* — arXiv:2308.10248
6. Zou et al. 2023 — *Representation Engineering: A Top-Down Approach to AI Transparency* — arXiv:2310.01405
7. Kim et al. 2017 — *Interpretability Beyond Feature Attribution (TCAV)* — arXiv:1711.11279
8. Radford et al. 2021 — *Learning Transferable Visual Models from Natural Language (CLIP)* — arXiv:2103.00020
9. Dosovitskiy et al. 2020 — *An Image is Worth 16x16 Words (ViT)* — arXiv:2010.11929
10. Hendrycks et al. 2020 — *The Many Faces of Robustness (ImageNet-R)* — arXiv:2006.16241
11. Hendrycks & Dietterich 2019 — *Benchmarking Neural Network Robustness (ImageNet-C)* — arXiv:1903.12261
12. Wang et al. 2019 — *Learning Robust Global Representations (ImageNet-Sketch)* — arXiv:1905.13549

(Paper adds standard extras to reach ~24: ObjectNet (Barbu et al. 2019, NeurIPS),
Causal Scrubbing (Chan et al. 2022, Alignment Forum), Makelov et al. *Towards Principled
Evaluation of SAEs* 2405.08366, Rajamanoharan et al. *JumpReLU / Gated SAEs* 2407.14435 /
2404.16014, Marks et al. *Sparse Feature Circuits* 2403.19647, Fel et al. vision
dictionary-learning, Hooker et al. *ROAR* 1806.10758, Geirhos et al. *Texture-vs-Shape
Bias* 1811.12231, etc.)

## 10. Ablations (A1–A5)
- **A1.** SAE type — **TopK vs L1 (vanilla) SAE** — effect on clean magnitudes and CFS.
- **A2.** TopK **`k` (sparsity level)** sweep — how sparsity trades off interpretability vs
  faithfulness; CFS per `k`.
- **A3.** **Manifold-projection rank `r`** sweep — the core knob: low `r` over-constrains
  (effect dies), high `r` lets the edit drift off-manifold; locate the CFS knee.
- **A4.** **Concept-selection interpretability threshold** — how strict the "well-defined
  concept" filter is, and the resulting reliable-concept fraction (the ~10–15% tail); CFS
  per threshold.
- **A5.** **Backbone layer & token choice** — which ViT-B/16 layer, and **patch vs CLS
  tokens** — for SAE training and steering; CFS per choice.

(Core ablation = **A3 × steering-strength** grid, the 2-D design grid in `fig5`/`fig3`.)

## 11. Error taxonomy (5)
| Error Type | Description | Expected Frequency |
|---|---|---|
| Off-Manifold Artifact | naive edit leaves the real-image manifold; readout moves but the change is non-realistic / undecodable (fake causal effect) | High for `naive_steer`/`clamp_steer`; rare for `onmanifold_steer` |
| Specificity Leakage | steering the target concept also moves **off-target** concept probes (entangled / polysemantic direction) | Medium; rises with steering strength |
| Polysemantic / Uninterpretable Feature | the selected SAE feature encodes several concepts (or none clean) → no clean knob; filtered by A4 threshold | Medium pre-selection; ~85% of raw features |
| OOD Collapse | concept is faithful in-distribution but CFS collapses under shift (Sketch/C-high/ObjectNet) — the headline failure mode | Rises sharply past the shift knee |
| Magnitude Saturation | beyond a strength threshold the readout flattens / clips; monotonicity breaks (knee in the strength sweep) | Rises with steering strength |

## 12. Code registry variant names
Base ships **`naive_steer`** (off-manifold activation-addition, so the repo runs and has a
falsifiable comparison point on day one). Steering-method variants registered in
`src/blocks/__init__.py` (the **pluggable component selected by config name** is the
**steering method**, not an attention block):
- **`naive_steer`** — off-manifold activation addition `a ← a + s·d` (baseline, ships in base).
- **`random_steer`** — same form, **random direction** (null/sanity baseline).
- **`clamp_steer`** — clamp the SAE feature to a fixed magnitude, no projection.
- **`onmanifold_steer`** — **ours**: project the edit onto the top-`r` real-image subspace,
  `a ← a + s·(P_M·d)`.

Config knobs: `sae_type` (`topk`/`l1`), `topk_k`, `proj_rank` (`r`), `steer_strength`
(`s`), `concept_select_thresh`, `backbone_layer`, `token_type` (`patch`/`cls`), plus the
matched-strength keys. **Offline smoke task:** a tiny **synthetic SAE over synthetic
activations with a planted concept** — generate Gaussian "activations" with one injected
concept direction, fit a toy TopK SAE, recover the planted feature, steer it with each
variant, and check `onmanifold_steer` yields higher synthetic-CFS than `naive`/`random`
(no model download / no GPU needed; real CLIP ViT-B/16 + ImageNet shifts are `TODO(M2)`).

## 13. Analytic cost helper (for `utils.py` + EDA notebook)
```python
def cfs_score(monotonicity: float, specificity: float, sufficiency: float,
              weights=(1.0, 1.0, 1.0)) -> float:
    """Causal Faithfulness Score in [0,1]: a weighted *harmonic* mean of the three
    components, so a near-zero in any single axis (e.g. an unspecific edit) drags the
    whole score down — faithfulness requires all three at once."""
    # each component clipped to [0,1]; harmonic mean => conjunctive ('AND') semantics
    ...

def onmanifold_projection_residual(delta, basis_r) -> float:
    """Fraction of the edit that lies OFF the top-r real-image subspace:
    ||delta - P_M·delta|| / ||delta||  (0 = fully on-manifold; the manifold-faithfulness
    diagnostic that distinguishes onmanifold_steer from naive_steer)."""
    ...
```
Provide a `faithfulness(variant, cfg)` **dispatcher** that returns CFS per steering variant
so `run_experiments` and the EDA notebook share one scoring model, and report a `cfs`
column (plus its three components and `offmanifold_residual`) in `results/metrics_all.csv`.

## 14. Headline equation (methodology)
**CFS** `= HM(monotonicity, specificity, sufficiency)` ∈ [0,1] (harmonic mean →
conjunctive: faithful only if *all three* hold). **On-manifold edit** `a' = a + s·(P_M·Δ)`,
where `P_M = U_r U_rᵀ` projects onto the top-`r` real-image activation subspace; naive
steering is the `r → ∞` (`P_M = I`) special case. The **OOD answer** is the curve
`CFS(shift_level)` and its slope `ΔCFS/Δshift`.

## 15. Figure manifest (9 — fixed filenames; paper & figure agents must match)
| File | Type | Shows |
|---|---|---|
| `fig_overview.png` | schematic | the whole study: frozen CLIP ViT-B/16 → TopK SAE on patch activations → thousands of concept directions → select well-defined concepts → on-manifold steer → measure CFS → sweep across the OOD ladder |
| `fig_method.png` | schematic | the on-manifold steering + CFS computation block: raw edit `Δ` → project `P_M·Δ` onto top-`r` real-image subspace → knob `s` → readout/off-target probes → monotonicity × specificity × sufficiency → CFS |
| `fig1_cfs_ood_sweep.png` | data | **HEADLINE**: CFS vs shift severity (clean → R → Sketch → C-1..5 → ObjectNet) for on-manifold vs naive; collapse knee marked, bootstrap CI band |
| `fig2_faithfulness_pareto.png` | data | specificity vs effect-size/monotonicity Pareto across all 5 steering methods; on-manifold on the frontier, random in the corner |
| `fig3_strength_sweep.png` | data | CFS vs steering strength `s` with a marked knee (magnitude-saturation onset) |
| `fig4_concept_reliability.png` | data | distribution of per-concept CFS — the long unreliable mass and the ~10–15% reliable tail (selection histogram) |
| `fig5_ood_heatmap.png` | data | 2-D grid: shift type (rows) × steering method (cols) → CFS heatmap; best cell boxed (on-manifold, clean) |
| `fig6_monotonicity_curve.png` | data | concept readout vs knob: smooth/monotone (on-manifold) vs jagged/non-monotone (naive) for an example concept |
| `fig7_by_method_bar.png` | data | mean CFS by steering variant (supervised / on-manifold / clamp / naive / random) with **bootstrap confidence intervals** |

All data figures use **illustrative placeholder numbers** (captions marked
"Illustrative" until real CLIP + ImageNet-shift runs land at M3).
