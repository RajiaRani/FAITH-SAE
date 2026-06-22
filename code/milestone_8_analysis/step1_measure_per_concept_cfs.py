#!/usr/bin/env python3
# ===========================================================================
#  step1_measure_per_concept_cfs.py  —  Milestone 8 (Analysis), Step 1
#  Regenerate the synthetic bank + SAE LOCALLY, select testable concepts, and
#  MEASURE a per-concept Causal Faithfulness Score (CFS) for EACH steering
#  method across the OOD shift ladder. This is the raw material the bootstrap
#  (step2) and the figures + findings (step3, step4) chew on.
#  FAITH-SAE  ·  author: Rajia Rani  ·  for research and educational purposes only
# ===========================================================================
#
#  ============ READ THIS FIRST: every term, from absolute zero ============
#
#  ACTIVATION
#    The list of numbers a neural network produces inside itself when it looks at
#    one input — its private "notes" about that input. Here each note is a length-
#    64 vector. We MAKE UP such vectors (synthetic) so the whole thing runs
#    offline on a laptop CPU. (Real CLIP ViT-B/16 notes are 768 numbers long.)
#    Tiny example: one activation is a list like [0.3, -1.2, 0.0, ...] (64 entries).
#
#  THE MANIFOLD (the "sheet")
#    Real activations do NOT fill the whole 64-D space; they cluster near a thin,
#    low-dimensional SHEET (the "manifold"), like the 2-D surface of a balloon
#    inside a 3-D room. We build our synthetic bank to live on a 24-D sheet so
#    there is a real sheet to project onto (on-manifold) or leak off of (naive).
#    Tiny number: a 24-D sheet inside a 64-D room leaves 40 directions "off-sheet".
#
#  CONCEPT DIRECTION  d
#    A fixed unit vector in activation space. Turning the concept "up" means
#    pushing activations along d. The SAE's decoder column j IS the concept
#    direction for feature j. We PLANT a handful of known directions in the bank
#    so we know the right answer to check against.
#
#  STEERING  &  STEERING STRENGTH  s
#    Deliberately editing an activation to turn a concept up/down, then watching
#    what changes. `s` is HOW HARD you push: a' = a + s*d. Tiny example:
#    a=[1,0], d=[0,1], s=3  ->  a'=[1,3]. EVERY method here uses the SAME s
#    (matched strength), so any score difference is caused by the METHOD, not by
#    one method pushing harder.
#
#  THE FIVE METHODS (what differs between them; names from DESIGN_BRIEF §12)
#    supervised_steer — steer the PLANTED ground-truth direction (label-expensive
#                       "gold" reference, TCAV-style). The ceiling a good
#                       unsupervised method should approach.
#    onmanifold_steer — OURS: project the edit onto the top-r real-image sheet,
#                       a' = a + s*(P_M d), so the edit stays realistic.
#    clamp_steer      — clamp the SAE feature to a fixed magnitude, no projection.
#    naive_steer      — baseline: raw off-manifold add a' = a + s*d (no projection).
#    random_steer     — null/sanity: add a RANDOM direction (no real concept).
#
#  CAUSAL FAITHFULNESS SCORE  (CFS)  — the headline number, in [0,1]
#    Asks: is this edit a REAL, clean causal lever? It is the HARMONIC MEAN
#    (an "AND": all three must be high) of:
#      * MONOTONICITY — turn the knob up, does the target readout go up SMOOTHLY
#        and in order? (Spearman rank correlation of knob vs readout.)
#      * SPECIFICITY  — does ONLY the target move, while OFF-target probes stay
#        flat? (1 - off-target drift.) Off-manifold edits smear -> low specificity.
#      * SUFFICIENCY  — is the effect BIG enough to matter? (a standardized
#        Cohen's-d effect size.)
#    Example: HM(0.9,0.9,0.9)=0.90, but HM(0.9,0.9,0.05)=0.13 — one weak axis
#    tanks the whole score. We REUSE the project's cfs_score() for this, so the
#    score is computed exactly the way the rest of the pipeline computes it.
#
#  SHIFT RUNG  (distribution shift, simulated)
#    A named level of "how out-of-distribution is the test data". Clean = the data
#    the SAE was trained on. Harder rungs = we CORRUPT the test activations a
#    little more (blur the sheet + add off-sheet noise), so the very subspace the
#    on-manifold method relies on stops describing the test data. Real runs swap in
#    clean ImageNet -> ImageNet-R -> Sketch -> ImageNet-C(1..5) -> ObjectNet.
#
#  ============ WHAT THIS SCRIPT DOES ============
#    1. Regenerate the synthetic activation bank locally (NO other milestone).
#    2. Train the project's real TopK SAE on it (a few hundred CPU steps).
#    3. Select the cleanest `n_select` discovered features as the testable set.
#    4. For each shift rung, corrupt the test activations, and for each steering
#       method MEASURE a PER-CONCEPT CFS (mono/spec/suff -> cfs) by real sweeps.
#    5. Write outputs/per_concept_cfs.csv (variant, shift, concept_id, cfs, ...).
#
#  RUN:  /usr/bin/python3 step1_measure_per_concept_cfs.py
#  ========================================================================

from __future__ import annotations

import csv

from _common import banner, load_cfg, outpath


# ---------------------------------------------------------------------------
#  Spearman rank correlation (the monotonicity measure) — Pearson on ranks, so
#  no scipy is needed inside the hot loop. Measures: does the readout rise in
#  ORDER as the knob rises? 1.0 = perfectly ordered, ~0 = no consistent response.
#  Tiny example: knob (0,1,2,3) vs readout (0.1,0.4,0.7,0.9) -> ~1.0 (monotone);
#  vs (0.1,0.9,0.2,0.8) -> ~0.0 (jagged).
# ---------------------------------------------------------------------------
def _spearman(a, b):
    import torch
    ar = a.argsort().argsort().float()
    br = b.argsort().argsort().float()
    ar = ar - ar.mean()
    br = br - br.mean()
    return float((ar * br).sum() / ((ar.norm() * br.norm()) + 1e-8))


# ---------------------------------------------------------------------------
#  The synthetic "real-image" activation bank — regenerated HERE so milestone 8
#  is fully INDEPENDENT of every other milestone (contract: regenerate inputs
#  locally). Gaussian activations living on a thin 24-D sheet, with a handful of
#  planted concept directions injected at random strengths. Same spirit as the
#  earlier milestones, kept here so this folder stands alone.
# ---------------------------------------------------------------------------
def make_activation_bank(cfg: dict):
    import torch
    g = torch.Generator().manual_seed(cfg["seed"])
    n_img, n_pat, dim = cfg["n_images"], cfg["n_patches"], cfg["dim"]
    mdim = cfg["manifold_dim"]

    # The sheet M: a random mdim-dimensional subspace of the dim-D space.
    raw_basis = torch.randn(dim, mdim, generator=g)
    U, _ = torch.linalg.qr(raw_basis)            # [dim, mdim] orthonormal sheet M

    # Planted concept directions, lying INSIDE the sheet and only mildly
    # separated (real concepts overlap, which is why steering one nudges others).
    nc = cfg["n_concepts"]
    coeff = torch.randn(mdim, nc, generator=g)
    concepts = (U @ coeff).T                      # [nc, dim], inside M
    concepts = concepts / (concepts.norm(dim=1, keepdim=True) + 1e-8)

    # Base activations: random combinations WITHIN the sheet (+ tiny off-sheet
    # noise), so the bank's natural directions ARE the sheet's directions.
    latent = torch.randn(n_img, n_pat, mdim, generator=g)
    bank = latent @ U.T                           # [n_img, n_pat, dim], in M
    bank = bank + cfg["off_manifold_noise"] * torch.randn(n_img, n_pat, dim, generator=g)

    # Switch ON a random subset of concepts per image at random strengths.
    for c in range(nc):
        present = (torch.rand(n_img, 1, 1, generator=g) < 0.4).float()
        amp = present * (1.5 + 2.0 * torch.rand(n_img, 1, 1, generator=g))
        bank = bank + amp * concepts[c].view(1, 1, dim)

    return bank, concepts, U


def train_sae(cfg: dict, bank):
    """Train the project's real TopK SAE (src/model.py) on the bank.

    We use the SAE submodule of the project's FaithSAE model (the only trainable
    part). The decoder columns it learns ARE the concept directions we steer.
    """
    import torch
    from src.model import _build

    FaithSAE = _build()
    model = FaithSAE(cfg)
    sae = model.sae
    opt = torch.optim.AdamW(sae.parameters(), lr=cfg["lr"])

    g = torch.Generator().manual_seed(cfg["seed"] + 100)
    n = bank.shape[0]
    for step in range(cfg["steps"]):
        idx = torch.randint(0, n, (cfg["batch"],), generator=g)
        a = bank[idx]
        _, _, loss = sae(a)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, cfg["steps"] // 5) == 0 or step == cfg["steps"] - 1:
            print(f"    SAE train step {step:4d}  recon_MSE = {float(loss):.4f}")
    return model


def select_concepts(model, concepts, cfg):
    """Pick the SAE features that best match a planted concept (the testable set).

    For each planted ground-truth direction we find the SAE decoder columns most
    aligned with it (largest |cosine|), and keep the top `n_select` distinct
    matches. This is the offline stand-in for the paper's interpretability filter
    (only the ~10-15% of features that steer cleanly survive selection)."""
    import torch
    with torch.no_grad():
        W = model.sae.dec.weight                  # [dim, sae_dim]; columns = dirs
        Wn = W / (W.norm(dim=0, keepdim=True) + 1e-8)
        scored = []
        for ci in range(concepts.shape[0]):
            d = concepts[ci] / (concepts[ci].norm() + 1e-8)
            cos = (Wn.T @ d).abs()                 # [sae_dim] alignment per feature
            # take the few best-aligned features for THIS planted concept
            top = torch.topk(cos, k=8).indices.tolist()
            for feat in top:
                scored.append((float(cos[feat]), feat, ci))
    # sort by alignment, dedupe features, keep the cleanest n_select
    scored.sort(reverse=True)
    seen, selected = set(), []
    for cosval, feat, ci in scored:
        if feat in seen:
            continue
        seen.add(feat)
        selected.append({"feature_id": feat, "planted_id": ci, "align": round(cosval, 4)})
        if len(selected) >= cfg["n_select"]:
            break
    return selected


def estimate_basis(bank, cfg):
    """Top-r real-image subspace U_r via PCA (SVD) of the CLEAN bank. This is the
    sheet the on-manifold method projects onto AND the reference the off-manifold
    residual is measured against. Estimated ONCE from in-distribution data — which
    is exactly why heavy shift eventually breaks on-manifold steering too."""
    import torch
    flat = bank.reshape(-1, cfg["dim"])
    flat = flat - flat.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(flat, full_matrices=False)
    r = min(cfg["proj_rank"], Vh.shape[0])
    return Vh[:r].T                               # [dim, r] = U_r


def corrupt(bank, basis, noise, seed):
    """Simulate distribution shift on the TEST activations.

    Two ingredients, both growing with the rung's `noise`:
      * blur the sheet — shrink the on-sheet part a little (style/texture loss);
      * add OFF-sheet noise — push points off the very subspace U_r the
        on-manifold method trusts.
    At noise=0 this returns the clean bank unchanged (the in-distribution rung)."""
    import torch
    if noise <= 0.0:
        return bank
    g = torch.Generator().manual_seed(seed)
    flat = bank.reshape(-1, bank.shape[-1])
    on = flat @ basis @ basis.T                   # the on-sheet component
    off = flat - on                               # the off-sheet component
    # shrink the on-sheet structure (lose some concept signal) and inject new
    # off-sheet noise (move into regions the model never trained on).
    blurred = (1.0 - 0.18 * noise) * on + off
    blurred = blurred + noise * torch.randn(flat.shape, generator=g)
    return blurred.reshape(bank.shape)


def measure_cfs(model, steer, test_flat, basis, planted_dir, sae_concept,
                off_dirs, cfg, use_planted: bool):
    """Per-concept CFS + off-manifold residual for ONE steering method on ONE
    (possibly corrupted) test set.

    Sweep the knob s = 0..steer_strength, read out the target concept and several
    off-target probes, and turn the sweep into monotonicity / specificity /
    sufficiency, then combine with the project's cfs_score(). We ALSO measure the
    OFF-MANIFOLD RESIDUAL of the effective edit (a' - a) against the clean sheet
    U_r with the project's onmanifold_projection_residual() — the genuine measured
    quantity that SEPARATES on-manifold (residual ~0) from naive (residual large)
    even when their harmonic-mean CFS is close. `use_planted` picks the SUPERVISED
    gold readout direction (planted truth) vs the SAE's own decoder column.
    Every number below is MEASURED — nothing is hard-coded."""
    import torch
    from src.utils import cfs_score, onmanifold_projection_residual

    smax = cfg["steer_strength"]
    n_steps = cfg["n_knob_steps"]
    # the direction we EDIT along: planted truth for supervised, else SAE column.
    d_edit = planted_dir if use_planted else model.sae.concept_direction(sae_concept)
    # the held-out readout direction we MEASURE: always the planted truth, so the
    # comparison across methods is apples-to-apples (does the real concept move?).
    d_tgt = planted_dir / (planted_dir.norm() + 1e-8)

    with torch.no_grad():
        def readout(a, d):
            return (a * d).sum(-1)                 # projection onto direction d

        knobs, tgt_read = [], []
        off_reads = [[] for _ in off_dirs]
        for j in range(n_steps):
            s = smax * j / (n_steps - 1)
            a_s = steer(test_flat, d_edit, s, sae=model.sae,
                        concept=sae_concept, basis=basis)
            knobs.append(s)
            tgt_read.append(float(readout(a_s, d_tgt).mean()))
            for i, do in enumerate(off_dirs):
                off_reads[i].append(float(readout(a_s, do).mean()))

        knobs = torch.tensor(knobs)
        tr = torch.tensor(tgt_read)

        # MONOTONICITY: ordered smooth response of the target readout to the knob.
        monotonicity = max(_spearman(knobs, tr), 0.0)
        # SPECIFICITY: 1 - (WORST off-target movement / target movement). The worst
        # off-target surfaces the leakage that off-manifold edits cause.
        tgt_move = (tr.max() - tr.min()).abs() + 1e-6
        worst_off = 0.0
        for col in off_reads:
            col = torch.tensor(col)
            worst_off = max(worst_off, float((col.max() - col.min()).abs()))
        specificity = float(max(0.0, 1.0 - worst_off / tgt_move))
        # SUFFICIENCY: standardized (Cohen's-d) effect size at full knob vs none.
        base = steer(test_flat, d_edit, 0.0, sae=model.sae,
                     concept=sae_concept, basis=basis)
        full = steer(test_flat, d_edit, smax, sae=model.sae,
                     concept=sae_concept, basis=basis)
        r0 = readout(base, d_tgt)
        r1 = readout(full, d_tgt)
        pooled = (r0.std() + r1.std()) / 2 + 1e-6
        d_eff = float((r1.mean() - r0.mean()).abs() / pooled)
        sufficiency = min(d_eff / 4.0, 1.0)        # d~4 is "ample" -> 1.0

        # OFF-MANIFOLD RESIDUAL of the EFFECTIVE edit (a'-a) at full knob vs the
        # CLEAN sheet U_r (the reference basis). 0 = the edit stays on the sheet
        # (on-manifold), large = it flies off (naive/clamp/random). MEASURED with
        # the project's own helper so it matches the rest of the pipeline.
        eff_edit = (full - base).reshape(-1, full.shape[-1]).mean(0)
        residual = onmanifold_projection_residual(eff_edit, basis)

    cfs = cfs_score(monotonicity, specificity, sufficiency)
    return {
        "monotonicity": round(monotonicity, 4),
        "specificity": round(specificity, 4),
        "sufficiency": round(sufficiency, 4),
        "offmanifold_residual": round(float(residual), 4),
        "cfs": round(cfs, 4),
    }


def main() -> None:
    import torch
    from src.model import build_steer
    from src.utils import set_seed

    cfg = load_cfg()
    set_seed(cfg["seed"])
    banner("STEP 1 — regenerate bank+SAE, select concepts, measure per-concept CFS")

    # 1) Bank + SAE (regenerated locally; this folder is self-contained).
    bank, planted, _ = make_activation_bank(cfg)
    print(f"  bank shape = {tuple(bank.shape)}  (n_images x n_patches x dim); "
          f"planted concepts = {planted.shape[0]}")
    model = train_sae(cfg, bank)

    # 2) Select the testable concept set and the in-distribution sheet U_r.
    selected = select_concepts(model, planted, cfg)
    basis = estimate_basis(bank, cfg)
    print(f"  selected {len(selected)} testable concepts "
          f"(SAE feature ids, best-aligned to planted directions)")

    # 3) Off-target probe panel = a spread of OTHER SAE features that must stay
    #    flat. A faithful edit moves the target and nothing else.
    g = torch.Generator().manual_seed(cfg["seed"] + 5)
    sel_feats = {s["feature_id"] for s in selected}
    off_ids, tries = [], 0
    while len(off_ids) < 8 and tries < 200:
        cand = int(torch.randint(0, cfg["sae_dim"], (1,), generator=g))
        if cand not in sel_feats and cand not in off_ids:
            off_ids.append(cand)
        tries += 1
    with torch.no_grad():
        off_dirs = []
        for oc in off_ids:
            do = model.sae.concept_direction(oc)
            off_dirs.append(do / (do.norm() + 1e-8))

    # 4) Sweep every shift rung x every method x every concept.
    rows = []
    flat_clean = bank.reshape(-1, cfg["dim"])[:512]   # a fixed slice for speed
    for rung in cfg["shift_rungs"]:
        test_flat = corrupt(flat_clean, basis, rung["noise"], cfg["seed"] + 31)
        for variant in cfg["variants"]:
            # supervised steers along the PLANTED direction (gold) using the naive
            # add form; that planted direction already lives ON the sheet, so it
            # reads as faithful — the label-expensive ceiling.
            steer = build_steer(variant if variant != "supervised_steer"
                                else "naive_steer", cfg)
            use_planted = (variant == "supervised_steer")
            for s in selected:
                d_planted = planted[s["planted_id"]]
                m = measure_cfs(model, steer, test_flat, basis, d_planted,
                                s["feature_id"], off_dirs, cfg, use_planted)
                rows.append({
                    "variant": variant,
                    "shift": rung["name"],
                    "shift_noise": rung["noise"],
                    "concept_id": s["feature_id"],
                    "monotonicity": m["monotonicity"],
                    "specificity": m["specificity"],
                    "sufficiency": m["sufficiency"],
                    "offmanifold_residual": m["offmanifold_residual"],
                    "cfs": m["cfs"],
                })
        # quick per-rung console summary (mean CFS + mean off-manifold residual of
        # ours vs naive at this rung — the residual is the clean separator).
        def mean_at(v, key):
            xs = [r[key] for r in rows
                  if r["variant"] == v and r["shift"] == rung["name"]]
            return sum(xs) / len(xs)
        print(f"  rung {rung['name']:>10}: mean CFS  "
              f"on-manifold={mean_at('onmanifold_steer', 'cfs'):.3f}"
              f"  naive={mean_at('naive_steer', 'cfs'):.3f}   |   "
              f"off-manifold residual  "
              f"on-manifold={mean_at('onmanifold_steer', 'offmanifold_residual'):.3f}"
              f"  naive={mean_at('naive_steer', 'offmanifold_residual'):.3f}")

    # 5) Write the per-concept table.
    out = outpath("per_concept_cfs.csv")
    fields = ["variant", "shift", "shift_noise", "concept_id",
              "monotonicity", "specificity", "sufficiency",
              "offmanifold_residual", "cfs"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  saved {len(rows)} rows -> {out}")
    print("STEP 1 done. Next: step2 bootstraps confidence intervals over concepts.")


# REAL RUN (M8): replace make_activation_bank/train_sae/corrupt with the MEASURED
# per-concept CFS from milestones 5-7 (real CLIP ViT-B/16 activations across the
# real OOD ladder clean ImageNet -> R -> Sketch -> ImageNet-C(1..5) -> ObjectNet).
# Point step2/step3/step4 at that real per_concept_cfs.csv; the bootstrap, figures,
# and findings are unchanged. Everything downstream is already data-driven.
if __name__ == "__main__":
    main()
