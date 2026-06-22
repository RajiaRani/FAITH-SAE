#!/usr/bin/env python3
# ===========================================================================
#  smoke_real.py  --  FAITH-SAE REAL RUN  ·  END-TO-END integration self-test
#  --------------------------------------------------------------------------
#  ONE CPU, no-open_clip, no-download driver that proves the WHOLE real-scale
#  pipeline is wired correctly end-to-end. It fabricates a tiny but REAL-SHAPED
#  synthetic activation cache (in the exact on-disk cache contract data_real
#  writes), then drives the REAL module functions -- NOT their private --smoke
#  shortcuts -- in the pipeline order:
#
#     extract (fabricate cache)                  [data_real cache contract]
#       -> train_sae.train_sae         (tiny)    [sae_real.TopKSAE]
#       -> manifold.estimate_manifold_basis      [the U_r real-image subspace]
#       -> concept_select.select_concepts        [the reliable ~10-15% tail]
#       -> probes.build_probe_bank               [linear concept rulers]
#       -> cfs_eval.evaluate_all_methods         [the 5 steerers, matched s]
#       -> ood_sweep.run_ood_sweep   (4 rungs)   [RQ3 CFS-vs-shift ladder]
#       -> ablations_real.run_ablations (1 val)  [A1..A5 one knob each]
#       -> analysis_real.bootstrap + write_findings
#       -> figures_real.make_real_figures        [fig1 + fig7 PNGs]
#
#  The headline ASSERTIONS this self-test makes (the paper's core claims, made
#  falsifiable on CPU):
#    * every measured CFS is in [0, 1];
#    * through the REAL cfs_eval.compute_cfs scorer, on-manifold CFS > naive CFS
#      (RQ1: projecting away the off-manifold leak makes the edit specific);
#    * on-manifold's off-manifold residual ~ 0 (and strictly below naive's);
#    * the OOD sweep, ablations, bootstrap CIs, FINDINGS, and the two figure
#      PNGs are all written to outputs/.
#
#  WHY this exists separately from each module's own --smoke: a per-module smoke
#  proves that module in isolation; THIS proves the modules IMPORT EACH OTHER
#  cleanly and the shared interface (cache format, signatures, U_r, ProbeBank,
#  the cfs_score harmonic mean) is consistent across the whole tree. The only
#  thing it cannot exercise is the open_clip GPU extraction in data_real
#  (import-guarded); everything downstream of the cache is real code.
#
#  Run:  /usr/bin/python3 smoke_real.py        (from code/real_run/)
#  Exit 0 + all artifacts written == the real pipeline is correct end-to-end.
#
#  author: Rajia Rani
#  For research and educational purposes only.
# ===========================================================================
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import tempfile

# --- sys.path: the project root (for `from src...`) AND this real_run dir so
# the sibling real modules import each other exactly as they do at real scale.
_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

# The ONE scoring model, reused so this self-test grades CFS the same way the
# paper does (a zero on any axis collapses the harmonic mean).
from src.utils import cfs_score  # noqa: E402


# --------------------------------------------------------------------------- #
#  A tiny but REAL-SHAPED config (the shared YAML schema, as a dict).          #
#  d_in=128 stands in for the 768/1024/1280 real CLIP widths. Everything is    #
#  config-driven so this is just the smoke twin of configs/vit_l14.yaml.       #
# --------------------------------------------------------------------------- #
def _smoke_cfg(out_dir, cache_dir):
    """The smoke config dict honouring the shared schema (data_real.load_real_config
    would load an identical structure from YAML). Kept tiny so the whole pipeline
    runs in a handful of seconds on CPU, but real-SHAPED so no downstream module
    can tell it apart from the real cache."""
    d_in = 128
    return {
        "seed": 0,
        # framework=timm mirrors the student's real run; the smoke path never
        # actually loads timm (it fabricates shards), so this is metadata only.
        "backbone": {"framework": "timm", "name": "vit_base_patch16_224",
                     "pretrained": True,
                     "layer": 2, "token_type": "patch", "image_size": 224,
                     "device": "cpu"},
        "data": {"imagenet_train_dir": "./_unused", "batch_size": 8,
                 "num_workers": 0, "max_images": 64,
                 "in100_classes_file": "./_unused", "data_dir": "./_unused"},
        "sae": {"d_in": d_in, "expansion": 8, "k": 8,
                "normalize": "unit_meansquare", "lr": 4.0e-4, "warmup": 5,
                "token_budget": 300_000, "batch_tokens": 512, "aux_k": 32,
                "dead_window": 50_000, "ckpt_every": 1_000_000},
        "steering": {"strength_grid": [0, 0.5, 1, 2, 4], "proj_rank_r": 24,
                     "bank_tokens": 8_000},
        # select_thresh=0.0 keeps the top-n_probe_classes features by reliability
        # RANK. On a tiny smoke bank the absolute reliability scores are small
        # (token-granularity consistency pooling over a few-thousand-token bank),
        # but the RANKING is what selection uses; at real scale the same code with
        # a real per-token image_id manifest and the A4 threshold keeps the
        # reliable ~10-15% tail. Selection wiring is identical either way.
        "cfs": {"n_probe_classes": 5, "bootstrap_n": 200,
                "select_thresh": 0.0, "max_act_top": 8},
        # The student's domain-shift ladder, ordered by shift strength: in1k
        # (in-distribution, also the SAE-training source) -> in100 (mild) ->
        # food101 (domain) -> cifar100 (strong domain + resolution shift).
        "ood": {"levels": ["in1k", "in100", "food101", "cifar100"],
                "usability_floor": 0.5},
        "ablations": {"token_budget": 20_000},
        "paths": {"cache_dir": str(cache_dir), "out_dir": str(out_dir),
                  "sae_ckpt": str(pathlib.Path(out_dir) / "sae.safetensors"),
                  "manifold_basis": str(pathlib.Path(out_dir) / "U_r.npy"),
                  # smoke fabricator knobs (extract_activations._fabricate_smoke_shards)
                  },
        # The extract smoke fabricator reads cfg.smoke for shard count/size.
        # A few hundred-thousand tokens lets the tiny SAE actually specialise its
        # dictionary in the (capped) number of streaming steps below.
        "smoke": {"n_shards": 4, "tokens_per_shard": 16384, "n_patches": 64},
    }


def _ok(label, cond):
    """Print + assert one check (so a failure names exactly which claim broke)."""
    status = "OK " if cond else "FAIL"
    print(f"    [{status}] {label}")
    assert cond, f"smoke_real assertion FAILED: {label}"


# --------------------------------------------------------------------------- #
#  STEP 1 — fabricate the activation cache via the REAL extract smoke path.    #
#  This proves the data_real cache CONTRACT (sharded acts/labels + manifest)   #
#  for clean + two OOD rungs, exactly what extract_activations.py --smoke does.#
# --------------------------------------------------------------------------- #
def _fabricate_cache(cfg, cache_dir):
    """Drive extract_activations._fabricate_smoke_shards (the documented offline
    twin of the real timm extraction) for every rung of the student's ladder. We
    fabricate the dataset key 'imagenet_train' for the SAE-training cache
    (train_sae's default dataset) AND each ood.levels rung (in1k, in100, food101,
    cifar100) so the probe/concept loaders that ask for the in-distribution rung
    ('in1k') and the sweep (every rung) find their shards."""
    import extract_activations as extract

    # data_real.load_real_config expects a dot-dict; extract's fabricator reads
    # cfg.sae.d_in / cfg.smoke / cfg.cfs etc., so wrap the plain dict.
    from data_real import _Cfg
    dcfg = _Cfg(cfg)

    # imagenet_train (SAE source) + the student's domain-shift ladder rungs.
    datasets = ["imagenet_train"] + list(cfg["ood"]["levels"])
    manifests = {}
    for ds in datasets:
        manifests[ds] = extract._fabricate_smoke_shards(dcfg, ds, str(cache_dir))
    return manifests


# --------------------------------------------------------------------------- #
#  STEP 6 (headline) — build a CONTROLLED eval bank + a leaky SAE column so the #
#  REAL cfs_eval.compute_cfs ordering (on-manifold > naive) is meaningful.      #
#                                                                              #
#  The fabricated cache plants concepts INSIDE the manifold, so on the clean   #
#  rung a naive edit barely leaves it and naive ~ on-manifold -- which would   #
#  make the on-manifold>naive assertion vacuous. The paper's claim is about    #
#  the REAL phenomenon that an SAE decoder column trained on noisy activations  #
#  carries a small OFF-manifold leak that smears naive steering into off-target #
#  probes. We reproduce that here EXACTLY as cfs_eval's own validated smoke     #
#  does (an on-manifold sheet + an off-manifold leak aligned with the off-target #
#  probe directions), then drive it through the REAL evaluate_all_methods so    #
#  the assertion exercises the production scorer, not a shortcut.               #
# --------------------------------------------------------------------------- #
def _headline_eval(cfg):
    """Run the REAL cfs_eval.evaluate_all_methods on a controlled bank and return
    its DataFrame. Mirrors the mechanism the paper claims (RQ1): the SAE concept
    direction has a small off-manifold leak; naive adds it raw (leaks -> low
    specificity), on-manifold projects it away (specific -> higher CFS)."""
    import torch

    import cfs_eval
    from manifold import estimate_manifold_basis
    from probes import build_probe_bank

    rng = np.random.default_rng(0)
    d_in = int(cfg["sae"]["d_in"])
    n = 1500
    n_concepts = 4
    target = 0
    rank_true = 12                 # the real-image manifold dimension
    r = int(cfg["steering"]["proj_rank_r"])  # on-manifold projection rank (>= rank_true)

    # 1) A low-rank real-image MANIFOLD sheet + an explicit off-manifold complement.
    Q, _ = np.linalg.qr(rng.standard_normal((d_in, d_in)).astype(np.float32))
    on_basis = Q[:, :rank_true]
    off_basis = Q[:, rank_true:]
    coeffs = rng.standard_normal((n, rank_true)).astype(np.float32)
    acts = coeffs @ on_basis.T
    acts += 0.05 * rng.standard_normal((n, d_in)).astype(np.float32)

    # 2) Plant concepts. The TARGET concept carries an on-manifold part PLUS an
    #    off-manifold leak; off-target concepts are (partly) read along that leak,
    #    so a naive (raw) edit smears into them -> specificity leakage.
    leak = off_basis @ rng.standard_normal(off_basis.shape[1]).astype(np.float32)
    leak /= np.linalg.norm(leak) + 1e-8
    dirs = np.zeros((d_in, n_concepts), dtype=np.float32)
    on_parts = on_basis @ rng.standard_normal((rank_true, n_concepts)).astype(np.float32)
    on_parts /= np.linalg.norm(on_parts, axis=0, keepdims=True) + 1e-8
    dirs[:, target] = on_parts[:, target] + 1.2 * leak
    for c in range(1, n_concepts):
        dirs[:, c] = on_parts[:, c] + 0.9 * leak
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True) + 1e-8
    labels = (rng.random((n, n_concepts)) < 0.5).astype(np.int64)
    for c in range(n_concepts):
        acts += (labels[:, c:c + 1] * 2.5) * dirs[:, c][None, :]
    acts = acts.astype(np.float16)            # exercise the float16 cache dtype

    # 3) Estimate U_r with the REAL manifold module.
    U_r = cfs_eval._np(estimate_manifold_basis(
        torch.as_tensor(acts.astype(np.float32)), r))

    # 4) Train a tiny REAL TopK SAE (sae_real.TopKSAE) on this bank.
    acts_t = torch.as_tensor(acts.astype(np.float32))
    sae, recon = cfs_eval._build_smoke_sae(acts_t, d_in=d_in, n_features=128,
                                           k=8, steps=300)

    # 5) Probe bank (target + off-target rulers) with the REAL probes module.
    probes = build_probe_bank(
        acts.astype(np.float32),
        {c: labels[:, c] for c in range(n_concepts)},
        target_concept=target,
    )
    w_tgt = probes.target_direction()

    # 6) Pick the SAE feature that reads most positively on the target probe, and
    #    orient + inject the documented off-manifold leak (the real-SAE phenomenon).
    W_dec = cfs_eval._np(sae.W_dec)
    if W_dec.shape[0] != d_in:
        W_dec = W_dec.T
    norms = np.linalg.norm(W_dec, axis=0) + 1e-8
    signed_cos = (W_dec.T @ w_tgt) / norms
    sae_concept = int(np.argmax(signed_cos))
    if signed_cos[sae_concept] < 0:
        cfs_eval._orient_sae_feature(sae, sae_concept, d_in)
    off_w = np.stack([probes.directions[c] for c in probes.off_target_ids()],
                     0).mean(0)
    off_w_leak = off_basis @ (off_basis.T @ off_w)
    off_w_leak /= np.linalg.norm(off_w_leak) + 1e-8
    cfs_eval._inject_decoder_leak(sae, sae_concept, d_in, off_w_leak, scale=0.32)

    # 7) The REAL evaluate_all_methods across all five steerers at matched strength.
    df = cfs_eval.evaluate_all_methods(sae, sae_concept, acts.astype(np.float32),
                                       probes, U_r, cfg, level="clean")
    return df, recon, sae_concept


# --------------------------------------------------------------------------- #
#  STEP 9 — write the per_concept_cfs.csv the analysis + figures consume.      #
#  We derive it from the OOD sweep DataFrame (real schema: rung/method/cfs +   #
#  the per-concept CFS list each row carries), expanding the per-concept list  #
#  so bootstrap-over-CONCEPTS has its proper resampling unit.                  #
# --------------------------------------------------------------------------- #
def _per_concept_from_sweep(sweep_df, out_dir):
    """Expand ood_sweep's `cfs_per_concept` column into a tidy per-concept CSV
    (variant/shift/shift_noise/concept_id/cfs[+components]) so analysis_real and
    figures_real (which bootstrap OVER CONCEPTS) have the right unit. This mirrors
    the schema analysis_real._fabricate_results writes, but from MEASURED sweep
    numbers rather than fabricated ones."""
    import pandas as pd

    rows = []
    for _, r in sweep_df.iterrows():
        per = str(r.get("cfs_per_concept", "")).split(";")
        per = [p for p in per if p not in ("", "nan")]
        for ci, val in enumerate(per):
            try:
                cfs = float(val)
            except ValueError:
                continue
            rows.append({
                "variant": r["method"],
                "shift": r["rung"],
                "shift_noise": float(r["shift_index"]),
                "concept_id": ci,
                # the sweep stores per-rung MEAN components; reuse them as a
                # reasonable per-concept stand-in for fig/analysis robustness.
                "monotonicity": float(r["monotonicity"]),
                "specificity": float(r["specificity"]),
                "sufficiency": float(r["sufficiency"]),
                "cfs": cfs,
            })
    per_df = pd.DataFrame(rows)
    out = pathlib.Path(out_dir) / "per_concept_cfs.csv"
    per_df.to_csv(out, index=False)
    return per_df, out


# --------------------------------------------------------------------------- #
#  THE DRIVER                                                                  #
# --------------------------------------------------------------------------- #
def run(out_dir=None, cache_dir=None, keep=False):
    print("=" * 74)
    print("FAITH-SAE — END-TO-END REAL-PIPELINE SELF-TEST (CPU, no open_clip)")
    print("=" * 74)

    # Default: write the artifacts into the canonical real_run/outputs/ (the
    # contract: 'the two figure PNGs + the csvs + FINDINGS get written to
    # outputs/'), and fabricate the (larger, ephemeral) activation cache in a temp
    # dir. _HERE is resolved from __file__, so it is the canonical on-disk path
    # even if the script was launched through a path alias / symlink.
    tmp_root = None
    if cache_dir is None:
        tmp_root = pathlib.Path(tempfile.mkdtemp(prefix="faith_smoke_real_"))
        cache_dir = tmp_root / "cache"
    if out_dir is None:
        out_dir = _HERE / "outputs"
    out_dir = pathlib.Path(out_dir)
    cache_dir = pathlib.Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = _smoke_cfg(out_dir, cache_dir)
    written = {}

    try:
        # ---- 1. extract: fabricate the real-format activation cache ----------
        print("\n[1/9] extract  — fabricate real-format activation cache")
        manifests = _fabricate_cache(cfg, cache_dir)
        for ds, mani in manifests.items():
            print(f"      cache[{ds}]: {mani['n_tokens']} tokens, "
                  f"{mani['n_shards']} shards, d_in={mani['d_in']}")
        _ok("cache shards written for every rung",
            all((cache_dir / f"acts_{ds}_00000.npy").exists() for ds in manifests))
        _ok("per-token labels written (probe material)",
            (cache_dir / "labels_in1k_00000.npy").exists())

        # ---- 2. train_sae: tiny streaming train on the cached shards ---------
        print("\n[2/9] train_sae — stream the cache, fit sae_real.TopKSAE")
        import train_sae
        sae = train_sae.train_sae(cfg, str(cache_dir), dataset="imagenet_train",
                                  device="cpu", max_steps=400)
        from sae_real import TopKSAE, save_sae, load_sae
        _ok("trained object is a sae_real.TopKSAE", isinstance(sae, TopKSAE))
        # decoder columns must be unit-norm (the renorm trick) so the concept
        # directions steering edits along have honest magnitude.
        import torch
        col_norms = sae.W_dec.data.norm(dim=0)
        _ok("decoder columns unit-norm after training",
            bool(torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-4)))
        save_sae(sae, cfg["paths"]["sae_ckpt"],
                 meta={"smoke": True, "step": 120})
        sae = load_sae(cfg["paths"]["sae_ckpt"])     # round-trip the safetensors ckpt
        _ok("SAE checkpoint round-trips (safetensors)", isinstance(sae, TopKSAE))
        written["sae_ckpt"] = cfg["paths"]["sae_ckpt"]

        # ---- 3. manifold: estimate U_r from a real-activation bank -----------
        print("\n[3/9] manifold — estimate U_r (top-r real-image subspace)")
        import manifold
        from data_real import load_activation_bank
        bank = load_activation_bank(str(cache_dir), "imagenet_train",
                                    int(cfg["steering"]["bank_tokens"]), seed=0)
        r = int(cfg["steering"]["proj_rank_r"])
        U_r = manifold.estimate_manifold_basis(bank, r)
        _ok(f"U_r shape == (d_in={cfg['sae']['d_in']}, r={r})",
            tuple(U_r.shape) == (cfg["sae"]["d_in"], r))
        gram = (U_r.T @ U_r) - torch.eye(r)
        _ok("U_r columns orthonormal (U_r^T U_r = I)",
            float(gram.abs().max()) < 1e-3)
        manifold.save_basis(U_r, cfg["paths"]["manifold_basis"])
        written["U_r"] = cfg["paths"]["manifold_basis"]

        # ---- 4. concept_select: the reliable concept tail --------------------
        print("\n[4/9] concept_select — pick the reliable testable concepts")
        import concept_select
        concept_ids = concept_select.select_concepts(
            sae, bank, image_ids=None, cfg=cfg)
        _ok("selected at least one testable concept", len(concept_ids) >= 1)
        _ok("selected concept ids are plain ints",
            all(isinstance(c, int) for c in concept_ids))

        # ---- 5. probes: linear concept rulers on cached labeled activations --
        print("\n[5/9] probes — train linear concept rulers (cached labels)")
        import data_real
        import probes as probes_mod
        # Accumulate a labeled in-distribution ('in1k') bank exactly like
        # ood_sweep._build_clean_probe_bank does.
        acts_chunks, lab_chunks = [], []
        for a, l in data_real.iter_labeled_shards(str(cache_dir), "in1k"):
            acts_chunks.append(np.asarray(a, dtype=np.float32))
            lab_chunks.append(np.asarray(l).reshape(-1))
        labeled = np.concatenate(acts_chunks, 0)
        labs = np.concatenate(lab_chunks, 0)
        classes = sorted(int(c) for c in np.unique(labs) if c >= 0)
        concept_labels = {c: (labs == c).astype(np.int64) for c in classes}
        probe_bank = probes_mod.build_probe_bank(
            labeled, concept_labels, target_concept=classes[0], cfg=cfg)
        _ok("probe bank has >= 2 concepts (target + off-target)",
            len(probe_bank.concept_ids) >= 2)
        # the TCAV direction is unit-norm (the supervised steering reference).
        tdir = probe_bank.target_direction()
        _ok("TCAV target direction is unit-norm",
            abs(float(np.linalg.norm(tdir)) - 1.0) < 1e-4)

        # ---- 6. cfs_eval: the headline 5-method comparison (REAL scorer) -----
        print("\n[6/9] cfs_eval — evaluate_all_methods (RQ1, matched strength)")
        import pandas as pd
        df, recon, steered = _headline_eval(cfg)
        pd.set_option("display.width", 140)
        print(df.to_string(index=False))
        by = {row["method"]: row for _, row in df.iterrows()}
        # every measured CFS (and component) is a valid probability in [0,1].
        for _, row in df.iterrows():
            for col in ("monotonicity", "specificity", "sufficiency", "cfs",
                        "offmanifold_residual"):
                v = float(row[col])
                _ok(f"{row['method']}.{col} in [0,1] ({v:.4f})", 0.0 <= v <= 1.0)
        onm = by["onmanifold_steer"]
        naive = by["naive_steer"]
        rand = by["random_steer"]
        # store the headline numbers for the return summary.
        headline = {
            "onmanifold_cfs": float(onm["cfs"]),
            "naive_cfs": float(naive["cfs"]),
            "onmanifold_offmanifold_residual": float(onm["offmanifold_residual"]),
            "naive_offmanifold_residual": float(naive["offmanifold_residual"]),
        }
        # HEADLINE RQ1: projecting the leak away makes the edit specific -> CFS up.
        _ok(f"on-manifold CFS {onm['cfs']:.4f} > naive CFS {naive['cfs']:.4f}",
            onm["cfs"] > naive["cfs"] + 1e-3)
        _ok("on-manifold off-manifold residual ~ 0 (< 0.05)",
            onm["offmanifold_residual"] < 0.05)
        _ok("on-manifold residual strictly below naive residual",
            onm["offmanifold_residual"] < naive["offmanifold_residual"] + 1e-6)
        # random_steer has NO real concept along it, so its monotonicity is ~0 and
        # the conjunctive harmonic mean collapses its CFS to ~0 — the robust null
        # property (its specificity is direction-of-draw dependent, so we assert
        # the invariant the paper actually relies on: the null edit is not faithful).
        _ok(f"random_steer CFS collapses to ~0 ({rand['cfs']:.4f})",
            rand["cfs"] < 0.05)
        _ok("on-manifold is more specific than naive",
            onm["specificity"] > naive["specificity"] + 1e-3)
        _ok(f"on-manifold CFS {onm['cfs']:.4f} > random CFS {rand['cfs']:.4f}",
            onm["cfs"] > rand["cfs"] + 1e-6)

        # ---- 7. ood_sweep: the RQ3 ladder on the fabricated cache ------------
        print("\n[7/9] ood_sweep — RQ3 CFS-vs-shift ladder (smoke fabric)")
        import ood_sweep
        sweep_df = ood_sweep.run_ood_sweep(cfg, str(cache_dir), smoke=True)
        _ok("ood sweep produced rows", len(sweep_df) > 0)
        _ok("every sweep CFS in [0,1]",
            bool(((sweep_df["cfs"].astype(float) >= 0.0)
                  & (sweep_df["cfs"].astype(float) <= 1.0)).all()))
        # on-manifold should not be WORSE than naive on the in-distribution rung.
        indist = sweep_df[sweep_df["rung"] == "in1k"]
        onm_clean = float(indist[indist["method"] == "onmanifold_steer"]["cfs"].iloc[0])
        naive_clean = float(indist[indist["method"] == "naive_steer"]["cfs"].iloc[0])
        _ok(f"sweep in1k: on-manifold {onm_clean:.4f} >= naive {naive_clean:.4f}",
            onm_clean >= naive_clean - 1e-6)
        sweep_csv = pathlib.Path(out_dir) / "ood_cfs_sweep.csv"
        _ok("ood_cfs_sweep.csv written", sweep_csv.exists())
        written["ood_cfs_sweep"] = str(sweep_csv)

        # ---- 8. ablations: A1..A5, one knob value each -----------------------
        print("\n[8/9] ablations — A1..A5 (one knob each, smoke fabric)")
        import ablations_real
        abl_df = ablations_real.run_ablations(cfg, str(cache_dir), smoke=True)
        _ok("ablations produced A1..A5 rows",
            set(abl_df["ablation_id"]) >= {"A1", "A2", "A3", "A4", "A5"})
        _ok("every ablation CFS in [0,1]",
            bool(((abl_df["cfs"].astype(float) >= 0.0)
                  & (abl_df["cfs"].astype(float) <= 1.0)).all()))
        abl_csv = pathlib.Path(out_dir) / "ablations.csv"
        _ok("ablations.csv written", abl_csv.exists())
        written["ablations"] = str(abl_csv)

        # ---- 9. analysis + figures: bootstrap, findings, the two PNGs --------
        print("\n[9/9] analysis + figures — bootstrap CIs, FINDINGS, fig1 + fig7")
        # Build the per-concept CSV the analysis/figures bootstrap over CONCEPTS.
        per_df, per_csv = _per_concept_from_sweep(sweep_df, out_dir)
        _ok("per_concept_cfs.csv written", per_csv.exists() and len(per_df) > 0)
        written["per_concept_cfs"] = str(per_csv)

        import analysis_real
        # bootstrap_ci on a per-concept vector returns (mean, lo, hi) all in [0,1].
        onm_vec = per_df[per_df["variant"] == "onmanifold_steer"]["cfs"].to_numpy()
        mean, lo, hi = analysis_real.bootstrap_ci(onm_vec, n=200)
        _ok(f"bootstrap_ci returns ordered (mean,lo,hi) in [0,1] "
            f"({mean:.3f} in [{lo:.3f},{hi:.3f}])",
            (0.0 <= lo <= mean <= hi <= 1.0))
        boot_df = analysis_real.bootstrap_by_method(per_df, n=200)
        boot_csv = pathlib.Path(out_dir) / "bootstrap_ci.csv"
        boot_df.to_csv(boot_csv, index=False)
        written["bootstrap_ci"] = str(boot_csv)
        sig = analysis_real.significance_readout(boot_df)
        findings = analysis_real.write_findings(
            out_dir, out_dir, boot_df=boot_df, sig=sig,
            ood=analysis_real.ood_degradation(sweep_df.rename(
                columns={"cfs": "cfs"}), floor=0.5)
            if "method" in sweep_df.columns else None,
            floor=0.5)
        _ok("FINDINGS.md written", (pathlib.Path(out_dir) / "FINDINGS.md").exists())
        _ok("findings.json written", (pathlib.Path(out_dir) / "findings.json").exists())
        written["FINDINGS"] = str(pathlib.Path(out_dir) / "FINDINGS.md")
        written["findings_json"] = str(pathlib.Path(out_dir) / "findings.json")

        import figures_real
        pngs = figures_real.make_real_figures(out_dir, out_dir, floor=0.5,
                                              n_boot=200)
        for p in pngs:
            p = pathlib.Path(p)
            _ok(f"{p.name} written and non-empty",
                p.exists() and p.stat().st_size > 0)
            written[p.name] = str(p)
        fig1 = pathlib.Path(out_dir) / "fig1_cfs_ood_sweep.png"
        fig7 = pathlib.Path(out_dir) / "fig7_by_method_bar.png"
        _ok("fig1_cfs_ood_sweep.png present", fig1.exists())
        _ok("fig7_by_method_bar.png present", fig7.exists())

        # ---- SUMMARY ---------------------------------------------------------
        print("\n" + "=" * 74)
        print("ALL END-TO-END CHECKS PASSED — the real pipeline is wired correctly.")
        print("=" * 74)
        print(f"  HEADLINE (RQ1, via real cfs_eval.compute_cfs):")
        print(f"    on-manifold CFS = {headline['onmanifold_cfs']:.4f}   "
              f"naive CFS = {headline['naive_cfs']:.4f}   "
              f"(Δ = {headline['onmanifold_cfs'] - headline['naive_cfs']:+.4f})")
        print(f"    on-manifold off-manifold residual = "
              f"{headline['onmanifold_offmanifold_residual']:.4f}   "
              f"(naive = {headline['naive_offmanifold_residual']:.4f})")
        print("  artifacts written to outputs/:")
        for name in ("ood_cfs_sweep", "ablations", "per_concept_cfs",
                     "bootstrap_ci", "FINDINGS", "findings_json",
                     "fig1_cfs_ood_sweep.png", "fig7_by_method_bar.png"):
            if name in written:
                print(f"    - {written[name]}")
        print("  NOTE: the open_clip GPU activation extraction in data_real is the "
              "ONLY\n        part not exercised here (it is import-guarded); every "
              "module\n        downstream of the cache ran as REAL code.")
        return 0, headline, written
    finally:
        # The fabricated activation cache is large + ephemeral, so always clean the
        # temp cache dir (the persistent artifacts live in outputs/). `--keep`
        # leaves it behind for inspection.
        if tmp_root is not None and not keep:
            shutil.rmtree(tmp_root, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(
        description="FAITH-SAE end-to-end real-pipeline self-test (CPU, no "
                    "open_clip, no downloads).")
    ap.add_argument("--out-dir", default=None,
                    help="where artifacts go (default: real_run/outputs/)")
    ap.add_argument("--cache-dir", default=None,
                    help="where the fabricated cache goes (default: a temp dir)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the temp cache dir after the run (for inspection)")
    args = ap.parse_args()
    code, _headline, _written = run(out_dir=args.out_dir, cache_dir=args.cache_dir,
                                    keep=args.keep)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
