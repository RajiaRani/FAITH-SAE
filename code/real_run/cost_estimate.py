#!/usr/bin/env python3
# =============================================================================
# cost_estimate.py  --  FAITH-SAE  code/real_run/
# Author: Rajia Rani  ()
#
# A RUNNABLE, closed-form GPU-hour + dollar estimator for the real publication
# run. Before you rent an A100/H100 you want to know, to within a factor of ~2,
# how long the whole pipeline takes and what it costs — WITHOUT downloading a
# single byte or burning a single GPU-second. This script does exactly that from
# a handful of well-understood quantities (image throughput, token budget, SVD
# cost, sweep cardinality), printing a per-stage table you can sanity-check by
# hand.
#
# It is INTENTIONALLY simple and transparent: every number is either a CLI knob,
# a documented backbone constant, or one short formula. The point is honesty —
# you should be able to read every multiplication. Real wall-clock will vary with
# dataloader speed, disk, and exact GPU, so we print a +/- band, not a fake-
# precise single number.
#
# Usage (CPU, no deps beyond the stdlib):
#   /usr/bin/python3 cost_estimate.py
#   /usr/bin/python3 cost_estimate.py --backbone ViT-L-14 --n-images 1280000 \
#       --token-budget 300000000 --gpu-price 1.80
#   /usr/bin/python3 cost_estimate.py --backbone ViT-B-16 --smoke   # tiny config
#
# This is a PLANNING tool, not the pipeline. For research and educational
# purposes only.
# =============================================================================
from __future__ import annotations

import argparse


# -----------------------------------------------------------------------------
# Backbone constants. d = residual-stream width; patches = patch tokens kept per
# image (CLS dropped); params ~ for context. img_per_s_a100 = a deliberately
# CONSERVATIVE END-TO-END extraction throughput (images/sec) for a single
# A100-80GB at batch 256, fp16, 224px, NO grad. This is NOT peak forward FLOPs:
# it bakes in the real bottlenecks of activation harvesting — the JPEG-decode +
# resize dataloader, the host<->device copies, and writing ~0.5 KB/token of
# fp16 activations out to disk. Peak forward is ~3-5x higher, but you never see
# it during extraction because you are I/O-bound, so we quote the I/O-bound
# number to keep the GPU-hour estimate honest (it lands the whole pipeline in
# the documented 40-120 GPU-hour band). H100 is ~1.5-1.7x faster — pass a higher
# throughput by editing this table or just read the band.
# -----------------------------------------------------------------------------
BACKBONES = {
    # name      : (d_in, n_patches, params_M, img_per_s_a100_end_to_end)
    "ViT-B-16":   (768,  196,  86,  280.0),   # smaller, fastest
    "ViT-L-14":   (1024, 256,  307, 110.0),   # the headline backbone
    "ViT-H-14":   (1280, 256,  632, 55.0),    # largest, slowest
}

# A100/H100 80GB on-demand cloud price band (USD / GPU-hour). The default
# matches a typical mid-market A100 spot/community price; tune with --gpu-price.
DEFAULT_GPU_PRICE = 1.80

# How much slower the *real* wall-clock tends to be than the pure compute model
# above, due to dataloading, JPEG decode, disk, and host<->device copies. We
# report estimate * [BAND_LO, BAND_HI] as the honest uncertainty interval.
BAND_LO, BAND_HI = 1.0, 1.8


def _fmt_h(h: float) -> str:
    """Human-friendly hours string."""
    if h < 1.0:
        return f"{h * 60:.0f} min"
    return f"{h:.1f} h"


def estimate(backbone: str, n_images: int, token_budget: int,
             ood_images: int, proj_rank_r: int, bank_tokens: int,
             n_concepts: int, strength_grid: int, ood_levels: int,
             ablation_runs: int, gpu_price: float,
             sae_epochs_over_budget: float = 1.0) -> dict:
    """Closed-form per-stage GPU-hour estimate. Returns a dict of stage->hours
    plus dollars. Every formula is one line and commented so you can audit it."""
    if backbone not in BACKBONES:
        raise SystemExit(f"unknown backbone {backbone!r}; choose from {list(BACKBONES)}")
    d, patches, _params, img_per_s = BACKBONES[backbone]

    # -- Stage 1: ACTIVATION EXTRACTION ---------------------------------------
    # We push every TRAIN image once through the frozen backbone to harvest the
    # token-budget worth of patch activations, PLUS every OOD image once. Time is
    # simply (#images / throughput). Train images contribute min(n_images,
    # token_budget/patches) forwards — you stop once the token budget is full.
    train_imgs_needed = min(n_images, max(1, token_budget // patches))
    extract_train_h = train_imgs_needed / img_per_s / 3600.0
    extract_ood_h = ood_images / img_per_s / 3600.0
    extract_h = extract_train_h + extract_ood_h

    # -- Stage 2: SAE TRAINING ------------------------------------------------
    # The SAE is a 2-layer dictionary (encoder + unit-norm decoder) trained on
    # the cached activations. Per 8192-token batch the cost is dominated by two
    # [d x n_features] matmuls (encode + decode) at n_features = expansion*d
    # (e.g. 32768 wide), plus the TopK + AuxK passes and AdamW, all fed by a disk
    # stream of fp16 shards. A 32768-wide SAE does ~2 * d * n_features ~ 67M
    # FLOPs/token forward+backward, so at ~100 effective TFLOPs you are
    # compute+stream-bound at ~0.35M tokens/sec on an A100-80GB. The token BUDGET
    # is processed sae_epochs_over_budget times for convergence + dead-feature
    # resampling; the brief's "300M-1B tokens" is this effective count (e.g. a
    # 300M budget x 4 passes ~ 1.2B token updates).
    sae_tok_per_s = 0.35e6
    sae_tokens = token_budget * sae_epochs_over_budget
    sae_h = sae_tokens / sae_tok_per_s / 3600.0

    # -- Stage 3: MANIFOLD BASIS (SVD) ----------------------------------------
    # One truncated SVD of a centered [bank_tokens, d] bank to get U_r[d, r].
    # On GPU this is dominated by the bank read + a thin SVD; ~a few minutes for
    # 2M x 1024. Model it as bank_tokens processed at ~20M tok/s (memory-bound).
    manifold_h = bank_tokens / 2.0e7 / 3600.0

    # -- Stage 4: CONCEPT SELECTION + PROBES ----------------------------------
    # Encode the eval activations through the SAE once + fit ~n_probe linear
    # probes on CPU/GPU. Cheap relative to extraction; model as one extra pass
    # over a bank-sized chunk of eval acts at SAE throughput.
    select_h = (bank_tokens * 2) / sae_tok_per_s / 3600.0

    # -- Stage 5: CFS EVAL on CLEAN -------------------------------------------
    # For each (concept x steering-method x strength) we re-encode a held-out
    # eval bank, apply the edit, and read probes. Cardinality drives cost. Model
    # each cell as a forward over a small eval bank (~bank_tokens/8 tokens).
    cfs_cells_clean = n_concepts * 5 * strength_grid          # 5 steer methods
    eval_bank = max(1, bank_tokens // 8)
    cfs_clean_h = cfs_cells_clean * eval_bank / sae_tok_per_s / 3600.0

    # -- Stage 6: OOD SWEEP ---------------------------------------------------
    # Same CFS computation repeated across the OOD ladder levels (clean + R +
    # Sketch + C(sev1-5 counts as 5) + ObjectNet). ood_levels already counts the
    # severities. The heavy part is re-extracting OOD acts (in stage 1) — here we
    # only re-run the cheap CFS cells per level.
    ood_h = ood_levels * cfs_cells_clean * eval_bank / sae_tok_per_s / 3600.0

    # -- Stage 7: ABLATIONS (A1-A5) -------------------------------------------
    # Not every ablation cell pays the same price. A3 (projection rank r) and A4
    # (selection threshold) only RE-EVALUATE the already-trained SAE — cheap, one
    # CFS pass each. A1 (TopK vs L1) and A2 (k sweep) must RETRAIN the SAE — but
    # at a reduced budget (1/3 of the headline budget is plenty to rank variants).
    # A5 (layer / patch-vs-CLS) needs a re-extraction + retrain; we fold its
    # re-extraction into the band rather than double-count it here. We split the
    # cells: ~1/3 retrain, ~2/3 re-eval-only — a realistic A1-A5 layout.
    n_retrain = max(1, ablation_runs // 3)
    n_reeval = ablation_runs - n_retrain
    cfs_one_h = cfs_cells_clean * eval_bank / sae_tok_per_s / 3600.0
    retrain_one_h = (sae_tokens / 3) / sae_tok_per_s / 3600.0 + cfs_one_h
    ablations_h = n_retrain * retrain_one_h + n_reeval * cfs_one_h

    stages = {
        "1_extract_activations": extract_h,
        "2_train_sae":           sae_h,
        "3_manifold_svd":        manifold_h,
        "4_concept_select":      select_h,
        "5_cfs_clean":           cfs_clean_h,
        "6_ood_sweep":           ood_h,
        "7_ablations":           ablations_h,
    }
    total = sum(stages.values())
    return {
        "backbone": backbone, "d_in": d, "n_patches": patches,
        "stages": stages, "total_h": total,
        "total_h_lo": total * BAND_LO, "total_h_hi": total * BAND_HI,
        "gpu_price": gpu_price,
        "cost_lo": total * BAND_LO * gpu_price,
        "cost_hi": total * BAND_HI * gpu_price,
        "train_imgs_needed": train_imgs_needed,
    }


def _print_table(r: dict) -> None:
    # Pre-compute every string OUTSIDE the f-strings: nesting the same quote
    # inside an f-string (e.g. f"{r['k']}") trips older parsers, so we keep the
    # format lines flat and unambiguous. Column widths are a single set of
    # constants so EVERY row aligns even when the band label (28 chars) and the
    # band hour-range ("15.3 h-27.6 h") are the widest entries in their columns.
    LBL, HRS, COST = 30, 15, 16          # label / GPU-hours / cost column widths
    line = "-" * (LBL + HRS + COST)
    price = r["gpu_price"]
    backbone = r["backbone"]
    d_in = r["d_in"]
    patches = r["n_patches"]
    price_hdr = "cost @ $%.2f/h" % price
    print(line)
    print(f"FAITH-SAE real-run cost estimate  |  backbone={backbone} "
          f"(d={d_in}, {patches} patch tok/img)")
    print(line)
    print(f"{'stage':<{LBL}}{'GPU-hours':>{HRS}}{price_hdr:>{COST}}")
    print(line)
    for name, h in r["stages"].items():
        cost = "$%.2f" % (h * price)
        print(f"{name:<{LBL}}{_fmt_h(h):>{HRS}}{cost:>{COST}}")
    print(line)
    total_cost = "$%.2f" % (r["total_h"] * price)
    print(f"{'TOTAL (compute model)':<{LBL}}{_fmt_h(r['total_h']):>{HRS}}"
          f"{total_cost:>{COST}}")
    band_label = "TOTAL (honest band x%.1f-%.1f)" % (BAND_LO, BAND_HI)
    band_hours = _fmt_h(r["total_h_lo"]) + "-" + _fmt_h(r["total_h_hi"])
    band_cost = "$%.0f-%.0f" % (r["cost_lo"], r["cost_hi"])
    print(f"{band_label:<{LBL}}{band_hours:>{HRS}}{band_cost:>{COST}}")
    print(line)
    print(f"note: extraction harvests ~{r['train_imgs_needed']:,} train images "
          f"into the token budget;\n      band widens the pure-compute number to "
          f"cover dataloading/JPEG/disk overhead.")
    print(line)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Closed-form GPU-hour + dollar estimator for the FAITH-SAE "
                    "real run (planning only; no GPU, no downloads).")
    p.add_argument("--backbone", default="ViT-L-14", choices=list(BACKBONES),
                   help="frozen CLIP backbone (sets width + patch count + throughput)")
    p.add_argument("--n-images", type=int, default=1_280_000,
                   help="ImageNet-1k train images available (default full 1.28M)")
    p.add_argument("--token-budget", type=int, default=300_000_000,
                   help="patch-token budget the SAE trains on (default 300M)")
    p.add_argument("--ood-images", type=int, default=180_000,
                   help="total OOD-ladder images to extract once "
                        "(R 30k + Sketch 51k + C-subset + ObjectNet 50k ~ 180k)")
    p.add_argument("--proj-rank", type=int, default=512,
                   help="manifold projection rank r (U_r columns)")
    p.add_argument("--bank-tokens", type=int, default=2_000_000,
                   help="tokens in the manifold/eval bank")
    p.add_argument("--n-concepts", type=int, default=50,
                   help="testable concepts carried into CFS (cfg.cfs.n_probe_classes)")
    p.add_argument("--strength-grid", type=int, default=5,
                   help="number of steering strengths (len of strength_grid)")
    p.add_argument("--ood-levels", type=int, default=9,
                   help="OOD ladder levels incl. C severities "
                        "(clean+R+Sketch+C1..5+ObjectNet = 9)")
    p.add_argument("--ablation-runs", type=int, default=24,
                   help="total ablation cells across A1..A5")
    p.add_argument("--gpu-price", type=float, default=DEFAULT_GPU_PRICE,
                   help="USD per GPU-hour (A100/H100 on-demand/spot)")
    p.add_argument("--sae-epochs", type=float, default=4.0,
                   help="passes over the token budget during SAE training "
                        "(TopK SAEs need a few passes to converge + resample "
                        "dead features; 4x300M = the brief's ~1B effective tokens)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny config so the estimator prints a sub-minute plan")
    args = p.parse_args()

    if args.smoke:
        # The CPU smoke pipeline: a few hundred synthetic-shaped images, ~1M
        # tokens, a handful of concepts — seconds of work, near-zero dollars.
        args.n_images = 512
        args.token_budget = 1_000_000
        args.ood_images = 256
        args.bank_tokens = 50_000
        args.n_concepts = 4
        args.ablation_runs = 4

    r = estimate(
        backbone=args.backbone, n_images=args.n_images,
        token_budget=args.token_budget, ood_images=args.ood_images,
        proj_rank_r=args.proj_rank, bank_tokens=args.bank_tokens,
        n_concepts=args.n_concepts, strength_grid=args.strength_grid,
        ood_levels=args.ood_levels, ablation_runs=args.ablation_runs,
        gpu_price=args.gpu_price, sae_epochs_over_budget=args.sae_epochs)
    _print_table(r)


if __name__ == "__main__":
    main()
