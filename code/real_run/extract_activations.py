#!/usr/bin/env /usr/bin/python3
# =============================================================================
# extract_activations.py  --  CLI entry: cache the frozen-backbone activations.
# Project: FAITH-SAE.  Author: Rajia Rani
#          ()
#
# WHAT IT DOES:
#   Runs the frozen ViT (a SUPERVISED timm ViT-S/B/L by default; open_clip CLIP
#   if backbone.framework: open_clip) over a dataset (in1k, or one of the OOD
#   rungs) and writes the SHARDED activation cache the rest of the pipeline reads:
#       cache_dir/acts_{ds}_{shard:05d}.npy    float16 [n_tokens, d_in]
#       cache_dir/labels_{ds}_{shard:05d}.npy  int64   [n_tokens]
#       cache_dir/manifest_{ds}.json
#
# USAGE (real, on a GPU box with timm + the datasets):
#   python3 extract_activations.py --config configs/vit_b_84m.yaml \
#           --dataset in1k --cache_dir ./cache
#   (repeat with --dataset in100 / food101 / cifar100)
#
# USAGE (offline smoke, this build box -- NO timm, NO GPU, NO download):
#   python3 extract_activations.py --config configs/smoke.yaml \
#           --dataset in1k --cache_dir ./cache_smoke --smoke
#   The smoke path FABRICATES a few synthetic-but-real-SHAPED shards (Gaussian
#   activations on a planted low-rank manifold, like src/data.synthetic_batch /
#   milestone_2's bank) so the WHOLE pipeline (train_sae -> manifold -> cfs ->
#   ood_sweep) is exercisable on CPU. Same on-disk format as the real path, so no
#   downstream module can tell the difference.
#
# DATASETS (the student's domain-shift ladder; sources + sizes):
#   in1k (ImageNet-1k train) in-dist + SAE-training source  ALREADY on the cluster
#   in100  100-class IN-1k subset (via in100_classes.txt)   derived from in1k
#   food101  101 food classes, ~5GB   torchvision.datasets.Food101(download=True)
#   cifar100 100 classes 32x32, ~170MB  torchvision.datasets.CIFAR100(download=True)
# -----------------------------------------------------------------------------
# For research and educational purposes only.
# =============================================================================
from __future__ import annotations

import argparse
import json
import os
import sys
import pathlib

import numpy as np

# Make this dir importable (sibling data_real) regardless of launch cwd.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from data_real import (                               # noqa: E402
    extract_activations,
    load_real_config,
    _shard_paths,
    _write_manifest,
)


# =============================================================================
# SMOKE PATH: fabricate synthetic-but-real-SHAPED shards (no open_clip / GPU).
# =============================================================================
def _fabricate_smoke_shards(cfg, dataset_name: str, cache_dir: str):
    """Write a handful of synthetic activation shards in the REAL cache format.

    The numbers are shaped exactly like CLIP patch-token activations would be:
      * a low-rank "real-image manifold" core (a random orthonormal basis of rank
        manifold_rank, each token a random combo of those directions) PLUS small
        full-rank noise -- so manifold.estimate_manifold_basis finds real
        structure and onmanifold projection is meaningful;
      * a few PLANTED concept directions added to a fraction of tokens -- so the
        SAE has recoverable features and the CFS probe has something real to steer;
      * an OOD-shift knob per non-clean dataset (a small mean shift + scale change)
        so the OOD sweep sees CFS actually degrade off-distribution.
    Mirrors src/data.py + milestone_2's synthetic bank, in the on-disk shard
    format, so every downstream real_run module runs unchanged on these shards.
    """
    os.makedirs(cache_dir, exist_ok=True)
    d_in = int(cfg.sae.d_in)
    smoke = cfg.get("smoke", {}) or {}
    n_shards = int(smoke.get("n_shards", 3))
    tokens_per_shard = int(smoke.get("tokens_per_shard", 4096))
    n_patches = int(smoke.get("n_patches", 256))
    manifold_rank = max(2, min(d_in // 2, 32))        # thin sheet inside d_in
    n_concepts = max(4, d_in // 16)
    n_classes = int(cfg.cfs.get("n_probe_classes", 5))

    # Seed PER DATASET so each OOD rung is reproducible but distinct.
    base_seed = int(cfg.get("seed", 0)) + (hash(dataset_name) % 9973)
    rng = np.random.default_rng(base_seed)

    # --- fixed structure shared across this dataset's shards ------------------
    # Orthonormal manifold basis B [d_in, manifold_rank] (QR of a Gaussian).
    B, _ = np.linalg.qr(rng.standard_normal((d_in, manifold_rank)))
    B = B[:, :manifold_rank]
    # Concept directions live INSIDE the manifold (real concepts are on-manifold).
    coeffs = rng.standard_normal((manifold_rank, n_concepts))
    concept_dirs = B @ coeffs                          # [d_in, n_concepts]
    concept_dirs /= (np.linalg.norm(concept_dirs, axis=0, keepdims=True) + 1e-8)

    # OOD shift: in-distribution = none; harder rungs = bigger off-manifold mean +
    # scale. New ladder (student's datasets) ordered by shift strength:
    #   in1k (in-dist) -> in100 (mild) -> food101 (domain) -> cifar100 (strong).
    # Legacy open_clip OOD names are kept so older smoke configs still fabricate.
    shift_level = {
        # student's ladder
        "in1k": 0.0, "in100": 0.4, "food101": 0.8, "cifar100": 1.2,
        # in-distribution aliases (SAE-training source)
        "clean": 0.0, "imagenet": 0.0, "imagenet_train": 0.0,
        # legacy OOD ladder (still supported)
        "imagenet_r": 0.6, "imagenet_sketch": 0.9,
        "imagenet_c": 0.7, "objectnet": 1.1,
    }.get(dataset_name, 0.4)
    ood_dir = rng.standard_normal(d_in)
    ood_dir /= (np.linalg.norm(ood_dir) + 1e-8)        # an OFF-manifold direction

    all_image_ids = []
    n_tokens_total = 0
    n_images_total = 0
    for s in range(n_shards):
        n_tok = tokens_per_shard
        n_imgs = max(1, n_tok // n_patches)
        n_tok = n_imgs * n_patches                     # keep tokens grouped by image

        # manifold core: random combos of the basis directions
        z = rng.standard_normal((n_tok, manifold_rank)).astype(np.float32)
        acts = (z @ B.T).astype(np.float32)            # [n_tok, d_in], on-manifold
        acts += 0.1 * rng.standard_normal((n_tok, d_in)).astype(np.float32)  # noise

        # plant concepts on a per-image basis (a class -> a present concept set)
        img_labels = rng.integers(0, n_classes, size=n_imgs).astype(np.int64)
        tok_labels = np.repeat(img_labels, n_patches)
        for i in range(n_imgs):
            c = img_labels[i] % n_concepts             # this class activates concept c
            sl = slice(i * n_patches, (i + 1) * n_patches)
            strength = 1.0 + 0.5 * rng.standard_normal()
            acts[sl] += strength * concept_dirs[:, c]

        # apply the OOD shift (off-manifold mean + a slight variance change)
        if shift_level > 0:
            acts += shift_level * ood_dir
            acts *= (1.0 + 0.15 * shift_level)

        ap, lp = _shard_paths(cache_dir, dataset_name, s)
        np.save(ap, acts.astype(np.float16))
        np.save(lp, tok_labels.astype(np.int64))

        all_image_ids.extend(
            [f"{dataset_name}/synthetic_{s:05d}_{i:06d}.jpg" for i in range(n_imgs)]
        )
        n_tokens_total += n_tok
        n_images_total += n_imgs

    manifest = _write_manifest(
        cache_dir, dataset_name, cfg,
        n_images=n_images_total, n_tokens=n_tokens_total,
        n_shards=n_shards, image_ids=all_image_ids,
    )
    return manifest


# =============================================================================
# CLI
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Extract & cache frozen-backbone patch activations (FAITH-SAE)."
    )
    ap.add_argument("--config", required=True, help="path to a real_run YAML config")
    ap.add_argument("--dataset", required=True,
                    help="in1k | in100 | food101 | cifar100")
    ap.add_argument("--cache_dir", default=None,
                    help="override cfg.paths.cache_dir for the output shards")
    ap.add_argument("--smoke", action="store_true",
                    help="fabricate tiny synthetic-real-shaped shards (no timm/GPU)")
    args = ap.parse_args(argv)

    cfg = load_real_config(args.config)
    cache_dir = args.cache_dir or cfg.paths.cache_dir

    if args.smoke:
        manifest = _fabricate_smoke_shards(cfg, args.dataset, cache_dir)
        mode = "SMOKE (synthetic shards)"
    else:
        # Real path: needs timm (or open_clip) + torchvision + the dataset + a GPU.
        manifest = extract_activations(cfg, args.dataset, cache_dir)
        mode = "REAL (timm/open_clip backbone)"

    print(f"[extract_activations] {mode}")
    print(f"  dataset={args.dataset}  cache_dir={cache_dir}")
    print("  manifest=" + json.dumps(
        {k: v for k, v in manifest.items() if k != "image_ids"}))
    print(f"  image_ids cached: {len(manifest['image_ids'])}")
    return manifest


if __name__ == "__main__":
    main()
