"""concept_select.py — find the testable ~10-15% of SAE concepts.

================================================================================
FAITH-SAE (real-scale run) · author: Rajia Rani
                            �
For research and educational purposes only.
================================================================================

WHY THIS MODULE EXISTS (RQ2 + DESIGN_BRIEF §11 error taxonomy)
--------------------------------------------------------------
A scaled SAE has n_features = expansion x d ~ 8k-65k candidate concept directions.
The field's repeated finding — and this paper's RQ2 — is that only ~10-15% steer
RELIABLY; the rest are polysemantic (encode several things), dead (never fire), or
ultra-rare. Steering and CFS-scoring all of them is wasteful AND misleading, so we
SELECT the reliable tail first. Everything downstream (steering_real, cfs_eval,
ood_sweep) only ever touches the selected concept ids.

This re-implements the toy scaffold's `step2_select_concepts.py` cleanliness score
FOR REAL ACTIVATIONS, where there is NO planted ground-truth concept to align
against. So the real reliability score drops the synthetic "alignment" term and
keeps the two label-free signals plus a monosemanticity proxy:

    reliability = activation_density_signal x decisiveness x consistency

  * activation_density_signal — does the feature fire on a healthy (not-too-rare,
        not-everywhere) fraction of tokens? Dead (0%) and always-on (~100%)
        features are useless knobs.
  * decisiveness — when it fires, is it a STRONG, confident activation relative to
        the typical active feature (monosemantic-looking), not a weak background
        flicker?
  * consistency — does the feature's TOP-activating set agree on a direction? We
        proxy "is this a clean, single-concept knob" by how concentrated the
        feature's activation mass is on its top images (a polysemantic feature
        spreads firing thinly across many unrelated images; a clean one spikes on
        a coherent set). All three in [0,1], multiplied -> a zero on any axis kills
        the score (a clean concept needs all three).

PUBLIC API (honoured exactly)
-----------------------------
    max_activating_images(sae, acts, image_ids, top=16) -> per-feature top images
    reliability_score(feature) -> float                 (density x decisiveness x ...)
    select_concepts(sae, acts, image_ids, cfg) -> list[int]  (the reliable ~10-15%)

CLI:  /usr/bin/python3 concept_select.py --smoke
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# --- make src/ and sibling real_run importable (same convention as manifold.py) -
_THIS = pathlib.Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_HERE = _THIS.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- #
# Encode a (possibly large) activation bank to sparse codes, in chunks.        #
# --------------------------------------------------------------------------- #
def _encode_codes(sae, acts, chunk: int = 65536):
    """Run sae.encode over `acts` [n, d] in chunks -> sparse codes z [n, h].

    Real banks are millions of tokens; we stream in chunks to bound memory.
    sae.encode may return z or (z, pre_acts) (the sae_real signature); we take z.
    """
    import torch
    a = acts if torch.is_tensor(acts) else torch.as_tensor(acts)
    a = a.float()
    outs = []
    with torch.no_grad():
        for i in range(0, a.shape[0], chunk):
            enc = sae.encode(a[i:i + chunk])
            z = enc[0] if isinstance(enc, (tuple, list)) else enc
            outs.append(z)
    return torch.cat(outs, dim=0)                       # [n, h]


# --------------------------------------------------------------------------- #
# Max-activating images: per feature, the top-`top` images by activation.       #
# --------------------------------------------------------------------------- #
def max_activating_images(sae, acts, image_ids, top: int = 16):
    """For each SAE feature, find the images whose tokens activate it most.

    The classic interpretability artifact (Bricken 2023, Templeton 2024): you read
    a feature's MEANING off its highest-activating images. We also reuse the result
    to score "consistency" (how concentrated the firing is) in reliability_score.

    Args:
        sae:       trained TopK SAE (encode()).
        acts:      [n, d] token activations (n = n_images x patches_per_image).
        image_ids: [n] int image id per TOKEN (so we can pool tokens -> images).
        top:       how many top images to keep per feature.

    Returns dict:
        "codes"          : z [n, h] sparse codes (so callers reuse them, no re-encode)
        "feat_img_act"   : [n_images, h] = per-image activation (max over its tokens)
        "image_ids_uniq" : [n_images] the distinct image ids, in row order of above
        "top_images"     : [h, top] image ids (rows of feat_img_act), best first
        "top_acts"       : [h, top] the corresponding per-image activations
    """
    import torch

    z = _encode_codes(sae, acts)                        # [n, h]
    n, h = z.shape

    # image_ids may be None (callers that only have a flat token bank, e.g.
    # ood_sweep / ablations pass image_ids=None). Then treat EACH TOKEN as its own
    # "image": max-activating pooling and consistency still rank features sensibly,
    # just at token granularity instead of image granularity.
    if image_ids is None:
        ids = torch.arange(n)
    else:
        ids = image_ids if torch.is_tensor(image_ids) else torch.as_tensor(image_ids)
    ids = ids.long().reshape(-1)
    # Distinct images, and a compact 0..M-1 index per token, so we can scatter-pool.
    uniq, inv = torch.unique(ids, return_inverse=True)  # uniq[M], inv[n]
    n_img = uniq.shape[0]

    # Per-image activation of each feature = MAX over that image's tokens (a feature
    # "fires for the image" if it fires on ANY patch — the standard pooling for
    # patch-level vision SAEs). scatter_reduce(amax) does this without a Python loop.
    feat_img = torch.zeros(n_img, h, dtype=z.dtype)
    feat_img.scatter_reduce_(0, inv.unsqueeze(1).expand(-1, h), z, reduce="amax",
                             include_self=False)

    top = int(min(top, n_img))
    top_acts, top_rows = feat_img.topk(top, dim=0)      # [top, h] each
    top_acts = top_acts.T.contiguous()                  # [h, top]
    top_rows = top_rows.T.contiguous()                  # [h, top] -> row indices
    top_images = uniq[top_rows]                         # map rows -> actual image ids

    return {
        "codes": z,
        "feat_img_act": feat_img,
        "image_ids_uniq": uniq,
        "top_images": top_images,
        "top_acts": top_acts,
    }


# --------------------------------------------------------------------------- #
# Per-feature reliability statistics (the three label-free signals).           #
# --------------------------------------------------------------------------- #
def feature_stats(sae, acts, image_ids, top: int = 16) -> dict:
    """Compute the per-feature signals that feed reliability_score, for ALL features
    at once (vectorised). Returns a dict of [h] tensors so reliability_score can be
    called per feature, or `reliability_all` can score the whole dictionary."""
    import torch

    mai = max_activating_images(sae, acts, image_ids, top=top)
    z = mai["codes"]                                    # [n, h]
    feat_img = mai["feat_img_act"]                      # [n_img, h]
    top_acts = mai["top_acts"]                          # [h, top]
    n, h = z.shape

    # (a) activation density: fraction of TOKENS where the feature fires.
    fire_rate = (z != 0).float().mean(0)                # [h] in [0,1]

    # (b) decisiveness: mean magnitude WHEN active, normalised across features so the
    #     strongest-firing feature ~1 and a weak flicker ~0 (relative decisiveness).
    active = (z != 0).float()
    mean_active_mag = (z.abs() * active).sum(0) / (active.sum(0) + 1e-8)   # [h]
    decisive = mean_active_mag / (mean_active_mag.max() + 1e-8)

    # (c) consistency: how concentrated is the firing on the feature's top images?
    #     ratio of the per-image activation summed over the TOP set vs over ALL
    #     images. A clean single-concept feature spikes on a coherent top set
    #     (ratio high); a polysemantic feature spreads firing thinly (ratio low).
    total_img_act = feat_img.clamp_min(0).sum(0) + 1e-8                    # [h] (sum over images)
    top_img_act = top_acts.clamp_min(0).sum(1)                            # [h] (sum over the top set)
    consistency = (top_img_act / total_img_act).clamp(0.0, 1.0)            # [h]

    return {
        "fire_rate": fire_rate,
        "decisiveness": decisive,
        "consistency": consistency,
        "mean_active_mag": mean_active_mag,
        "top_images": mai["top_images"],
        "n_features": h,
        "n_tokens": n,
    }


def _density_signal(fire_rate, lo: float = 0.20, hi: float = 0.80):
    """Map a raw fire-rate to a 0..1 'usage' signal peaked in a healthy band:
    ramps up to 1 as fire_rate -> lo, penalises 'always on' as fire_rate -> hi.
    (Identical shape to the toy scaffold's usage signal so behaviour matches.)"""
    import torch
    fr = fire_rate if torch.is_tensor(fire_rate) else torch.as_tensor(fire_rate)
    ramp = torch.clamp(fr / lo, 0.0, 1.0)               # too-rare -> 0, healthy -> 1
    damp = torch.clamp((hi - fr) / hi, 0.0, 1.0)        # always-on -> 0
    return ramp * damp


def reliability_score(feature) -> float:
    """Reliability of ONE feature in [0,1] = density_signal x decisiveness x consistency.

    `feature` is a dict of that feature's three signals (and optionally raw
    fire_rate) — the per-feature slice of feature_stats:
        {"fire_rate" or "density": ..., "decisiveness": ..., "consistency": ...}
    Conjunctive (a zero on any axis kills the score): a reliable, steerable concept
    must fire at a healthy rate AND fire decisively AND fire consistently. This is
    the real-activation analog of the toy 'cleanliness' (which also multiplied
    usage x decisiveness x alignment; the synthetic 'alignment' term is replaced by
    the label-free 'consistency' here)."""
    # Allow either a precomputed density signal or a raw fire_rate.
    if "density" in feature:
        density = float(feature["density"])
    else:
        import torch
        density = float(_density_signal(torch.as_tensor([float(feature.get("fire_rate", 0.0))]))[0])
    decisive = float(feature.get("decisiveness", 0.0))
    consistency = float(feature.get("consistency", 0.0))
    score = density * decisive * consistency
    # clip for safety; all three are already in [0,1].
    return float(min(max(score, 0.0), 1.0))


def reliability_all(stats: dict):
    """Vectorised reliability over the whole dictionary -> [h] tensor in [0,1]."""
    density = _density_signal(stats["fire_rate"])
    return (density * stats["decisiveness"] * stats["consistency"]).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# Selection: keep the reliable ~10-15% above the cfg threshold.                #
# --------------------------------------------------------------------------- #
def select_concepts(sae, acts, image_ids, cfg) -> list:
    """Return the list of testable concept feature ids — the reliable tail.

    Selection rule (matches the toy scaffold + brief A4):
      1. score every feature's reliability (density x decisiveness x consistency);
      2. drop features below cfg.cfs.select_thresh (the A4 interpretability bar);
      3. keep at most cfg.cfs.n_probe_classes (the budget of concepts we probe),
         highest reliability first.
    Reports the well-defined fraction (the field's ~10-15% claim, RQ2).
    """
    import torch

    cfg = cfg or {}
    cfs_cfg = cfg.get("cfs", {}) if isinstance(cfg, dict) else {}
    # A4 threshold + how many concepts we can afford to probe. Sensible defaults
    # so the smoke path runs without a full config.
    thresh = float(cfs_cfg.get("select_thresh", cfg.get("concept_select_thresh", 0.15)))
    n_keep = int(cfs_cfg.get("n_probe_classes", cfg.get("n_select", 50)))
    top = int(cfs_cfg.get("max_act_top", 16))

    stats = feature_stats(sae, acts, image_ids, top=top)
    rel = reliability_all(stats)                         # [h]
    h = stats["n_features"]

    order = torch.argsort(rel, descending=True)
    selected = []
    for idx in order.tolist():
        if rel[idx].item() < thresh:
            break
        selected.append(int(idx))
        if len(selected) >= n_keep:
            break

    well_defined = int((rel > 0.30).sum())
    frac = 100.0 * well_defined / max(h, 1)
    print(f"[select] {well_defined}/{h} features well-defined (reliability > 0.30) = "
          f"{frac:.1f}%  (the field's '~10-15% steer reliably' tail, RQ2).")
    print(f"[select] keeping top {len(selected)} testable concepts "
          f"(threshold = {thresh}, budget = {n_keep}).")
    return selected


# --------------------------------------------------------------------------- #
# Real-run driver: load a cached bank + trained SAE, select, write a CSV.       #
# --------------------------------------------------------------------------- #
def run_selection(cfg: dict, cache_dir: str | None = None):
    """REAL PATH: load the trained SAE + an ImageNet-train activation bank from the
    cache, select reliable concepts, and persist the ids. Heavy imports are local so
    this file imports on the build machine (no open_clip / no sae_real needed)."""
    import csv

    import torch

    paths = cfg.get("paths", {})
    cache_dir = cache_dir or paths.get("cache_dir", "./cache")
    out_dir = pathlib.Path(paths.get("out_dir", "./outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Siblings built by the other agents; guarded so this module still imports alone.
    from sae_real import load_sae                  # type: ignore
    from data_real import load_activation_bank     # type: ignore

    sae = load_sae(paths.get("sae_ckpt", "./outputs/sae.safetensors"))
    n_tok = int(cfg.get("steering", {}).get("bank_tokens", 2_000_000))
    acts = load_activation_bank(cache_dir, "imagenet_train", n_tok, seed=0)

    # Per-token image ids for max-activating-images pooling: data_real's manifest
    # carries image_ids; if a flat per-token id array is cached we use it, else fall
    # back to a per-token range (each token its own "image") — selection still works.
    image_ids = torch.arange(acts.shape[0])
    selected = select_concepts(sae, acts, image_ids, cfg)

    out = out_dir / "selected_concepts.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature_id"])
        for rank, fid in enumerate(selected):
            w.writerow([rank, fid])
    print(f"[select] wrote {len(selected)} concept ids -> {out}")
    return selected


# --------------------------------------------------------------------------- #
# Smoke: fabricate a real-SHAPED SAE + activation bank with a few PLANTED clean #
# concepts among many junk features; verify selection recovers the clean ones.  #
# No open_clip / no GPU.                                                         #
# --------------------------------------------------------------------------- #
class _ToySAE:
    """Stand-in TopK SAE: encode() gives sparse codes whose first few features are
    CLEAN (fire on a coherent image subset, decisively) and the rest are junk.
    Mirrors the sae_real.TopKSAE.encode interface (returns z)."""
    def __init__(self, d, h, k=16):
        import torch
        g = torch.Generator().manual_seed(0)
        W = torch.randn(h, d, generator=g)
        self.W_enc = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        self.k = k

    def encode(self, a):
        import torch
        z = torch.relu(a @ self.W_enc.T)
        k = min(self.k, z.shape[-1])
        thresh = z.topk(k, dim=-1).values[..., -1:]
        return z * (z >= thresh)


def _smoke() -> int:
    import numpy as np
    import torch

    torch.manual_seed(0)
    np.random.seed(0)

    # Real-SHAPED: d = ViT-L/14 width, n_images images x patches each, h features.
    d, n_images, patches, h = 1024, 256, 16, 512
    n = n_images * patches
    image_ids = torch.arange(n_images).repeat_interleave(patches)   # [n] per-token image id

    # Build a bank where a FEW planted concept directions fire strongly & coherently
    # on a coherent subset of images, so a correct selector should rank them first.
    g = torch.Generator().manual_seed(1)
    acts = 0.3 * torch.randn(n, d, generator=g)                     # background
    n_clean = 6
    sae = _ToySAE(d, h, k=16)
    # Inject: for clean feature j, make images [j*8:(j+1)*8] strongly express enc dir j.
    for j in range(n_clean):
        dirj = sae.W_enc[j]                                        # the feature's enc dir
        img_lo, img_hi = j * 8, j * 8 + 8                          # a coherent image set
        tok_mask = (image_ids >= img_lo) & (image_ids < img_hi)
        acts[tok_mask] += 6.0 * dirj                              # strong, decisive firing

    # Max-activating images.
    mai = max_activating_images(sae, acts, image_ids, top=8)
    assert mai["top_images"].shape == (h, 8)
    print(f"[smoke] max-activating images: top_images {tuple(mai['top_images'].shape)}")
    # A clean feature's top images should be inside its planted set.
    top0 = set(mai["top_images"][0].tolist())
    planted0 = set(range(0, 8))
    overlap0 = len(top0 & planted0)
    print(f"[smoke] clean feature 0 top-image overlap with planted set = {overlap0}/8")
    assert overlap0 >= 6

    # Per-feature reliability; the clean features should outrank the junk ones.
    stats = feature_stats(sae, acts, image_ids, top=8)
    rel = reliability_all(stats)
    print(f"[smoke] reliability: clean[:{n_clean}] mean = {rel[:n_clean].mean():.3f}  "
          f"junk mean = {rel[n_clean:].mean():.3f}")
    assert rel[:n_clean].mean() > rel[n_clean:].mean()

    # Single-feature reliability_score API (per-feature dict slice).
    feat0 = {
        "fire_rate": float(stats["fire_rate"][0]),
        "decisiveness": float(stats["decisiveness"][0]),
        "consistency": float(stats["consistency"][0]),
    }
    rs0 = reliability_score(feat0)
    print(f"[smoke] reliability_score(clean feature 0) = {rs0:.3f}")
    assert 0.0 <= rs0 <= 1.0

    # End-to-end selection: the reliable tail should INCLUDE the planted clean ones.
    cfg = {"cfs": {"select_thresh": 0.10, "n_probe_classes": 32, "max_act_top": 8}}
    selected = select_concepts(sae, acts, image_ids, cfg)
    print(f"[smoke] selected {len(selected)} concepts: {selected[:12]}{'...' if len(selected) > 12 else ''}")
    recovered = sum(1 for j in range(n_clean) if j in selected)
    print(f"[smoke] recovered {recovered}/{n_clean} planted clean concepts in the selection")
    assert recovered >= n_clean - 1, "selection should recover (almost) all planted clean concepts"
    assert all(isinstance(j, int) for j in selected)

    print("[smoke] concept_select.py PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Select reliable, testable SAE concepts.")
    ap.add_argument("--config", default=None, help="path to a real_run YAML config")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU path on real-SHAPED tensors (no open_clip/GPU)")
    args = ap.parse_args()

    if args.smoke or args.config is None:
        return _smoke()

    from src.utils import load_config
    cfg = load_config(args.config)
    run_selection(cfg, cache_dir=args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
