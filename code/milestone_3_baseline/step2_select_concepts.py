#!/usr/bin/env python3
# ===========================================================================
#  step2_select_concepts.py  —  Milestone 3 (Baseline), Part A (cont.)
#  Look at the SAE's learned concept directions and SELECT a few "testable"
#  ones by an interpretability/cleanliness score.
#  FAITH-SAE  ·  author: Rajia Rani  ·  educational use only
# ===========================================================================
#
#  ============ WHY THIS STEP EXISTS ============
#  The trained SAE gives us `sae_dim` (256) candidate concept features. But the
#  field's finding (and this paper's RQ2) is that only ~10-15% of SAE features
#  steer RELIABLY — most are polysemantic (encode several things) or junk (never
#  really activate). So before we spend effort steering, we SELECT a handful of
#  clean, well-defined concepts. This step computes a "cleanliness score" per
#  feature and keeps the top `n_select`.
#
#  ============ TERMS, FROM ZERO ============
#  CONCEPT DIRECTION (recap)
#    Decoder column j of the SAE: a length-`dim` vector d_j that feature j paints
#    into the activation when it fires. We will later steer by pushing along d_j.
#
#  CLEANLINESS / INTERPRETABILITY SCORE (what we invent here)
#    A single number per feature that is HIGH when the feature behaves like a
#    crisp, nameable concept. We combine three easy, offline signals:
#      (a) USAGE  — does the feature actually fire on a fair share of inputs?
#                   A feature that never fires (a "dead" feature) is useless.
#                   Example: fires on 12% of vectors -> usage signal ~ good;
#                   fires on 0% -> dead -> score 0.
#      (b) DECISIVENESS — when it fires, is it usually one of the FEW strong,
#                   confidently-on features (not a weak background flicker)?
#                   We measure the average rank/strength of the feature when it
#                   is active. Strong, decisive firing -> monosemantic-looking.
#      (c) ALIGNMENT — does the feature's direction line up with one of the
#                   PLANTED ground-truth concepts? (Offline only: we planted the
#                   answer in step 1, so we can reward features that found it.)
#                   Cosine similarity 1.0 = perfectly aligned; 0 = unrelated.
#    cleanliness = usage_signal * decisiveness * alignment  (all in [0,1], so a
#    zero on any axis kills the score — a clean concept needs all three).
#
#  COSINE SIMILARITY (used in alignment)
#    "How parallel are two arrows?" = (u . v) / (|u| |v|), in [-1, 1]. 1 = same
#    direction, 0 = perpendicular, -1 = opposite. Example: u=[1,0], v=[1,1] ->
#    cos = 1/sqrt(2) ~= 0.71.
#
#  ============ WHAT THIS SCRIPT DOES ============
#    1. Load the SAE checkpoint from step 1.
#    2. Re-make the same activation bank, encode it, gather per-feature stats.
#    3. Score every feature's cleanliness; keep the top `n_select` above threshold.
#    4. Write outputs/selected_concepts.csv (the concepts step 3 will steer).
#  ========================================================================

from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.model import _build  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402

# Reuse the EXACT bank generator from step 1 so the data matches the checkpoint.
from step1_train_sae import make_activation_bank  # noqa: E402


def load_sae(cfg: dict):
    """Rebuild the TopKSAE and load the trained weights from the checkpoint."""
    import torch

    ckpt_path = os.path.join(HERE, cfg["sae_ckpt"])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run step1_train_sae.py first.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    FaithSAE = _build()
    model = FaithSAE(cfg)
    model.sae.load_state_dict(ckpt["state_dict"])
    return model.sae, ckpt["planted_concepts"]


def score_concepts(cfg: dict):
    import torch

    set_seed(cfg["seed"])
    sae, planted = load_sae(cfg)

    # 1) Re-make the bank and encode it to sparse feature codes z.
    bank, _ = make_activation_bank(cfg)                    # [N, P, dim]
    flat = bank.reshape(-1, cfg["dim"])                    # [N*P, dim] vectors
    with torch.no_grad():
        z = sae.encode(flat)                               # [N*P, sae_dim] codes
    n_vec, n_feat = z.shape

    # 2) (a) USAGE: fraction of vectors where the feature fires (nonzero).
    fire_rate = (z != 0).float().mean(0)                   # [sae_dim] in [0,1]
    # Reward a healthy, not-too-rare, not-everywhere firing rate. We map fire_rate
    # to a 0..1 "usage signal" peaked around a sensible band (~3%-40%).
    usage = torch.clamp(fire_rate / 0.20, 0.0, 1.0)        # ramps up to 1 by 20%
    usage = usage * torch.clamp((0.80 - fire_rate) / 0.80, 0.0, 1.0)  # penalise "always on"

    # 2) (b) DECISIVENESS: when active, how strong is the feature relative to the
    #    typical active value? High, confident magnitudes => monosemantic-looking.
    active_mask = (z != 0).float()
    sum_active = (z.abs() * active_mask).sum(0)
    cnt_active = active_mask.sum(0) + 1e-8
    mean_active_mag = sum_active / cnt_active               # [sae_dim]
    # Normalise across features to [0,1] (relative decisiveness).
    decisive = mean_active_mag / (mean_active_mag.max() + 1e-8)

    # 2) (c) ALIGNMENT: each decoder column vs the BEST-matching planted concept.
    dec = sae.dec.weight.detach()                          # [dim, sae_dim]
    dec_unit = dec / (dec.norm(dim=0, keepdim=True) + 1e-8) # unit columns
    planted_unit = planted / (planted.norm(dim=1, keepdim=True) + 1e-8)  # [C, dim]
    cos = (planted_unit @ dec_unit).abs()                  # [C, sae_dim] |cos|
    alignment, best_concept = cos.max(0)                   # best planted match

    # 3) Combine: cleanliness = usage * decisive * alignment (conjunctive).
    cleanliness = usage * decisive * alignment             # [sae_dim] in [0,1]

    # 4) Rank features, keep top n_select that clear the threshold.
    order = torch.argsort(cleanliness, descending=True)
    thresh = cfg.get("concept_select_thresh", 0.0)
    selected = []
    for idx in order.tolist():
        if cleanliness[idx].item() < thresh:
            break
        selected.append(idx)
        if len(selected) >= cfg["n_select"]:
            break

    rows = []
    for rank, j in enumerate(selected):
        rows.append({
            "rank": rank,
            "feature_id": int(j),
            "cleanliness": round(float(cleanliness[j]), 4),
            "fire_rate": round(float(fire_rate[j]), 4),
            "decisiveness": round(float(decisive[j]), 4),
            "alignment": round(float(alignment[j]), 4),
            "matched_planted_concept": int(best_concept[j]),
        })

    # Report context: how many of all features cleared a "well-defined" bar — the
    # field's ~10-15% reliable-tail figure (RQ2).
    well_defined = int((cleanliness > 0.30).sum())
    print(f"[select] {well_defined}/{n_feat} features look well-defined "
          f"(cleanliness > 0.30)  =  {100*well_defined/n_feat:.1f}%  "
          f"(the field's '~10-15% steer reliably' tail).")
    print(f"[select] keeping top {len(rows)} testable concepts "
          f"(threshold = {thresh}):")
    for r in rows:
        print(f"    feature {r['feature_id']:3d}  cleanliness={r['cleanliness']:.3f}  "
              f"fire={r['fire_rate']:.2f}  align={r['alignment']:.2f}  "
              f"~planted#{r['matched_planted_concept']}")

    out = os.path.join(HERE, cfg["concepts_csv"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[save] selected concepts -> {out}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Select clean, testable SAE concepts.")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    score_concepts(cfg)
    print("\nDONE step 2. Now run: /usr/bin/python3 step3_naive_steer_cfs.py")


if __name__ == "__main__":
    main()
