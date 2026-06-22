"""Evaluation: SAE reconstruction loss + the CFS faithfulness probe.

The CFS probe sweeps the steering knob and measures the three faithfulness
components empirically (Monotonicity, Specificity, Sufficiency), then combines
them with the analytic `cfs_score`. Real benchmarks (ImageNet-R/Sketch/C/
ObjectNet, the OOD CFS-vs-severity curve) swap in at M2-M3.
"""
from __future__ import annotations

import argparse

from .data import synthetic_batch
from .utils import cfs_score, get_logger

log = get_logger("evaluate")


def recon_loss(model, batches: int = 8, cfg: dict | None = None) -> float:
    import torch
    cfg = cfg or {"n_patches": 16, "dim": 64}
    losses = []
    with torch.no_grad():
        for i in range(batches):
            x, _ = synthetic_batch(16, cfg["n_patches"], cfg["dim"], seed=2000 + i)
            _, loss = model(x)
            losses.append(float(loss))
    return sum(losses) / len(losses)


def _spearman(a, b) -> float:
    """Spearman rho via Pearson on ranks (no scipy needed)."""
    import torch
    ar = a.argsort().argsort().float()
    br = b.argsort().argsort().float()
    ar = ar - ar.mean(); br = br - br.mean()
    denom = (ar.norm() * br.norm()) + 1e-8
    return float((ar * br).sum() / denom)


def cfs_probe(model, cfg: dict | None = None, concept: int = 0,
              n_steps: int = 6) -> dict:
    """Empirical Causal Faithfulness Score for the model's selected steerer.

    Monotonicity = Spearman(knob, target readout); Specificity = 1 - off-target
    drift; Sufficiency = standardized effect size at full knob. Combined via the
    analytic harmonic-mean `cfs_score` (brief §13)."""
    import torch
    cfg = cfg or {"n_patches": 16, "dim": 64, "steer_strength": 4.0}
    dim = cfg["dim"]
    smax = cfg.get("steer_strength", 4.0)
    off_concept = concept + 1                     # an unrelated SAE feature

    x, _ = synthetic_batch(32, cfg["n_patches"], dim, seed=9000)
    with torch.no_grad():
        # readout direction = the SAE concept we steer (held-out linear probe);
        # off-target = an unrelated feature that should stay put.
        d_tgt = model.sae.concept_direction(concept)
        d_tgt = d_tgt / (d_tgt.norm() + 1e-8)
        d_off = model.sae.concept_direction(off_concept)
        d_off = d_off / (d_off.norm() + 1e-8)

        def readout(a, d):
            return (a * d).sum(-1).mean(-1)       # [B]

        knobs, target_read, off_read = [], [], []
        for j in range(n_steps):
            s = smax * j / (n_steps - 1)
            a_s = model.steered_activations(x, concept, s)
            knobs.append(s)
            target_read.append(float(readout(a_s, d_tgt).mean()))
            off_read.append(float(readout(a_s, d_off).mean()))

        knobs = torch.tensor(knobs)
        tr = torch.tensor(target_read)
        ofr = torch.tensor(off_read)

        # Monotonicity: ordered smooth response of the target readout to the knob.
        monotonicity = max(_spearman(knobs, tr), 0.0)
        # Specificity: 1 - normalized off-target drift relative to target movement.
        tgt_move = (tr.max() - tr.min()).abs() + 1e-6
        off_move = (ofr.max() - ofr.min()).abs()
        specificity = float(max(0.0, 1.0 - off_move / tgt_move))
        # Sufficiency: standardized effect size (Cohen's-d-style) at full knob.
        base = model.steered_activations(x, concept, 0.0)
        full = model.steered_activations(x, concept, smax)
        r0 = readout(base, d_tgt); r1 = readout(full, d_tgt)
        pooled = (r0.std() + r1.std()) / 2 + 1e-6
        d_eff = float((r1.mean() - r0.mean()).abs() / pooled)
        sufficiency = min(d_eff / 4.0, 1.0)       # map to [0,1]; d~4 is "ample"

    cfs = cfs_score(monotonicity, specificity, sufficiency)
    return {"monotonicity": round(monotonicity, 4),
            "specificity": round(specificity, 4),
            "sufficiency": round(sufficiency, 4),
            "cfs": round(cfs, 4)}


def smoke():
    from .train import SMOKE_CFG, train
    model, _ = train(SMOKE_CFG, steps=30)
    rl = recon_loss(model, cfg=SMOKE_CFG)
    probe = cfs_probe(model, cfg=SMOKE_CFG)
    log.info("smoke eval: recon=%.4f cfs=%.3f", rl, probe["cfs"])
    return {"recon": rl, **probe}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
