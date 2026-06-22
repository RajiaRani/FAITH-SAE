"""Shared training loop: fit the TopK SAE on (frozen) synthetic activations.

Day-one usage (CPU, no downloads): python -m src.train --smoke
Only the SAE trains; the backbone is frozen and the steerer is parameter-free.
This verifies the whole pipeline (backbone + SAE + optimizer + logging) end to end.
"""
from __future__ import annotations

import argparse

from .data import synthetic_batch
from .model import make_model
from .utils import count_params, get_logger, set_seed

log = get_logger("train")

SMOKE_CFG = dict(seed=0, dim=64, n_patches=16, d_model=64, sae_dim=128,
                 topk_k=8, sae_type="topk", steer="naive_steer",
                 proj_rank=16, steer_strength=4.0)


def train(cfg: dict, steps: int = 50, lr: float = 3e-3, batch: int = 16):
    import torch
    set_seed(cfg.get("seed", 0))
    model = make_model(cfg)
    # only SAE params carry gradients (backbone frozen, steerer parameter-free)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    log.info("steerer '%s' params=%d (trainable=%d)",
             cfg.get("steer"), count_params(model), sum(p.numel() for p in params))
    last = None
    for step in range(steps):
        x, _ = synthetic_batch(batch, cfg["n_patches"], cfg["dim"], seed=step)
        _, loss = model(x)
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.detach())
        if step % max(1, steps // 5) == 0:
            log.info("step %d loss %.4f", step, last)
    return model, last


def smoke():
    model, loss = train(SMOKE_CFG, steps=30)
    assert loss is not None and loss == loss, "loss is NaN"
    log.info("smoke OK, final loss %.4f", loss)
    return loss


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        from .utils import load_config
        train(load_config(args.config))
