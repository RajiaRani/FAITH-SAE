#!/usr/bin/env python3
# =============================================================================
#  train_sae.py  —  Streaming trainer for the REAL FAITH-SAE TopK SAE.
#  Author: Rajia Rani
# =============================================================================
#
#  WHAT THIS SCRIPT DOES
#  ---------------------
#  Trains the production :class:`sae_real.TopKSAE` on the cached CLIP patch-token
#  activations written by ``data_real.extract_activations`` (sharded float16
#  ``acts_imagenet_train_*.npy`` files). It is a STREAMING trainer: the full bank
#  is ~300M x 1024 float16 (~600 GB) and never fits in RAM/VRAM, so we iterate the
#  shards from disk, slice them into ``batch_tokens``-sized minibatches, and run
#  one AdamW step per minibatch until the ``token_budget`` is spent.
#
#  THE ENGINEERING THAT MATTERS AT SCALE (each commented WHY inline):
#    * LINEAR LR WARMUP then hold — SAEs are unstable in the first ~1k steps; a
#      warmup avoids an early loss spike that kills features.
#    * AMP (autocast + GradScaler) on CUDA — bf16/fp16 matmuls ~2-3x throughput on
#      A100/H100. GUARDED so on a CPU-only box autocast is a harmless no-op.
#    * DECODER RENORM after every step — pins ||concept direction|| = 1 (see
#      sae_real.normalize_decoder for why).
#    * DEAD-FEATURE TRACKING over a token window — feeds AuxK so the dictionary
#      stays fully used; we also report the true long-window dead fraction.
#    * SAFETENSORS CHECKPOINTS every ``ckpt_every`` steps — resumable, pickle-free.
#    * METRICS: FVU / explained-variance + L0 + dead% logged throughout.
#
#  TWO PATHS, ONE CODE:
#    * REAL (default): streams real shards via ``data_real.iter_activation_shards``.
#    * ``--smoke``: fabricates synthetic-but-real-SHAPED activation shards on CPU
#      (no open_clip, no dataset, no GPU), trains a tiny SAE for a few hundred
#      steps, and ASSERTS the loss decreases — the offline integration check.
#
#  For research and educational purposes only.
# =============================================================================
from __future__ import annotations

import argparse
import os
import sys
import time
import pathlib
from typing import Dict, Iterator, Optional

# --- Make the repo root (for ``import src...``) AND this real_run/ dir importable.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import torch

# Local SAE (always importable on CPU; needs only torch + safetensors).
from sae_real import (
    TopKSAE, L1SAE, build_sae, sae_loss, l1_sae_loss,
    normalize_activations, save_sae,
)

# Reuse the project's logging/seed helpers (single source of truth).
try:
    from src.utils import get_logger, set_seed
    log = get_logger("train_sae")
except Exception:                                            # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("train_sae")

    def set_seed(seed: int = 0) -> None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
#  Config loading                                                              #
# --------------------------------------------------------------------------- #
def load_real_config(path: str) -> dict:
    """Load a real_run YAML config. Prefers ``data_real.load_real_config`` (the
    canonical owner of the shared schema) if that module is importable; otherwise
    falls back to the project's ``src.utils.load_config`` so this trainer still
    runs standalone before data_real.py exists / when open_clip is missing."""
    try:
        from data_real import load_real_config as _lrc      # type: ignore
        return _lrc(path)
    except Exception:
        from src.utils import load_config
        return load_config(path)


# --------------------------------------------------------------------------- #
#  Activation stream → token minibatches                                       #
# --------------------------------------------------------------------------- #
def _iter_real_shards(cache_dir: str, dataset: str) -> Iterator[np.ndarray]:
    """Yield each activation shard ``[n_tokens, d_in]`` (float16) for ``dataset``.

    Delegates to ``data_real.iter_activation_shards`` (the cache-format owner) so
    there is one reader. Guarded import: if data_real is unavailable we read the
    shard files directly using the documented naming convention, so the trainer
    can run on a pre-populated cache even without the rest of the pipeline."""
    try:
        from data_real import iter_activation_shards          # type: ignore
        yield from iter_activation_shards(cache_dir, dataset)
        return
    except Exception:
        # Fallback reader honouring the documented cache format:
        #   cache_dir/acts_{dataset}_{shard:05d}.npy  float16 [n_tokens, d_in]
        import glob
        pattern = os.path.join(cache_dir, f"acts_{dataset}_*.npy")
        for fp in sorted(glob.glob(pattern)):
            yield np.load(fp, mmap_mode="r")


def stream_token_batches(cache_dir: str, dataset: str, batch_tokens: int,
                         token_budget: int, seed: int = 0
                         ) -> Iterator[torch.Tensor]:
    """Turn the on-disk shards into a stream of ``[batch_tokens, d_in]`` float32
    torch tensors, shuffling WITHIN each shard (cheap, and enough decorrelation
    when shards are written in image order). Stops once ``token_budget`` tokens
    have been emitted — that is what bounds the run length on the real bank."""
    rng = np.random.default_rng(seed)
    emitted = 0
    for shard in _iter_real_shards(cache_dir, dataset):
        arr = np.asarray(shard)                              # may be memmapped
        n = arr.shape[0]
        order = rng.permutation(n)                           # in-shard shuffle
        for start in range(0, n, batch_tokens):
            idx = order[start:start + batch_tokens]
            if idx.size == 0:
                continue
            # float16 on disk -> float32 for stable training math.
            batch = torch.from_numpy(np.ascontiguousarray(arr[idx])).float()
            yield batch
            emitted += batch.shape[0]
            if emitted >= token_budget:
                return


# --------------------------------------------------------------------------- #
#  Smoke data: fabricate real-SHAPED activation shards on CPU                  #
# --------------------------------------------------------------------------- #
def _fabricate_smoke_shards(cfg: dict, cache_dir: str, seed: int = 0) -> str:
    """Write a couple of synthetic-but-real-SHAPED float16 shards so the smoke
    path exercises the EXACT real streaming/cache code (no open_clip, no data).

    The fabricated activations are NOT white noise: we plant a low-rank manifold
    plus a few additive "concept" directions (mirroring the teaching scaffold), so
    the SAE actually has structure to learn and the loss visibly drops — which is
    what the smoke assertion checks."""
    sae_cfg = cfg["sae"]
    d = int(sae_cfg["d_in"])
    n_images = int(cfg.get("data", {}).get("max_images", 32) or 32)
    # ~256 patch tokens/image at real scale; keep it small but >1 for the smoke.
    tokens_per_image = 64
    g = torch.Generator().manual_seed(seed)

    # Low-rank manifold M (real activations live near a thin subspace).
    manifold_dim = max(8, d // 4)
    U, _ = torch.linalg.qr(torch.randn(d, manifold_dim, generator=g))
    # A few planted concept directions inside M.
    n_concepts = 6
    concepts = (U @ torch.randn(manifold_dim, n_concepts, generator=g)).T
    concepts = concepts / (concepts.norm(dim=1, keepdim=True) + 1e-8)

    os.makedirs(cache_dir, exist_ok=True)
    n_shards = 2
    dataset = "imagenet_train"
    rng = np.random.default_rng(seed)
    for s in range(n_shards):
        n_imgs = n_images // n_shards
        latent = torch.randn(n_imgs, tokens_per_image, manifold_dim, generator=g)
        acts = latent @ U.T                                  # lives in M
        acts = acts + 0.05 * torch.randn(n_imgs, tokens_per_image, d, generator=g)
        for c in range(n_concepts):
            present = (torch.rand(n_imgs, 1, 1, generator=g) < 0.4).float()
            amp = present * (1.5 + 2.0 * torch.rand(n_imgs, 1, 1, generator=g))
            acts = acts + amp * concepts[c].view(1, 1, d)
        flat = acts.reshape(-1, d).numpy().astype(np.float16)  # [n_tok, d]
        labels = rng.integers(0, 10, size=flat.shape[0]).astype(np.int64)
        np.save(os.path.join(cache_dir, f"acts_{dataset}_{s:05d}.npy"), flat)
        np.save(os.path.join(cache_dir, f"labels_{dataset}_{s:05d}.npy"), labels)
    # A minimal manifest so a real reader could also pick these up.
    import json
    manifest = {"d_in": d, "layer": cfg["backbone"]["layer"],
                "token_type": cfg["backbone"]["token_type"],
                "n_images": n_images, "n_tokens": int(n_images * tokens_per_image),
                "n_shards": n_shards, "backbone": cfg["backbone"]["name"],
                "image_ids": []}
    with open(os.path.join(cache_dir, f"manifest_{dataset}.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("[smoke] fabricated %d shards of real-shaped activations in %s",
             n_shards, cache_dir)
    return dataset


# --------------------------------------------------------------------------- #
#  The trainer                                                                  #
# --------------------------------------------------------------------------- #
def _select_loss_fn(sae) -> "callable":
    """Pick the matching loss for the SAE type (A1 ablation uses L1SAE)."""
    return l1_sae_loss if isinstance(sae, L1SAE) else sae_loss


def _warmup_lr(step: int, base_lr: float, warmup: int) -> float:
    """Linear warmup from 0 to ``base_lr`` over ``warmup`` steps, then constant.
    (TopK SAEs do not need a decay schedule to converge; warmup is the key bit.)"""
    if warmup <= 0:
        return base_lr
    return base_lr * min(1.0, (step + 1) / float(warmup))


def train_sae(cfg: dict, cache_dir: str, dataset: str = "imagenet_train",
              device: Optional[str] = None, max_steps: Optional[int] = None
              ) -> "torch.nn.Module":
    """Stream the cached activations and fit the TopK SAE.

    Parameters mirror the contract: ``cfg`` is a loaded real_run config dict and
    ``cache_dir`` points at the activation shards. Returns the trained SAE module.
    ``max_steps`` (smoke/debug) caps optimizer steps regardless of token budget.
    """
    set_seed(int(cfg.get("seed", 0)))
    sae_cfg = cfg["sae"]
    paths = cfg.get("paths", {})
    out_dir = paths.get("out_dir", "./outputs")
    ckpt_path = paths.get("sae_ckpt", os.path.join(out_dir, "sae.safetensors"))
    os.makedirs(out_dir, exist_ok=True)

    # Device: honour cfg, but fall back to CPU when CUDA is absent (build box).
    if device is None:
        want = str(cfg.get("backbone", {}).get("device", "cuda"))
        device = want if (want == "cuda" and torch.cuda.is_available()) else "cpu"
    use_amp = (device == "cuda")                             # AMP only on GPU

    # Build the SAE the config asks for (TopK by default; L1 for A1).
    sae = build_sae(cfg).to(device)
    loss_fn = _select_loss_fn(sae)
    norm_mode = str(sae_cfg.get("normalize", "unit_meansquare"))
    base_lr = float(sae_cfg.get("lr", 4e-4))
    warmup = int(sae_cfg.get("warmup", 1000))
    batch_tokens = int(sae_cfg.get("batch_tokens", 8192))
    token_budget = int(sae_cfg.get("token_budget", 300_000_000))
    ckpt_every = int(sae_cfg.get("ckpt_every", 5000))
    aux_coef = float(sae_cfg.get("aux_coef", 1.0 / 32.0))

    opt = torch.optim.AdamW(sae.parameters(), lr=base_lr, betas=(0.9, 0.999),
                            eps=6.25e-10)                    # Gao-2024 eps
    # GradScaler keeps fp16 grads from underflowing; enabled only under AMP. The
    # torch>=2.4 API is ``torch.amp.GradScaler('cuda', ...)`` (older builds expose
    # ``torch.cuda.amp.GradScaler``), so prefer the new one and fall back.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):                      # pragma: no cover
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    log.info("SAE=%s d_in=%d n_features=%d device=%s norm=%s",
             type(sae).__name__, sae.d_in, sae.n_features, device, norm_mode)
    log.info("budget=%s tokens | batch_tokens=%d | lr=%.1e warmup=%d",
             f"{token_budget:,}", batch_tokens, base_lr, warmup)

    stream = stream_token_batches(cache_dir, dataset, batch_tokens, token_budget,
                                  seed=int(cfg.get("seed", 0)) + 1)

    step = 0
    bias_set = False
    t0 = time.perf_counter()
    metrics: Dict[str, float] = {}
    for batch in stream:
        x = batch.to(device, non_blocking=True)
        # Scale-normalise so the loss is comparable across backbones/layers.
        x, _scale = normalize_activations(x, norm_mode)

        # On the very first batch, seed b_dec with the data mean (free recon of
        # the mean; the dictionary then models deviations from it).
        if not bias_set:
            sae.set_decoder_bias(x.mean(0))
            bias_set = True

        # Linear LR warmup.
        lr = _warmup_lr(step, base_lr, warmup)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        # AMP autocast (no-op dtype on CPU). bf16 on GPU is numerically safest.
        amp_dtype = torch.bfloat16 if use_amp else torch.float32
        with torch.autocast(device_type=("cuda" if use_amp else "cpu"),
                            dtype=amp_dtype, enabled=use_amp):
            x_hat, z, info = sae(x)
            loss, metrics = loss_fn(x, x_hat, z, info, {"aux_coef": aux_coef})

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        # Pin decoder columns to unit norm AFTER the step (the renorm trick).
        sae.normalize_decoder()
        # Advance the dead-feature counters (TopK only; no-op for L1).
        sae.update_dead_tracking(info)

        # --- Logging: FVU / explained-variance + L0 + dead% ------------------
        if step % max(1, ckpt_every // 20) == 0 or step == 0:
            true_dead = _true_dead_fraction(sae)
            rate = (info["n_tokens"]) / max(time.perf_counter() - t0, 1e-6)
            log.info("step %6d | lr %.2e | loss %.4f | recon %.4f | FVU %.3f "
                     "| EV %.3f | L0 %.1f | dead %.1f%% | %.0f tok/s",
                     step, lr, metrics["loss"], metrics["recon_mse"],
                     metrics["fvu"], metrics["explained_variance"],
                     metrics["l0"], 100.0 * true_dead, rate)
            t0 = time.perf_counter()

        # --- Periodic checkpoint --------------------------------------------
        if ckpt_every > 0 and step > 0 and step % ckpt_every == 0:
            _save_ckpt(sae, ckpt_path, cfg, step, metrics)

        step += 1
        if max_steps is not None and step >= max_steps:
            break

    # Final checkpoint.
    _save_ckpt(sae, ckpt_path, cfg, step, metrics)
    log.info("DONE: %d steps, final loss %.4f, EV %.3f, dead %.1f%% -> %s",
             step, metrics.get("loss", float("nan")),
             metrics.get("explained_variance", float("nan")),
             100.0 * _true_dead_fraction(sae), ckpt_path)
    return sae


def _true_dead_fraction(sae) -> float:
    """Long-window dead fraction from the module's ``steps_since_fired`` buffer
    (the real metric, vs the per-batch proxy carried in the loss). L1SAE has no
    such buffer, so we report 0."""
    buf = getattr(sae, "steps_since_fired", None)
    thresh = getattr(sae, "dead_steps_threshold", None)
    if buf is None or thresh is None:
        return 0.0
    return float((buf >= thresh).float().mean().item())


def _save_ckpt(sae, ckpt_path: str, cfg: dict, step: int,
               metrics: Dict[str, float]) -> None:
    """Write a safetensors checkpoint with run metadata."""
    meta = {"step": step, "seed": cfg.get("seed", 0),
            "backbone": cfg.get("backbone", {}).get("name", "?"),
            "layer": cfg.get("backbone", {}).get("layer", -1),
            "normalize": cfg.get("sae", {}).get("normalize", "unit_meansquare"),
            "final_loss": metrics.get("loss"),
            "explained_variance": metrics.get("explained_variance")}
    save_sae(sae, ckpt_path, meta=meta)


# --------------------------------------------------------------------------- #
#  Smoke entrypoint                                                             #
# --------------------------------------------------------------------------- #
def smoke(cache_dir: Optional[str] = None) -> Dict[str, float]:
    """Tiny CPU train on fabricated real-shaped shards; assert loss decreases."""
    import tempfile
    cfg = load_real_config(_default_smoke_cfg_path())
    cfg["backbone"]["device"] = "cpu"
    # A few hundred steps is enough to see the loss fall on planted structure.
    steps = 300
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="faith_sae_smoke_")
    cfg.setdefault("paths", {})
    cfg["paths"]["out_dir"] = os.path.join(cache_dir, "outputs")
    cfg["paths"]["sae_ckpt"] = os.path.join(cache_dir, "outputs",
                                            "sae_smoke.safetensors")
    dataset = _fabricate_smoke_shards(cfg, cache_dir, seed=0)

    # Capture the loss at the start vs the end to assert a real decrease. We do a
    # short warmup of the stream, record the first few steps' loss, then train.
    losses = _train_and_collect_losses(cfg, cache_dir, dataset, steps)
    early = float(np.mean(losses[: max(1, len(losses) // 10)]))
    late = float(np.mean(losses[-max(1, len(losses) // 10):]))
    log.info("[smoke] early_loss=%.4f  late_loss=%.4f  (n=%d steps)",
             early, late, len(losses))
    assert all(np.isfinite(losses)), "smoke loss became NaN/Inf"
    assert late < early * 0.95, (
        f"smoke FAILED: loss did not decrease (early={early:.4f} "
        f"late={late:.4f}); the SAE is not learning the planted structure")
    log.info("[smoke] OK — loss decreased %.1f%% (%.4f -> %.4f)",
             100.0 * (1 - late / early), early, late)
    return {"early": early, "late": late, "n_steps": len(losses)}


def _train_and_collect_losses(cfg, cache_dir, dataset, steps):
    """Smoke variant of the loop that records per-step loss for the assertion.
    Mirrors :func:`train_sae` but loops for a fixed step count over the tiny bank,
    re-streaming the fabricated shards as needed (they are small)."""
    set_seed(int(cfg.get("seed", 0)))
    sae = build_sae(cfg)
    sae_cfg = cfg["sae"]
    norm_mode = str(sae_cfg.get("normalize", "unit_meansquare"))
    base_lr = float(sae_cfg.get("lr", 4e-4))
    warmup = int(sae_cfg.get("warmup", 5))
    batch_tokens = int(sae_cfg.get("batch_tokens", 256))
    opt = torch.optim.AdamW(sae.parameters(), lr=base_lr)
    losses, bias_set, step = [], False, 0
    while step < steps:
        # token_budget high so the stream yields enough batches; re-loop shards.
        for batch in stream_token_batches(cache_dir, dataset, batch_tokens,
                                          token_budget=10 ** 9,
                                          seed=step + 1):
            x, _ = normalize_activations(batch, norm_mode)
            if not bias_set:
                sae.set_decoder_bias(x.mean(0))
                bias_set = True
            lr = _warmup_lr(step, base_lr, warmup)
            for pg in opt.param_groups:
                pg["lr"] = lr
            x_hat, z, info = sae(x)
            loss, m = sae_loss(x, x_hat, z, info, {"aux_coef": 1 / 32})
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sae.normalize_decoder()
            sae.update_dead_tracking(info)
            losses.append(m["recon_mse"])
            step += 1
            if step >= steps:
                break
    return losses


def _default_smoke_cfg_path() -> str:
    """Locate configs/smoke.yaml next to this file."""
    here = pathlib.Path(__file__).resolve().parent
    return str(here / "configs" / "smoke.yaml")


# --------------------------------------------------------------------------- #
#  CLI                                                                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train the FAITH-SAE production TopK SAE (streaming).")
    ap.add_argument("--config", default=None,
                    help="path to a real_run YAML config (e.g. configs/vit_l14.yaml)")
    ap.add_argument("--cache_dir", default=None,
                    help="dir with acts_{dataset}_*.npy shards (default: cfg paths)")
    ap.add_argument("--dataset", default="in1k",
                    help="which cached dataset's shards to train on (the in-"
                         "distribution rung; the student's ladder names it 'in1k')")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU train on fabricated real-shaped acts; asserts "
                         "loss decreases (no open_clip / GPU / downloads)")
    args = ap.parse_args()

    if args.smoke:
        smoke(cache_dir=args.cache_dir)
        return

    if args.config is None:
        ap.error("--config is required for the real path (or pass --smoke)")
    cfg = load_real_config(args.config)
    cache_dir = args.cache_dir or cfg.get("paths", {}).get("cache_dir", "./cache")
    train_sae(cfg, cache_dir, dataset=args.dataset)


if __name__ == "__main__":
    main()
