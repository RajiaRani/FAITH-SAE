#!/usr/bin/env python3
# =============================================================================
#  sae_real.py  —  Production TopK Sparse Autoencoder for the REAL FAITH-SAE run
#  Author: Rajia Rani
# =============================================================================
#
#  WHAT THIS FILE IS (and why it exists separately from src/model.py)
#  -----------------------------------------------------------------
#  The teaching scaffold in ``src/model.py`` ships a tiny TopK SAE that is great
#  for explaining the IDEA on a laptop. This module is the SCALED, GPU-ready
#  counterpart: same math, but with the engineering tricks that make an SAE on
#  ~300M-1B CLIP ViT patch-token activations actually train well:
#
#    * a *pre-encoder bias* ``b_dec`` subtracted from the input (and added back at
#      decode) — centring the data so the dictionary models DEVIATIONS from the
#      mean activation, not the mean itself (Bricken/Templeton 2023, Gao 2024);
#    * ``z = TopK(relu(W_enc (x - b_dec) + b_enc))`` — hard top-k sparsity, no L1
#      coefficient to babysit (Gao et al. 2024, "Scaling and Evaluating SAEs");
#    * *unit-norm decoder columns*, RE-NORMALISED after every optimizer step, so a
#      feature cannot cheat the recon loss by simply scaling its column up;
#    * an **AuxK auxiliary loss** that revives DEAD features (ones that have not
#      fired for a long window) by asking the top-``aux_k`` of them to reconstruct
#      the residual the live features missed — the standard anti-dead-feature
#      trick from Gao 2024 that keeps the dictionary fully utilised;
#    * ``normalize_activations`` with the ``unit_meansquare`` convention so the
#      loss scale is comparable across backbones/layers (E[||x||^2 / d] = 1).
#
#  We also provide a minimal **L1SAE** (vanilla L1-penalty SAE) used ONLY by the
#  A1 ablation (TopK vs L1). It deliberately reuses the same ``b_dec`` / decoder
#  conventions so the ablation changes ONE thing: the sparsity mechanism.
#
#  RUNNABILITY: this module imports cleanly on a CPU-only box with no open_clip
#  and no datasets — it only needs torch + safetensors. The heavy lifting (real
#  activations) is supplied by data_real.py / train_sae.py; here we just define
#  the model and its losses on whatever ``[n_tokens, d_in]`` tensors arrive.
#
#  For research and educational purposes only.
# =============================================================================
from __future__ import annotations

import json
import math
import os
import sys
import pathlib
from typing import Dict, Optional, Tuple

# --- Make the project's src/ AND this real_run/ dir importable. --------------
# parents[2] of code/real_run/sae_real.py is the repo root (so ``import src...``
# works); inserting this file's own dir lets sibling real_run modules import us.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Activation normalization                                                    #
# --------------------------------------------------------------------------- #
def normalize_activations(x: torch.Tensor, mode: str = "unit_meansquare"
                          ) -> Tuple[torch.Tensor, float]:
    """Scale an activation batch so its magnitude is convention-controlled.

    WHY: CLIP residual-stream activations at layer 22 have some arbitrary overall
    scale (and that scale differs across backbones/layers). If we trained the MSE
    loss on raw activations, the loss number — and the right ``lr`` — would change
    every time we swap backbone or layer. Normalising fixes the scale so a single
    config transfers, and so FVU/explained-variance are interpretable.

    Modes
    -----
    ``unit_meansquare`` : divide by sqrt(E[||x||^2 / d]) so the *mean squared
        coordinate* is 1, i.e. E[||x||^2] = d. This is the Gao-2024 convention
        ("normalize so that E[||x||^2] = d"). The decoder/`b_dec` then live in a
        unit-RMS space, which keeps AuxK and recon losses comparable.
    ``unit_norm``       : divide each vector by its own L2 norm (per-token). Useful
        as an A5-style alternative; changes the geometry, so not the default.
    ``none``            : identity (return scale 1.0).

    Returns ``(x_normalized, scale)`` where ``scale`` is the scalar we divided by
    (a Python float for ``unit_meansquare``/``none``; for ``unit_norm`` the per-
    token division is baked in and we return 1.0 as the reportable global scale).
    The scale is handed back so callers can UN-normalise reconstructions or store
    it in the checkpoint for inference-time consistency.
    """
    if mode in (None, "none"):
        return x, 1.0
    if mode == "unit_meansquare":
        # E[||x||^2 / d] estimated over the whole batch; one global scalar.
        d = x.shape[-1]
        ms = (x.float().pow(2).sum(dim=-1) / d).mean()        # mean of ||x||^2/d
        scale = float(math.sqrt(max(ms.item(), 1e-12)))
        return x / scale, scale
    if mode == "unit_norm":
        # Per-token L2 normalisation (each row becomes a unit vector).
        return F.normalize(x.float(), dim=-1), 1.0
    raise ValueError(f"unknown normalize mode '{mode}'")


# --------------------------------------------------------------------------- #
#  The production TopK SAE                                                      #
# --------------------------------------------------------------------------- #
class TopKSAE(nn.Module):
    """TopK sparse autoencoder over backbone activations (Gao et al. 2024).

    Forward math (per the real-run recipe)::

        x_centered = x - b_dec                       # subtract pre-encoder bias
        pre_acts   = relu(W_enc @ x_centered + b_enc)
        z          = TopK_k(pre_acts)                # keep k largest, rest -> 0
        x_hat      = W_dec @ z + b_dec               # decoder cols are concepts

    The DECODER columns are unit-norm "concept directions" — exactly the vectors
    steering_real.py edits along. ``b_dec`` is initialised to the data mean by the
    trainer (``set_decoder_bias``) so the model immediately reconstructs the mean.
    """

    def __init__(self, d_in: int, n_features: int, k: int, aux_k: int = 256,
                 dead_steps_threshold: int = 10_000_000):
        super().__init__()
        self.d_in = int(d_in)
        self.n_features = int(n_features)
        self.k = int(k)
        # AuxK can ask for more dead features than exist early on; clamp at use.
        self.aux_k = int(aux_k)
        self.dead_steps_threshold = int(dead_steps_threshold)

        # Encoder: W_enc [n_features, d_in], bias b_enc [n_features].
        self.W_enc = nn.Parameter(torch.empty(self.n_features, self.d_in))
        self.b_enc = nn.Parameter(torch.zeros(self.n_features))
        # Decoder: W_dec [d_in, n_features]; column j is concept direction d_j.
        self.W_dec = nn.Parameter(torch.empty(self.d_in, self.n_features))
        # Pre-encoder / decoder bias (the "geometric median"/mean of the data).
        self.b_dec = nn.Parameter(torch.zeros(self.d_in))

        self._init_weights()

        # --- Dead-feature bookkeeping (not learnable; a buffer so it ckpts) ----
        # ``steps_since_fired[j]`` = number of TOKENS seen since feature j last
        # fired. A feature with a count above ``dead_steps_threshold`` is "dead"
        # and becomes eligible for AuxK revival. Stored as a buffer so it rides
        # along in state_dict and survives checkpoint/resume.
        self.register_buffer("steps_since_fired",
                             torch.zeros(self.n_features, dtype=torch.long),
                             persistent=True)

    # ------------------------------------------------------------------ init --
    def _init_weights(self) -> None:
        """Tied-ish init (Gao 2024): decoder columns are random unit vectors and
        the encoder is initialised as the decoder's transpose. This gives a
        sensible starting dictionary and immediately-meaningful gradients."""
        with torch.no_grad():
            # Decoder columns ~ N(0,1), then renormalised to unit length.
            W = torch.randn(self.d_in, self.n_features)
            W = W / (W.norm(dim=0, keepdim=True) + 1e-8)
            self.W_dec.copy_(W)
            # Encoder = decoder^T (the standard tied initialisation).
            self.W_enc.copy_(W.t().contiguous())
            self.b_enc.zero_()
            self.b_dec.zero_()

    @torch.no_grad()
    def set_decoder_bias(self, mean_vec: torch.Tensor) -> None:
        """Initialise ``b_dec`` to the data mean (called once by the trainer on
        the first batch). Reconstructing the mean for free lets the dictionary
        spend its capacity on the interesting deviations."""
        self.b_dec.copy_(mean_vec.to(self.b_dec.dtype).to(self.b_dec.device))

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Renormalise decoder columns to unit L2 norm. CALLED AFTER EVERY
        optimizer step. WHY: without this, a feature could trivially lower the
        recon loss by scaling its column up and its code down (or vice-versa),
        which makes the "concept direction" magnitude meaningless and destabilises
        TopK selection. Pinning ||column|| = 1 makes the code ``z`` the honest
        per-feature strength we later steer with."""
        norm = self.W_dec.data.norm(dim=0, keepdim=True)
        self.W_dec.data /= (norm + 1e-8)

    # --------------------------------------------------------------- encode ---
    def _topk(self, pre_acts: torch.Tensor) -> torch.Tensor:
        """Keep the k largest values per row, zero the rest (hard sparsity)."""
        k = min(self.k, pre_acts.shape[-1])
        # ``topk`` gives values+indices; we scatter the kept values back into a
        # zero tensor so ``z`` is exactly as wide as the dictionary.
        vals, idx = pre_acts.topk(k, dim=-1)
        z = torch.zeros_like(pre_acts)
        z.scatter_(-1, idx, vals)
        return z

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(z, pre_acts)``: the sparse TopK code and the dense relu
        pre-activations (the latter is what AuxK ranks dead features by)."""
        x_centered = x - self.b_dec
        pre_acts = F.relu(F.linear(x_centered, self.W_enc, self.b_enc))
        z = self._topk(pre_acts)
        return z, pre_acts

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """x_hat = W_dec @ z + b_dec (add the pre-encoder bias back)."""
        return F.linear(z, self.W_dec) + self.b_dec

    # ----------------------------------------------------------- aux (dead) ---
    def _auxk_reconstruction(self, x: torch.Tensor, x_hat: torch.Tensor,
                             pre_acts: torch.Tensor) -> Optional[torch.Tensor]:
        """Reconstruct the RESIDUAL (x - x_hat) using ONLY currently-dead
        features, restricted to their top ``aux_k`` pre-activations.

        WHY: a dead feature gets no gradient from the main TopK loss (it is never
        selected), so it stays dead forever — wasted dictionary capacity. AuxK
        gives those features a job: explain what the live features could NOT. If a
        dead feature is genuinely useful here it earns gradient and comes back to
        life. Returns ``None`` when there are no dead features (nothing to do).
        """
        dead_mask = self.steps_since_fired >= self.dead_steps_threshold  # [F]
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return None
        aux_k = min(self.aux_k, n_dead)
        # Consider pre-activations of ONLY the dead features; -inf elsewhere so
        # the topk never picks a live feature.
        masked = pre_acts.masked_fill(~dead_mask.unsqueeze(0), float("-inf"))
        vals, idx = masked.topk(aux_k, dim=-1)
        # Some rows may have fewer than aux_k finite dead pre-acts; relu away the
        # -inf-derived garbage by clamping negatives/non-finite to 0.
        vals = torch.where(torch.isfinite(vals), vals, torch.zeros_like(vals))
        vals = F.relu(vals)
        z_aux = torch.zeros_like(pre_acts)
        z_aux.scatter_(-1, idx, vals)
        # Decode WITHOUT the bias (we are modelling the residual, not the signal).
        residual_hat = F.linear(z_aux, self.W_dec)
        return residual_hat

    # ------------------------------------------------------------- forward ----
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Return ``(x_hat, z, info)``. ``info`` carries everything the loss and
        the dead-feature tracker need (pre_acts, the AuxK residual reconstruction,
        and the live-feature mask for this batch)."""
        z, pre_acts = self.encode(x)
        x_hat = self.decode(z)
        # Which features fired anywhere in this batch (for dead-feature tracking).
        fired = (z != 0).any(dim=0)                              # [n_features]
        residual_hat = self._auxk_reconstruction(x, x_hat.detach(), pre_acts)
        info = {
            "pre_acts": pre_acts,
            "fired": fired,
            "auxk_residual_hat": residual_hat,
            "n_tokens": x.shape[0],
        }
        return x_hat, z, info

    # ----------------------------------------------- dead-feature tracking ----
    @torch.no_grad()
    def update_dead_tracking(self, info: Dict) -> None:
        """Advance the per-feature "tokens since last fired" counter. Features
        that fired this batch reset to 0; the rest increase by the batch size.
        Call once per optimizer step AFTER the forward pass."""
        n_tokens = int(info["n_tokens"])
        fired = info["fired"]
        self.steps_since_fired += n_tokens
        self.steps_since_fired[fired] = 0

    def concept_direction(self, concept: int) -> torch.Tensor:
        """Decoder column ``concept`` = the unit activation-space direction that
        feature switches on. This is the vector steering_real.py edits along."""
        return self.W_dec[:, concept]


# --------------------------------------------------------------------------- #
#  The TopK SAE loss (recon MSE + AuxK)                                         #
# --------------------------------------------------------------------------- #
def sae_loss(x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor,
             info: Dict, cfg) -> Tuple[torch.Tensor, Dict]:
    """Total loss = reconstruction MSE + ``aux_coef`` * AuxK auxiliary loss.

    Reported metrics (all detached floats) feed the trainer's logs:
      * ``recon_mse``  : the primary objective.
      * ``aux_loss``   : AuxK residual MSE (0 when no dead features).
      * ``fvu``        : Fraction of Variance Unexplained = ||x - x_hat||^2 /
                         ||x - mean(x)||^2. 0 = perfect, 1 = no better than mean.
      * ``explained_variance`` = 1 - fvu (the headline "how much did the SAE
                         capture" number).
      * ``l0``         : average number of ACTIVE features per token (~ k).
      * ``dead_frac``  : fraction of the dictionary currently dead.
    """
    aux_coef = _cfg_get(cfg, "aux_coef", 1.0 / 32.0)  # Gao-2024 uses ~1/32.

    # --- Primary reconstruction MSE. -----------------------------------------
    recon_mse = F.mse_loss(x_hat, x)

    # --- FVU / explained variance (normalise recon by the data's own variance).
    # Denominator = variance of x about its mean = how hard the data is to model.
    with torch.no_grad():
        total_var = (x - x.mean(dim=0, keepdim=True)).pow(2).mean()
        fvu = (recon_mse / (total_var + 1e-8)).clamp(min=0.0)
        explained_variance = (1.0 - fvu).clamp(min=-1.0, max=1.0)
        l0 = (z != 0).float().sum(dim=-1).mean()
        dead_frac = (info_dead_mask(info, z)).float().mean()

    # --- AuxK auxiliary loss: dead features reconstruct the live residual. ----
    residual_hat = info.get("auxk_residual_hat", None)
    if residual_hat is not None:
        # The residual the LIVE features missed (use detached x_hat so AuxK does
        # not fight the main objective; it only trains the dead-feature paths).
        residual = (x - x_hat.detach())
        aux_loss = F.mse_loss(residual_hat, residual)
    else:
        aux_loss = x.new_zeros(())

    loss = recon_mse + aux_coef * aux_loss

    metrics = {
        "loss": float(loss.detach()),
        "recon_mse": float(recon_mse.detach()),
        "aux_loss": float(aux_loss.detach()),
        "fvu": float(fvu.detach()),
        "explained_variance": float(explained_variance.detach()),
        "l0": float(l0.detach()),
        "dead_frac": float(dead_frac.detach()),
    }
    return loss, metrics


def info_dead_mask(info: Dict, z: torch.Tensor) -> torch.Tensor:
    """Helper: per-feature dead mask used only for the reported ``dead_frac``.

    We do NOT have direct access to the module here, so we approximate "dead this
    batch" as "did not fire in this batch". The trainer separately reports the
    true long-window dead fraction from the module's ``steps_since_fired`` buffer;
    this per-batch number is a cheap proxy that travels with the loss."""
    fired = info.get("fired", None)
    if fired is not None:
        return ~fired
    return ~(z != 0).any(dim=0)


# --------------------------------------------------------------------------- #
#  Minimal L1-penalty SAE (ablation A1: TopK vs L1)                             #
# --------------------------------------------------------------------------- #
class L1SAE(nn.Module):
    """Vanilla L1-sparse autoencoder — the A1 ablation's "other" SAE type.

    Same ``b_dec`` / unit-norm-decoder conventions as :class:`TopKSAE`, but the
    sparsity mechanism is a soft L1 PENALTY on the code instead of a hard top-k.
    This is the classic Bricken-2023-style SAE; A1 asks whether it changes clean
    feature magnitudes and the downstream CFS. Kept deliberately minimal: it is
    not the production model, only a controlled comparison point.
    """

    def __init__(self, d_in: int, n_features: int, l1_coef: float = 1e-3):
        super().__init__()
        self.d_in = int(d_in)
        self.n_features = int(n_features)
        self.l1_coef = float(l1_coef)
        self.W_enc = nn.Parameter(torch.empty(self.n_features, self.d_in))
        self.b_enc = nn.Parameter(torch.zeros(self.n_features))
        self.W_dec = nn.Parameter(torch.empty(self.d_in, self.n_features))
        self.b_dec = nn.Parameter(torch.zeros(self.d_in))
        with torch.no_grad():
            W = torch.randn(self.d_in, self.n_features)
            W = W / (W.norm(dim=0, keepdim=True) + 1e-8)
            self.W_dec.copy_(W)
            self.W_enc.copy_(W.t().contiguous())

    @torch.no_grad()
    def set_decoder_bias(self, mean_vec: torch.Tensor) -> None:
        self.b_dec.copy_(mean_vec.to(self.b_dec.dtype).to(self.b_dec.device))

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        norm = self.W_dec.data.norm(dim=0, keepdim=True)
        self.W_dec.data /= (norm + 1e-8)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_centered = x - self.b_dec
        pre_acts = F.relu(F.linear(x_centered, self.W_enc, self.b_enc))
        # No top-k: the L1 penalty (applied in the loss) is what drives sparsity.
        return pre_acts, pre_acts

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        z, pre_acts = self.encode(x)
        x_hat = self.decode(z)
        fired = (z > 0).any(dim=0)
        info = {"pre_acts": pre_acts, "fired": fired,
                "auxk_residual_hat": None, "n_tokens": x.shape[0],
                "l1_coef": self.l1_coef}
        return x_hat, z, info

    @torch.no_grad()
    def update_dead_tracking(self, info: Dict) -> None:
        # L1 SAE has no AuxK revival; tracking is a no-op (kept for trainer parity).
        return

    def concept_direction(self, concept: int) -> torch.Tensor:
        return self.W_dec[:, concept]


def l1_sae_loss(x, x_hat, z, info, cfg) -> Tuple[torch.Tensor, Dict]:
    """Recon MSE + L1 penalty on the (decoder-norm-weighted) code. Mirrors the
    metric set of :func:`sae_loss` so the trainer can log A1 uniformly."""
    recon_mse = F.mse_loss(x_hat, x)
    l1_coef = info.get("l1_coef", _cfg_get(cfg, "l1_coef", 1e-3))
    l1 = z.abs().sum(dim=-1).mean()                     # sum over features, mean over tokens
    loss = recon_mse + l1_coef * l1
    with torch.no_grad():
        total_var = (x - x.mean(dim=0, keepdim=True)).pow(2).mean()
        fvu = (recon_mse / (total_var + 1e-8)).clamp(min=0.0)
        l0 = (z > 1e-6).float().sum(dim=-1).mean()
        dead_frac = info_dead_mask(info, z).float().mean()
    metrics = {
        "loss": float(loss.detach()),
        "recon_mse": float(recon_mse.detach()),
        "aux_loss": 0.0,
        "fvu": float(fvu.detach()),
        "explained_variance": float((1.0 - fvu).detach()),
        "l0": float(l0.detach()),
        "dead_frac": float(dead_frac.detach()),
    }
    return loss, metrics


# --------------------------------------------------------------------------- #
#  Save / load (safetensors preferred, torch fallback)                         #
# --------------------------------------------------------------------------- #
def save_sae(sae: nn.Module, path: str, meta: Optional[Dict] = None) -> str:
    """Persist an SAE. We use **safetensors** (fast, zero-copy, no pickle) for the
    tensors and a sidecar JSON for the architecture metadata needed to rebuild the
    module. Falls back to a single ``torch.save`` ``.pt`` if safetensors is absent
    or the path does not end in ``.safetensors``.

    Returns the path actually written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    arch = _arch_dict(sae)
    if meta:
        arch.update({"meta": meta})

    use_safetensors = path.endswith(".safetensors")
    if use_safetensors:
        try:
            from safetensors.torch import save_file
        except Exception as exc:                          # pragma: no cover
            raise RuntimeError(
                "safetensors requested but not importable; install safetensors "
                f"or use a .pt path. ({exc})")
        # safetensors stores only tensors -> CPU, contiguous, and metadata as str.
        tensors = {k: v.detach().cpu().contiguous()
                   for k, v in sae.state_dict().items()}
        save_file(tensors, path, metadata={"arch": json.dumps(arch)})
        # Sidecar JSON too, so humans/tools can read arch without safetensors.
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(arch, f, indent=2)
        return path
    # torch fallback.
    torch.save({"state_dict": sae.state_dict(), "arch": arch}, path)
    return path


def load_sae(path: str, device: str = "cpu") -> nn.Module:
    """Rebuild an SAE from a checkpoint written by :func:`save_sae`. Reads the
    architecture metadata, constructs the right class, loads the tensors."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file, safe_open
        # Recover the arch from the safetensors metadata (or the sidecar JSON).
        arch = None
        try:
            with safe_open(path, framework="pt", device="cpu") as f:
                md = f.metadata() or {}
            if "arch" in md:
                arch = json.loads(md["arch"])
        except Exception:
            arch = None
        if arch is None and os.path.exists(path + ".json"):
            with open(path + ".json", "r", encoding="utf-8") as f:
                arch = json.load(f)
        if arch is None:
            raise RuntimeError(f"no architecture metadata found for {path}")
        sae = _build_from_arch(arch)
        state = load_file(path, device="cpu")
        sae.load_state_dict(state)
    else:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        arch = blob["arch"]
        sae = _build_from_arch(arch)
        sae.load_state_dict(blob["state_dict"])
    return sae.to(device)


# --------------------------------------------------------------------------- #
#  Small internal helpers                                                       #
# --------------------------------------------------------------------------- #
def _cfg_get(cfg, key, default):
    """Read ``key`` from either a dict, a nested {'sae': {...}} config, or an
    object with attributes — so callers can pass whatever they have."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        if key in cfg:
            return cfg[key]
        if "sae" in cfg and isinstance(cfg["sae"], dict) and key in cfg["sae"]:
            return cfg["sae"][key]
        return default
    return getattr(cfg, key, default)


def _arch_dict(sae: nn.Module) -> Dict:
    """Serialise the constructor args needed to rebuild ``sae``."""
    if isinstance(sae, TopKSAE):
        return {"class": "TopKSAE", "d_in": sae.d_in, "n_features": sae.n_features,
                "k": sae.k, "aux_k": sae.aux_k,
                "dead_steps_threshold": sae.dead_steps_threshold}
    if isinstance(sae, L1SAE):
        return {"class": "L1SAE", "d_in": sae.d_in, "n_features": sae.n_features,
                "l1_coef": sae.l1_coef}
    raise TypeError(f"cannot serialise SAE of type {type(sae).__name__}")


def _build_from_arch(arch: Dict) -> nn.Module:
    cls = arch.get("class", "TopKSAE")
    if cls == "TopKSAE":
        return TopKSAE(d_in=arch["d_in"], n_features=arch["n_features"],
                       k=arch["k"], aux_k=arch.get("aux_k", 256),
                       dead_steps_threshold=arch.get("dead_steps_threshold",
                                                     10_000_000))
    if cls == "L1SAE":
        return L1SAE(d_in=arch["d_in"], n_features=arch["n_features"],
                     l1_coef=arch.get("l1_coef", 1e-3))
    raise ValueError(f"unknown SAE class '{cls}'")


def build_sae(cfg) -> nn.Module:
    """Construct the SAE the config asks for. ``sae.expansion`` x ``sae.d_in`` =
    n_features; ``sae.type`` selects TopK (default) vs L1 (A1 ablation)."""
    sae_cfg = cfg.get("sae", cfg) if isinstance(cfg, dict) else cfg
    d_in = int(_cfg_get(sae_cfg, "d_in", 1024))
    expansion = int(_cfg_get(sae_cfg, "expansion", 32))
    n_features = int(_cfg_get(sae_cfg, "n_features", d_in * expansion))
    sae_type = str(_cfg_get(sae_cfg, "type", "topk")).lower()
    if sae_type in ("l1", "vanilla"):
        return L1SAE(d_in=d_in, n_features=n_features,
                     l1_coef=float(_cfg_get(sae_cfg, "l1_coef", 1e-3)))
    return TopKSAE(
        d_in=d_in, n_features=n_features,
        k=int(_cfg_get(sae_cfg, "k", 32)),
        aux_k=int(_cfg_get(sae_cfg, "aux_k", 256)),
        dead_steps_threshold=int(_cfg_get(sae_cfg, "dead_window", 10_000_000)),
    )


# --------------------------------------------------------------------------- #
#  Tiny self-test (CPU, no open_clip): ``python3 sae_real.py``                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    d_in, F_, k = 64, 256, 8
    sae = TopKSAE(d_in=d_in, n_features=F_, k=k, aux_k=16,
                  dead_steps_threshold=10)
    x = torch.randn(128, d_in)
    xn, scale = normalize_activations(x, "unit_meansquare")
    sae.set_decoder_bias(xn.mean(0))
    x_hat, z, info = sae(xn)
    loss, m = sae_loss(xn, x_hat, z, info, {"aux_coef": 1 / 32})
    assert z.shape == (128, F_) and (z != 0).sum(-1).max() <= k
    assert torch.isfinite(loss)
    # decoder columns must be ~unit-norm after a renorm.
    sae.normalize_decoder()
    cols = sae.W_dec.data.norm(dim=0)
    assert torch.allclose(cols, torch.ones_like(cols), atol=1e-5)
    print("sae_real self-test OK | L0=%.1f FVU=%.3f EV=%.3f dead=%.2f scale=%.3f"
          % (m["l0"], m["fvu"], m["explained_variance"], m["dead_frac"], scale))
