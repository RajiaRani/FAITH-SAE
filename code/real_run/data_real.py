# =============================================================================
# data_real.py  --  THE DATA / BACKBONE / ACTIVATION-CACHE SPINE of FAITH-SAE.
# Project: FAITH-SAE -- Are SAE concept directions in vision models causally
#          faithful under distribution shift?
# Author:  Rajia Rani  ()
#
# WHAT THIS MODULE OWNS (everyone else imports from here):
#   * load_real_config(path)        -> dict   (the shared YAML schema loader)
#   * build_backbone(cfg)           -> (model, preprocess, capture)
#         `capture` is a forward hook on the chosen ViT residual block that
#         returns the residual-stream patch tokens [batch, n_patches, d].
#   * iter_image_batches(cfg, ds)   -> yields (pixels[B,3,H,W], labels[B], ids[B])
#   * extract_activations(cfg,ds,cache_dir) -> writes sharded .npy + manifest.json
#   * iter_activation_shards(cache_dir, ds) -> yields np.float16 [n, d]
#   * load_activation_bank(cache_dir, ds, n_tokens, seed) -> torch.FloatTensor
#
# THE CACHE FORMAT (the contract train_sae / manifold / cfs all read):
#   cache_dir/acts_{ds}_{shard:05d}.npy     float16  [n_tokens, d_in]  (patch tokens)
#   cache_dir/labels_{ds}_{shard:05d}.npy   int64    [n_tokens]        (class/token, -1)
#   cache_dir/manifest_{ds}.json            {d_in, layer, token_type, n_images,
#                                            n_tokens, n_shards, backbone, image_ids}
#
# BACKBONE: the student's run uses standard SUPERVISED timm ViTs (ViT-S/B/L,
# ~22M/86M/304M) as the frozen backbone, selected by backbone.framework: timm
# (the default). The original open_clip CLIP path is preserved and selected with
# backbone.framework: open_clip.
#
# DATASETS: the OOD ladder is the student's own datasets, ordered by shift
# strength:  in1k (in-distribution, also the SAE-training source) -> in100 (mild,
# a 100-class IN-1k subset) -> food101 (domain shift) -> cifar100 (strong domain +
# resolution shift, 32x32 upsampled to 224).
#
# HONESTY: timm, open_clip and the real datasets are NOT installed on the build
# box and there is NO GPU. So this file is written to be CORRECT + GPU-ready, but
# the timm / open_clip / torchvision-dataset imports are GUARDED (try/except) so
# the module still imports on a bare CPU box for the integration self-test. The
# real path is the default; the synthetic smoke path (extract_activations.py
# --smoke) needs none of the optional deps.
# -----------------------------------------------------------------------------
# For research and educational purposes only.
# =============================================================================
from __future__ import annotations

import glob
import json
import os
import sys
import pathlib

import numpy as np

# --- Make src/ (the toy scaffold) and this real_run dir importable ------------
# Per the project convention: parents[2] of this file is the project root
# (.../25_..._FAITH_SAE), whose src/ holds the math we REUSE. We also add the
# real_run dir itself so sibling modules import cleanly.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                       # real_run/  (sibling imports)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # project root

# --- GUARDED heavy imports ----------------------------------------------------
# torch is present on the build box (2.8). open_clip + torchvision.datasets are
# NOT -- guard them so `import data_real` always succeeds for the self-test.
try:
    import torch
    _HAVE_TORCH = True
except Exception as _e:                               # pragma: no cover
    torch = None
    _HAVE_TORCH = False

try:
    import open_clip                                  # the real CLIP backbones
    _HAVE_OPEN_CLIP = True
except Exception:                                     # not installed on build box
    open_clip = None
    _HAVE_OPEN_CLIP = False

# REAL RUN: the student's setup uses standard supervised timm ViTs (S/B/L) as the
# frozen backbone instead of open_clip. timm is import-guarded exactly like
# open_clip so the module still imports (and the CPU self-test still passes) on a
# box without it. The framework is selected per-config (backbone.framework).
try:
    import timm                                       # the supervised ViT backbones
    _HAVE_TIMM = True
except Exception:                                     # not installed on build box
    timm = None
    _HAVE_TIMM = False

try:
    import torchvision
    from torchvision import datasets as tv_datasets
    from torchvision import transforms as tv_transforms
    _HAVE_TORCHVISION = True
except Exception:
    torchvision = None
    tv_datasets = None
    tv_transforms = None
    _HAVE_TORCHVISION = False


# =============================================================================
# CONFIG
# =============================================================================
class _Cfg(dict):
    """Dict you can also dot-access: cfg.backbone.layer as well as cfg['backbone'].

    The shared schema is nested (backbone/data/sae/...), and dot-access keeps the
    module signatures readable (cfg.backbone.layer) while staying a plain dict for
    json/yaml round-tripping. Missing keys raise AttributeError with a clear name.
    """

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return _Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v


def load_real_config(path: str) -> _Cfg:
    """Load a real_run YAML (vit_l14 / vit_b16 / vit_h14 / smoke) into a dot-dict.

    Falls back to JSON if PyYAML is somehow unavailable (mirrors src.utils). Note
    YAML's 300_000_000 underscore-int literals parse fine (PyYAML treats the
    underscores as digit separators), so the big token budgets load as ints.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml
        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    return _Cfg(data)


# =============================================================================
# BACKBONE  +  the residual-stream CAPTURE hook
# =============================================================================
def _backbone_framework(cfg) -> str:
    """Which backbone family to load: 'timm' (default) or 'open_clip'.

    REAL RUN: the student's supercomputer run uses standard supervised `timm`
    ViTs (S/B/L), so 'timm' is the default. The original open_clip CLIP path is
    preserved and selected by setting backbone.framework: open_clip in the YAML.
    """
    bb = cfg.backbone
    try:
        fw = bb.framework
    except AttributeError:
        fw = None
    return str(fw) if fw else "timm"


def build_backbone(cfg):
    """Load the frozen ViT and install a residual-stream patch-token capture hook.

    Dispatches on backbone.framework:
      * 'timm'      (DEFAULT) -- a standard SUPERVISED timm ViT (ViT-S/B/L). This
                    is what the student's supercomputer run uses.
      * 'open_clip' -- the original CLIP ViT path (kept fully working).

    Returns (model, preprocess, capture):
      model      -- the backbone module, eval()'d, all grads off, on device.
      preprocess -- the matching val-time image transform (resize/centercrop/norm).
      capture    -- a zero-arg callable returning the LAST captured patch-token
                    block: a torch.Tensor [batch, n_patches, d] (CLS already
                    dropped). Call the model forward, THEN capture() to read it.

    WHY a hook (not the pooled embedding): we need the *intermediate* residual-
    stream output of transformer block `cfg.backbone.layer`, not the final CLS /
    head embedding -- that intermediate is the representation the SAE reconstructs.
    """
    if not _HAVE_TORCH:
        raise RuntimeError("torch is required for build_backbone().")
    framework = _backbone_framework(cfg)
    if framework == "timm":
        return _build_backbone_timm(cfg)
    if framework == "open_clip":
        return _build_backbone_open_clip(cfg)
    raise ValueError(
        f"unknown backbone.framework {framework!r} (use 'timm' or 'open_clip')."
    )


def _build_backbone_timm(cfg):
    """REAL RUN: load a frozen SUPERVISED timm ViT + a patch-token capture hook.

    Loads `timm.create_model(cfg.backbone.name, pretrained=True, num_classes=0)`
    (num_classes=0 drops the classifier head -- we only want the trunk), frozen +
    eval + no-grad, and registers a forward hook on the chosen transformer block.

    SHAPE NOTE: timm ViT blocks live in `model.blocks` (an nn.ModuleList) and run
    in NLD order, so a block's output is already [batch, tokens, width] -- NO
    transpose needed (unlike open_clip's LND). Token 0 is the CLS token; for
    token_type=patch we drop it (keep the 196 patch tokens @224/16); for
    token_type=cls we keep ONLY token 0 (the patch-vs-CLS ablation).

    The matching val transform is built from timm's own data config so the
    resize/crop/normalization exactly match the pretrained weights.
    """
    if not _HAVE_TIMM:
        raise RuntimeError(
            "timm is not installed. `pip install timm` to run the REAL supervised "
            "ViT backbone, or use extract_activations.py --smoke for the offline "
            "synthetic path (no timm needed)."
        )

    device = cfg.backbone.device
    # num_classes=0 -> headless trunk; pretrained=True -> the supervised weights.
    model = timm.create_model(
        cfg.backbone.name, pretrained=bool(cfg.backbone.get("pretrained", True)),
        num_classes=0,
    )
    model = model.to(device).eval()
    for p in model.parameters():                      # FROZEN: no grads ever
        p.requires_grad_(False)

    # Build the val-time transform that matches the pretrained weights (timm ships
    # the right resize/crop/mean/std in the model's pretrained data config).
    try:
        data_cfg = timm.data.resolve_model_data_config(model)
        preprocess = timm.data.create_transform(**data_cfg, is_training=False)
    except Exception:                                  # pragma: no cover - old timm
        # Minimal fallback transform (224 center-crop + ImageNet norm) if the timm
        # data-config helpers are unavailable on the installed version.
        if not _HAVE_TORCHVISION:
            raise RuntimeError(
                "timm data-config helpers unavailable and torchvision missing; "
                "cannot build the val transform."
            )
        size = int(cfg.backbone.get("image_size", 224))
        preprocess = tv_transforms.Compose([
            tv_transforms.Resize(size + 32),
            tv_transforms.CenterCrop(size),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225)),
        ])

    # --- locate the transformer block list inside the timm ViT ----------------
    try:
        blocks = model.blocks                          # nn.ModuleList of ViT blocks
    except AttributeError as e:                        # pragma: no cover
        raise RuntimeError(
            "Could not find model.blocks -- this timm model variant has a different "
            "trunk layout; adjust _build_backbone_timm() (or pick a vit_* model)."
        ) from e

    layer = int(cfg.backbone.layer)
    if not (0 <= layer < len(blocks)):
        raise ValueError(
            f"backbone.layer={layer} out of range for {cfg.backbone.name} "
            f"({len(blocks)} blocks)."
        )

    token_type = cfg.backbone.token_type
    _stash = {}                                       # closure cell for the hook

    def _hook(_module, _inputs, output):
        # timm ViT blocks output the hidden state directly in NLD =
        # [batch, seq, width]; some return a tuple (hidden, ...). Detach off-graph.
        h = output[0] if isinstance(output, tuple) else output
        h = h.detach()
        if token_type == "patch":
            h = h[:, 1:, :]                           # drop token 0 = CLS
        elif token_type == "cls":
            h = h[:, :1, :]                           # keep ONLY the CLS token
        else:
            raise ValueError(f"unknown token_type {token_type!r}")
        _stash["acts"] = h.contiguous()               # [batch, n_tokens, width]

    handle = blocks[layer].register_forward_hook(_hook)
    model._faith_hook = handle
    # Tag the framework so extract_activations() knows which forward to call.
    model._faith_framework = "timm"

    def capture():
        if "acts" not in _stash:
            raise RuntimeError("capture() called before a model forward pass.")
        return _stash["acts"]

    return model, preprocess, capture


def _build_backbone_open_clip(cfg):
    """Load the frozen open_clip ViT and install a residual-stream capture hook.

    Returns (model, preprocess, capture):
      model      -- the open_clip CLIP module, eval()'d, all grads off, on device.
      preprocess -- the matching val-time image transform (resize/centercrop/norm).
      capture    -- a zero-arg callable returning the LAST captured patch-token
                    block: a torch.Tensor [batch, n_patches, d] (CLS already
                    dropped). Call the model forward, THEN capture() to read it.

    WHY a hook (not model.encode_image): we need the *intermediate* residual-stream
    output of resblock `cfg.backbone.layer`, not the final pooled CLS embedding.
    open_clip's ViT trunk is model.visual.transformer.resblocks (an nn.ModuleList);
    we register a forward hook on resblock[layer] and stash its output.

    SHAPE NOTE (the subtle bit): open_clip's transformer runs in LND order, i.e.
    the block output is [seq, batch, width]. token 0 is the CLS token. We
    transpose to [batch, seq, width] and (for token_type=patch) drop token 0 to
    keep exactly the 196 (B/16) / 256 (L/14, H/14) patch tokens. For
    token_type=cls we keep ONLY token 0 (the A5 patch-vs-CLS ablation).
    """
    if not _HAVE_OPEN_CLIP:
        raise RuntimeError(
            "open_clip is not installed. `pip install open_clip_torch` to run the "
            "REAL backbone, or use extract_activations.py --smoke for the offline "
            "synthetic path (no open_clip needed)."
        )

    device = cfg.backbone.device
    # open_clip bundles the model + the train/val transforms; we want the val one.
    model, _preprocess_train, preprocess = open_clip.create_model_and_transforms(
        cfg.backbone.name, pretrained=cfg.backbone.pretrained
    )
    model = model.to(device).eval()
    for p in model.parameters():                      # FROZEN: no grads ever
        p.requires_grad_(False)

    # --- locate the residual block list inside the visual trunk ---------------
    # open_clip's standard ViT exposes model.visual.transformer.resblocks.
    try:
        resblocks = model.visual.transformer.resblocks
    except AttributeError as e:                        # pragma: no cover
        raise RuntimeError(
            "Could not find model.visual.transformer.resblocks -- this open_clip "
            "model variant has a different trunk layout; adjust build_backbone()."
        ) from e

    layer = int(cfg.backbone.layer)
    if not (0 <= layer < len(resblocks)):
        raise ValueError(
            f"backbone.layer={layer} out of range for {cfg.backbone.name} "
            f"({len(resblocks)} resblocks)."
        )

    token_type = cfg.backbone.token_type
    _stash = {}                                       # closure cell for the hook

    def _hook(_module, _inputs, output):
        # `output` is the resblock's residual-stream output. open_clip runs the
        # transformer in LND = [seq, batch, width]. Some forks return a tuple; the
        # first element is the hidden state. Detach + move off-graph immediately.
        h = output[0] if isinstance(output, tuple) else output
        h = h.detach()
        # LND -> NLD : [seq, batch, width] -> [batch, seq, width]
        h = h.transpose(0, 1).contiguous()
        if token_type == "patch":
            h = h[:, 1:, :]                           # drop token 0 = CLS
        elif token_type == "cls":
            h = h[:, :1, :]                           # keep ONLY the CLS token
        else:
            raise ValueError(f"unknown token_type {token_type!r}")
        _stash["acts"] = h                            # [batch, n_tokens, width]

    handle = resblocks[layer].register_forward_hook(_hook)
    # Keep the handle on the model so callers can model._faith_hook.remove() if
    # they ever need to (we don't auto-remove -- the backbone lives for the run).
    model._faith_hook = handle
    # Tag the framework so extract_activations() knows which forward to call.
    model._faith_framework = "open_clip"

    def capture():
        if "acts" not in _stash:
            raise RuntimeError("capture() called before a model forward pass.")
        return _stash["acts"]

    return model, preprocess, capture


# =============================================================================
# IMAGE ITERATORS  --  the STUDENT'S domain-shift ladder (REAL RUN)
#
# The OOD ladder is the four datasets the student already has, ordered by shift
# strength (this is the new axis the model-size sweep crosses):
#     in1k     in-distribution  (ImageNet-1k train; ALSO the SAE-training source)
#     in100    mild shift       (a 100-class SUBSET of ImageNet, via a class list)
#     food101  domain shift     (torchvision Food101, download=True)
#     cifar100 strong shift     (torchvision CIFAR100, 32x32 upsampled -> 224)
#
# in1k/in100 are ImageFolders pointed at $IMAGENET_DIR (already on the cluster);
# food101/cifar100 are torchvision datasets that download themselves (small).
# Everything is GUARDED -- none of it touches the CPU synthetic self-test, which
# never calls these iterators (it fabricates shards directly).
# =============================================================================
def _in100_class_list(cfg):
    """REAL RUN: load the 100 ImageNet wnids that define IN-100, or None.

    IN-100 is a 100-class SUBSET of ImageNet-1k: we filter the IN-1k ImageFolder
    down to these classes. The wnid list is read from data.in100_classes_file
    (default: in100_classes.txt next to this module). One wnid per line (e.g.
    n01440764); blank lines and #-comments ignored. If the file is empty/missing
    we return None and the caller falls back to the full IN-1k classes (so the
    pipeline still runs before the student fills the list in).
    """
    data = cfg.data
    path = data.get("in100_classes_file", None) if hasattr(data, "get") else None
    if not path:
        path = str(_HERE / "in100_classes.txt")
    if not os.path.exists(path):
        return None
    wnids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            wnids.append(line)
    return wnids or None


def _imagefolder_with_id(root, preprocess, class_filter=None):
    """Return an ImageFolder that also yields a stable per-image id (rel path).

    class_filter (optional) restricts the ImageFolder to a set of class subdir
    names (the IN-100 wnid list) -- this is how IN-100 is derived from IN-1k.
    """
    class _IFWithId(tv_datasets.ImageFolder):
        def __getitem__(self, index):
            pixels, label = super().__getitem__(index)
            path, _ = self.samples[index]
            image_id = os.path.relpath(path, root)
            return pixels, label, image_id

    if class_filter is not None:
        keep = set(class_filter)

        # find_classes is the torchvision hook that decides which subdirs become
        # classes; overriding it filters IN-1k down to the IN-100 subset, keeping
        # each kept class's ORIGINAL IN-1k index is not required (probes are
        # one-vs-rest over whatever labels are present), so we relabel 0..K-1.
        class _IFSubset(_IFWithId):
            def find_classes(self, directory):
                classes, _ = super().find_classes(directory)
                classes = [c for c in classes if c in keep]
                if not classes:
                    raise RuntimeError(
                        f"IN-100 class list matched 0 of the subdirs under {directory}; "
                        "check in100_classes.txt holds the right wnids."
                    )
                class_to_idx = {c: i for i, c in enumerate(classes)}
                return classes, class_to_idx

        return _IFSubset(root, transform=preprocess)
    return _IFWithId(root, transform=preprocess)


def _build_image_dataset(cfg, dataset_name: str, preprocess):
    """Return a torch Dataset of (pixels, label, image_id) for `dataset_name`.

    REAL RUN: the four rungs of the student's ladder.
      in1k / clean / imagenet_train -> ImageFolder($IMAGENET_DIR)            (in-dist)
      in100                         -> ImageFolder($IMAGENET_DIR) filtered    (mild)
                                       to the IN-100 wnid class list
      food101                       -> torchvision.datasets.Food101(download=True)
      cifar100                      -> torchvision.datasets.CIFAR100(download=True);
                                       32x32 is upsampled to 224 by `preprocess`,
                                       so CIFAR is a STRONG domain+resolution shift.

    The image_id is a stable per-image string (-> manifest -> max-activating-image
    lookup). For the torchvision tensor datasets we synthesize an index-based id.
    """
    if not _HAVE_TORCHVISION:
        raise RuntimeError(
            "torchvision is required for the real image iterators. Use --smoke for "
            "the offline synthetic path."
        )
    data = cfg.data
    name = dataset_name

    # --- in1k (in-distribution; the SAE-training source) ---------------------
    if name in ("in1k", "clean", "imagenet", "imagenet_train"):
        return _imagefolder_with_id(data.imagenet_train_dir, preprocess)

    # --- in100 (mild shift: a 100-class IN-1k subset) ------------------------
    if name == "in100":
        # IN-100 is derived from IN-1k by filtering to a fixed 100-class wnid list.
        # Point at the IN-100 dir if the student staged one; else filter IN-1k.
        in100_dir = data.get("in100_dir", None) if hasattr(data, "get") else None
        if in100_dir and os.path.isdir(in100_dir):
            return _imagefolder_with_id(in100_dir, preprocess)
        return _imagefolder_with_id(
            data.imagenet_train_dir, preprocess,
            class_filter=_in100_class_list(cfg),
        )

    # --- food101 (domain shift; torchvision downloads it -- small) -----------
    if name == "food101":
        root = data.get("food101_dir", None) if hasattr(data, "get") else None
        root = root or os.path.join(getattr(data, "data_dir", "./data"), "food101")
        base = tv_datasets.Food101(root=root, split="test", download=True,
                                   transform=preprocess)
        return _TVWithId(base, prefix="food101")

    # --- cifar100 (strong domain + resolution shift; 32x32 -> 224) -----------
    if name == "cifar100":
        root = data.get("cifar100_dir", None) if hasattr(data, "get") else None
        root = root or os.path.join(getattr(data, "data_dir", "./data"), "cifar100")
        # CIFAR is LOW-RES (32x32); `preprocess` upsamples to 224, so the ViT sees
        # blurry, off-distribution inputs -> the strongest shift on the ladder.
        base = tv_datasets.CIFAR100(root=root, train=False, download=True,
                                    transform=preprocess)
        return _TVWithId(base, prefix="cifar100")

    raise ValueError(
        f"unknown dataset_name {name!r} (expected in1k|in100|food101|cifar100)."
    )


# Base class chosen at import time: torch.utils.data.Dataset when torch is present
# (the real path), else plain object so the module still imports on a bare box.
_TV_BASE = torch.utils.data.Dataset if _HAVE_TORCH else object


class _TVWithId(_TV_BASE):
    """Wrap a torchvision (image, label) dataset to also return a stable id.

    Food101 / CIFAR100 yield (pixels, label); we synthesize a per-image id of the
    form '<prefix>/<index:07d>' so the cache manifest carries provenance the same
    way the ImageFolder rungs do.
    """

    def __init__(self, base, prefix):
        self.base = base
        self.prefix = prefix

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        pixels, label = self.base[index]
        return pixels, int(label), f"{self.prefix}/{index:07d}"


def iter_image_batches(cfg, dataset_name: str):
    """Yield (pixels[B,3,H,W] float32, labels[B] int64, image_ids[B] list[str]).

    Streams the dataset through a torch DataLoader using the open_clip val
    transform built in build_backbone(). Honours cfg.data.batch_size /
    num_workers and the optional cfg.data.max_images cap (handy for partial runs).
    The backbone forward + activation capture is done by extract_activations(),
    NOT here -- this iterator only delivers preprocessed pixels.
    """
    if not _HAVE_TORCH or not _HAVE_TORCHVISION:
        raise RuntimeError("torch + torchvision required for iter_image_batches().")
    # We need the SAME preprocess the model expects -> build the backbone's
    # transform. (Loading the model too is cheap relative to the forward pass and
    # guarantees the transform matches the weights.)
    _model, preprocess, _capture = build_backbone(cfg)
    dataset = _build_image_dataset(cfg, dataset_name, preprocess)

    def _collate(batch):
        pix = torch.stack([b[0] for b in batch], 0)
        lab = torch.tensor([b[1] for b in batch], dtype=torch.long)
        ids = [b[2] for b in batch]
        return pix, lab, ids

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        collate_fn=_collate,
        pin_memory=(cfg.backbone.device != "cpu"),
        drop_last=False,
    )

    max_images = cfg.data.get("max_images", None)
    seen = 0
    for pix, lab, ids in loader:
        if max_images is not None and seen >= max_images:
            break
        if max_images is not None and seen + pix.shape[0] > max_images:
            keep = max_images - seen
            pix, lab, ids = pix[:keep], lab[:keep], ids[:keep]
        seen += pix.shape[0]
        yield pix, lab, ids


# =============================================================================
# ACTIVATION EXTRACTION  ->  sharded .npy + manifest.json (the cache contract)
# =============================================================================
def _shard_paths(cache_dir: str, dataset_name: str, shard: int):
    return (
        os.path.join(cache_dir, f"acts_{dataset_name}_{shard:05d}.npy"),
        os.path.join(cache_dir, f"labels_{dataset_name}_{shard:05d}.npy"),
    )


def _manifest_path(cache_dir: str, dataset_name: str):
    return os.path.join(cache_dir, f"manifest_{dataset_name}.json")


def _write_manifest(cache_dir, dataset_name, cfg, n_images, n_tokens, n_shards,
                    image_ids):
    """Write cache_dir/manifest_{ds}.json per the cache contract."""
    # pretrained is a weight-tag string for open_clip but a bool (True) for timm,
    # so render it generically. Record the framework too for provenance.
    pretrained = cfg.backbone.get("pretrained", True)
    framework = _backbone_framework(cfg)
    manifest = {
        "dataset": dataset_name,
        "d_in": int(cfg.sae.d_in),
        "layer": int(cfg.backbone.layer),
        "token_type": cfg.backbone.token_type,
        "n_images": int(n_images),
        "n_tokens": int(n_tokens),
        "n_shards": int(n_shards),
        "framework": framework,
        "backbone": f"{cfg.backbone.name}:{pretrained}",
        "image_ids": list(image_ids),
    }
    with open(_manifest_path(cache_dir, dataset_name), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return manifest


def extract_activations(cfg, dataset_name: str, cache_dir: str,
                        tokens_per_shard: int = 2_000_000):
    """Run the frozen backbone over `dataset_name` and write the activation cache.

    For every image batch: forward the backbone, grab the residual-stream patch
    tokens via the capture hook, flatten [B, n_patches, d] -> [B*n_patches, d],
    cast to float16, broadcast the image's class label across its tokens, and
    append to the current shard buffer. When the buffer reaches `tokens_per_shard`
    we flush acts_{ds}_{shard}.npy + labels_{ds}_{shard}.npy. Finally we write the
    manifest. This is the cache EVERY downstream module reads (train_sae streams
    the shards; manifold/cfs sample a bank from them).

    Returns the manifest dict. The real path needs open_clip + torchvision + a GPU;
    the offline synthetic equivalent lives in extract_activations.py --smoke.
    """
    if not _HAVE_TORCH:
        raise RuntimeError("torch required for extract_activations().")
    os.makedirs(cache_dir, exist_ok=True)

    model, _preprocess, capture = build_backbone(cfg)
    device = cfg.backbone.device
    d_in = int(cfg.sae.d_in)

    acts_buf, labels_buf = [], []
    buf_tokens = 0
    shard = 0
    n_tokens_total = 0
    n_images_total = 0
    all_image_ids = []

    def _flush():
        nonlocal acts_buf, labels_buf, buf_tokens, shard, n_tokens_total
        if buf_tokens == 0:
            return
        acts = np.concatenate(acts_buf, axis=0).astype(np.float16)
        labels = np.concatenate(labels_buf, axis=0).astype(np.int64)
        ap, lp = _shard_paths(cache_dir, dataset_name, shard)
        np.save(ap, acts)
        np.save(lp, labels)
        n_tokens_total += acts.shape[0]
        shard += 1
        acts_buf, labels_buf, buf_tokens = [], [], 0

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda"
        else _nullcontext()
    )
    framework = getattr(model, "_faith_framework", _backbone_framework(cfg))
    with torch.no_grad():
        for pix, lab, ids in iter_image_batches(cfg, dataset_name):
            pix = pix.to(device, non_blocking=True)
            with autocast:
                # Forward the trunk; the hook stashes the chosen block's residual-
                # stream patch tokens as a side effect. timm ViTs are forwarded as
                # model(pix); open_clip uses its image tower model.visual(pix).
                if framework == "timm":
                    model(pix)
                else:
                    model.visual(pix)
            h = capture()                              # [B, n_patches, d] on device
            B, P, D = h.shape
            if D != d_in:
                raise ValueError(
                    f"captured width {D} != cfg.sae.d_in {d_in}; check backbone/layer."
                )
            flat = h.reshape(B * P, D).to(torch.float16).cpu().numpy()
            # Broadcast each image's class label across its P patch tokens. -1 if
            # the dataset has no usable label (kept for the contract's int dtype).
            lab_np = lab.cpu().numpy()
            tok_labels = np.repeat(lab_np, P).astype(np.int64)

            acts_buf.append(flat)
            labels_buf.append(tok_labels)
            buf_tokens += flat.shape[0]
            n_images_total += B
            all_image_ids.extend(ids)
            if buf_tokens >= tokens_per_shard:
                _flush()
    _flush()                                           # final partial shard

    return _write_manifest(
        cache_dir, dataset_name, cfg,
        n_images=n_images_total, n_tokens=n_tokens_total,
        n_shards=shard, image_ids=all_image_ids,
    )


class _nullcontext:
    """Tiny no-op context manager (autocast stand-in on CPU)."""

    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# =============================================================================
# CACHE READERS  (the interface train_sae / manifold / cfs use)
# =============================================================================
def read_manifest(cache_dir: str, dataset_name: str) -> dict:
    """Load cache_dir/manifest_{ds}.json (raises if extraction hasn't run)."""
    path = _manifest_path(cache_dir, dataset_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No manifest at {path}. Run extract_activations for '{dataset_name}' first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _shard_files(cache_dir: str, dataset_name: str):
    """Sorted list of activation shard files for a dataset (glob the cache dir)."""
    pat = os.path.join(cache_dir, f"acts_{dataset_name}_*.npy")
    return sorted(glob.glob(pat))


def iter_activation_shards(cache_dir: str, dataset_name: str):
    """Yield np.float16 [n_tokens, d_in] arrays, one per shard, in order.

    Memory-mapped (mmap_mode='r') so a 300M-token cache streams without loading
    every shard into RAM at once -- train_sae iterates this to feed the SAE.
    """
    files = _shard_files(cache_dir, dataset_name)
    if not files:
        raise FileNotFoundError(
            f"No activation shards for '{dataset_name}' in {cache_dir}. "
            f"Run extract_activations first (or --smoke to fabricate them)."
        )
    for f in files:
        yield np.load(f, mmap_mode="r")


def iter_labeled_shards(cache_dir: str, dataset_name: str):
    """Yield (acts[n,d] float16, labels[n] int64) pairs, one per shard.

    The label sibling shares the shard index with its acts file. Probe training
    (probes.py) and concept selection use this to align tokens with class labels.
    """
    afiles = _shard_files(cache_dir, dataset_name)
    if not afiles:
        raise FileNotFoundError(
            f"No activation shards for '{dataset_name}' in {cache_dir}."
        )
    for af in afiles:
        lf = af.replace("acts_", "labels_")
        labels = np.load(lf, mmap_mode="r") if os.path.exists(lf) else None
        yield np.load(af, mmap_mode="r"), labels


def load_activation_bank(cache_dir: str, dataset_name: str, n_tokens: int,
                         seed: int = 0):
    """Sample a random bank of `n_tokens` activations as a torch.FloatTensor [n,d].

    Used by manifold.estimate_manifold_basis (SVD of a real-activation bank) and by
    the CFS eval (a held-out evaluation set of real tokens). We reservoir-style
    sample WITHOUT loading the whole cache: walk shards, draw a proportional slice
    from each with a seeded RNG, concatenate, then trim/shuffle to exactly n_tokens.

    Returns float32 (SVD / probe math wants full precision even though the cache is
    stored as float16 to save disk/IO).
    """
    if not _HAVE_TORCH:
        raise RuntimeError("torch required for load_activation_bank().")
    rng = np.random.default_rng(seed)
    files = _shard_files(cache_dir, dataset_name)
    if not files:
        raise FileNotFoundError(
            f"No activation shards for '{dataset_name}' in {cache_dir}."
        )
    # First pass: per-shard token counts (cheap header read via mmap .shape).
    counts = [np.load(f, mmap_mode="r").shape[0] for f in files]
    total = int(sum(counts))
    take = min(int(n_tokens), total)
    # Per-shard quota: proportional, then distribute any rounding remainder so the
    # quotas sum to EXACTLY `take` (proportional rounding can fall a few short).
    quotas = [min(c, int(take * c / total)) for c in counts]
    deficit = take - sum(quotas)
    i = 0
    # Round-robin the remainder onto shards that still have spare capacity.
    while deficit > 0:
        if quotas[i] < counts[i]:
            quotas[i] += 1
            deficit -= 1
        i = (i + 1) % len(counts)
    chunks = []
    for f, c, q in zip(files, counts, quotas):
        if q == 0:
            continue
        arr = np.load(f, mmap_mode="r")
        idx = rng.choice(c, size=q, replace=False)
        idx.sort()                                    # sorted -> fast mmap gather
        chunks.append(np.asarray(arr[idx], dtype=np.float32))
    bank = np.concatenate(chunks, axis=0)             # exactly `take` rows
    rng.shuffle(bank)                                 # decorrelate shard ordering
    return torch.from_numpy(np.ascontiguousarray(bank)).float()


# =============================================================================
# SELF-CHECK: `python3 data_real.py` confirms the module imports + reports deps.
# (No open_clip / dataset access -- this is the offline import sanity line.)
# =============================================================================
if __name__ == "__main__":
    print("[data_real] import OK")
    print(f"  torch={_HAVE_TORCH}  timm={_HAVE_TIMM}  "
          f"open_clip={_HAVE_OPEN_CLIP}  torchvision={_HAVE_TORCHVISION}")
    print("  (timm / open_clip / datasets absent on the build box is EXPECTED; the "
          "real path runs on a GPU box, the smoke path needs neither.)")
