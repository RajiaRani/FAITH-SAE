#!/usr/bin/env python
# =============================================================================
# run_full.py  --  FAITH-SAE, the "real result" upgrade of run.py
# -----------------------------------------------------------------------------
# run.py gave ONE number from ONE concept, ONE seed, ONE (clean) dataset.
# That cannot answer a research question -- it is a single noisy dot.
#
# This script answers the question honestly by adding the four things a paper
# needs, all in one self-contained file:
#
#   1. A BIGGER MODEL   -- vit_large_patch16_224 (~307M params), your real scale.
#   2. MANY CONCEPTS    -- the top-N SAE features, not just the single strongest.
#   3. MANY SEEDS       -- re-train the SAE several times; concepts are random,
#                          so we must average over that randomness.
#   4. AN OOD SPLIT     -- the heart of FAITH-SAE: evaluate faithfulness on
#                          DISTRIBUTION-SHIFTED (corrupted) images, not just
#                          clean ones. The claim lives or dies here.
#
# Every measurement is repeated over (concepts x seeds) and reported as
# mean +/- 95% confidence interval, so we can say whether a difference is REAL
# or just noise.  Run `python run_full.py --smoke` to test the whole pipeline in
# seconds on synthetic data with no downloads.
#
# For research and educational purposes only.
# =============================================================================
import argparse, csv, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr

# ----------------------------- settings --------------------------------------
P = argparse.ArgumentParser()
P.add_argument("--model",   default="vit_large_patch16_224",  # 307M; or vit_base_/vit_small_patch16_224
               help="timm ViT name")
P.add_argument("--concepts", type=int, default=8)   # how many SAE features to test
P.add_argument("--seeds",    type=int, default=3)    # how many times to re-train the SAE
P.add_argument("--tokens",   type=int, default=100000)  # patch activations used to train the SAE
P.add_argument("--steps",    type=int, default=1000)    # SAE training steps (more -> cleaner features)
P.add_argument("--features", type=int, default=2048)    # SAE width (dictionary size)
P.add_argument("--topk",     type=int, default=16)      # active features per token
P.add_argument("--rank",     type=int, default=64)      # PCA rank = dimension of the "manifold"
P.add_argument("--smoke",    action="store_true", help="synthetic data, no model/dataset download")
A = P.parse_args()

STRENGTHS = [0.0, 1.0, 2.0, 4.0]              # how hard we push the concept knob
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-9
print(f"device = {DEVICE} | model = {A.model} | concepts = {A.concepts} | seeds = {A.seeds} | smoke = {A.smoke}")


# --------------------------- the ViT backbone --------------------------------
def load_vit():
    """Load a frozen, pretrained ViT and hook a MIDDLE transformer block so we
    can read its activations. Middle layer = len(blocks)//2 (layer 6 for base,
    layer 12 for large) -- mid-depth features are the most concept-like."""
    import timm
    model = timm.create_model(A.model, pretrained=True, num_classes=0).eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    layer = len(model.blocks) // 2
    buf = {}
    model.blocks[layer].register_forward_hook(lambda m, i, o: buf.__setitem__("a", o))
    print(f"loaded {A.model}: width = {model.embed_dim}, hooked layer {layer}/{len(model.blocks)}")
    return model, buf, model.embed_dim


def make_transform(corrupt):
    """Clean vs OOD image preprocessing. The OOD version applies a Gaussian
    BLUR + additive NOISE -- the same family of corruptions as ImageNet-C, a
    real, recognisable distribution shift (not a different task)."""
    import torchvision.transforms as T
    steps = [T.Resize(224), T.CenterCrop(224)]
    if corrupt:
        steps.append(T.GaussianBlur(kernel_size=5, sigma=2.0))
    steps.append(T.ToTensor())
    if corrupt:
        steps.append(T.Lambda(lambda t: (t + 0.12 * torch.randn_like(t)).clamp(0, 1)))
    steps.append(T.Normalize([0.5] * 3, [0.5] * 3))
    return T.Compose(steps)


def collect_activations(model, buf, width, n_tokens, train, corrupt):
    """Run images through the ViT and stack the per-PATCH activations
    (dropping the CLS token). `train`/`corrupt` pick clean-train vs OOD-test."""
    import torchvision
    ds = torchvision.datasets.CIFAR100(root="./data", train=train, download=True,
                                       transform=make_transform(corrupt))
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=2)
    chunks, n = [], 0
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(DEVICE))
            chunks.append(buf["a"][:, 1:, :].reshape(-1, width).cpu())
            n += chunks[-1].shape[0]
            if n >= n_tokens:
                break
    return torch.cat(chunks)[:n_tokens].to(DEVICE)


# ------------------------------ the SAE --------------------------------------
class TopKSAE(nn.Module):
    """A sparse autoencoder: compress each activation to a few active 'concept'
    features, then reconstruct it. The decoder columns are the concept directions."""
    def __init__(self, d, nf, k):
        super().__init__(); self.k = k
        self.enc = nn.Linear(d, nf); self.dec = nn.Linear(nf, d, bias=False)
    def forward(self, x):
        z = F.relu(self.enc(x)); val, idx = z.topk(self.k, dim=-1)
        return self.dec(torch.zeros_like(z).scatter_(-1, idx, val)), z


def train_sae(width, X, seed):
    """Re-seed and train a fresh SAE. Different seeds -> different learned
    features, which is exactly the randomness we average over."""
    torch.manual_seed(seed)
    sae = TopKSAE(width, A.features, A.topk).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for step in range(A.steps):
        idx = torch.randint(0, X.shape[0], (4096,), device=X.device)
        xhat, _ = sae(X[idx]); loss = F.mse_loss(xhat, X[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return sae, loss.item()


def manifold_basis(X):
    """Top-r PCA directions of the data = an orthonormal basis for the 'manifold'
    (the thin slab of activation space the model actually uses)."""
    with torch.no_grad():
        _, _, V = torch.pca_lowrank(X, q=A.rank)
    return V                                  # (width, rank)


def pick_concepts(sae, X, n):
    """Take the n most-used SAE features; each one's normalised decoder column
    is a unit concept direction in activation space."""
    with torch.no_grad():
        _, Z = sae(X[: min(8000, X.shape[0])])
    top = Z.mean(0).topk(n).indices
    dirs = []
    for j in top:
        d = sae.dec.weight[:, int(j)].detach()
        dirs.append(d / (d.norm() + EPS))
    return dirs                               # list of (width,) unit vectors


# --------------------------- the faithfulness score --------------------------
def score(kind, d, others, Ur, base):
    """Compute the FAITH-SAE sub-metrics for steering ONE concept `d`.

    naive      : edit = s * d                (push straight along the concept)
    onmanifold : edit = s * proj(d)          (push only the part inside the manifold)

    mono : does the readout rise monotonically as we push harder? (Spearman, 0..1)
    spec : SPECIFICITY -- how little the push leaks into OTHER concepts (0..1).
           1.0 = perfectly on-target; lower = it also moves unrelated concepts.
    suff : SUFFICIENCY -- size of the on-target effect vs the data's own spread (0..1).
    CFS  : harmonic mean(mono, spec, suff) -- a 'weakest-link' score.
    offman: diagnostic -- fraction of the edit that lands OFF the manifold (lower better)."""
    proj = lambda v: Ur @ (Ur.t() @ v)
    edit_dir = d if kind == "naive" else proj(d)

    # readout of concept d as we increase the push
    means = [((base + s * edit_dir) @ d).mean().item() for s in STRENGTHS]
    mono = max(0.0, spearmanr(STRENGTHS, means).correlation)

    base_std = (base @ d).std().item() + EPS
    suff = float(np.clip((means[-1] - means[0]) / base_std / 4.0, 0.0, 1.0))

    # specificity: at max push, how much do we move OTHER concepts vs this one?
    smax = STRENGTHS[-1]
    on_target = abs((smax * edit_dir) @ d).item() + EPS
    off_target = float(np.mean([abs(((smax * edit_dir) @ o).item()) for o in others])) if others else 0.0
    spec = max(0.0, 1.0 - min(1.0, off_target / on_target))

    # off-manifold residual of the edit (diagnostic, not in CFS)
    e = smax * edit_dir
    offman = (1.0 - (proj(e).norm() / (e.norm() + EPS)).item())

    cfs = 3.0 / (1.0 / (mono + EPS) + 1.0 / (spec + EPS) + 1.0 / (suff + EPS))
    return dict(mono=mono, spec=spec, suff=suff, cfs=cfs, offman=offman)


# ------------------------------- driver --------------------------------------
def get_activations(width):
    """Clean (in-distribution, train) and OOD (corrupted, test) activations,
    both centred by the CLEAN mean so the OOD shift is preserved."""
    if A.smoke:                                       # synthetic: no downloads
        torch.manual_seed(0)
        clean = torch.randn(8000, width, device=DEVICE) * 3.0
        ood = torch.randn(6000, width, device=DEVICE) * 4.0 + 1.5   # shifted + wider
    else:
        model, buf, _ = MODEL                         # reuse the loaded ViT
        clean = collect_activations(model, buf, width, A.tokens, train=True,  corrupt=False)
        ood = collect_activations(model, buf, width, 6000,     train=False, corrupt=True)
    mu = clean.mean(0, keepdim=True)
    return clean - mu, ood - mu


# load the model once (smoke skips the download entirely)
if A.smoke:
    WIDTH = 1024
    MODEL = None
else:
    MODEL = load_vit()
    WIDTH = MODEL[2]

clean, ood = get_activations(WIDTH)
Ur = manifold_basis(clean)
base_clean, base_ood = clean[:4000], ood[:4000]
print(f"activations: clean {tuple(clean.shape)}  ood {tuple(ood.shape)}")

# accumulate one record per (seed, concept, domain, variant)
records = []
for seed in range(A.seeds):
    sae, mse = train_sae(WIDTH, clean, seed)
    dirs = pick_concepts(sae, clean, A.concepts)
    print(f"seed {seed}: SAE mse {mse:.4f}, picked {len(dirs)} concepts")
    for ci, d in enumerate(dirs):
        others = [o for k, o in enumerate(dirs) if k != ci]   # for specificity
        for domain, base in [("clean", base_clean), ("ood", base_ood)]:
            for kind in ["naive", "onmanifold"]:
                r = score(kind, d, others, Ur, base)
                r.update(domain=domain, variant=kind, seed=seed, concept=ci)
                records.append(r)


# ----------------------------- aggregate -------------------------------------
def agg(domain, variant, key):
    vals = np.array([r[key] for r in records if r["domain"] == domain and r["variant"] == variant])
    mean = float(vals.mean())
    ci = 1.96 * float(vals.std()) / math.sqrt(len(vals))     # 95% confidence half-width
    return mean, ci

n_cell = A.seeds * A.concepts
print(f"\n================ FAITH-SAE aggregate  (samples/cell = {n_cell}) ================")
print(f"{'domain':6s} {'variant':11s}  {'mono':>12s} {'spec':>12s} {'suff':>12s} {'CFS':>14s} {'offman':>8s}")
rows = [["domain", "variant", "mono", "mono_ci", "spec", "spec_ci",
         "suff", "suff_ci", "cfs", "cfs_ci", "offman"]]
cfs_tbl = {}
for domain in ["clean", "ood"]:
    for variant in ["naive", "onmanifold"]:
        mo, moc = agg(domain, variant, "mono"); sp, spc = agg(domain, variant, "spec")
        su, suc = agg(domain, variant, "suff"); cf, cfc = agg(domain, variant, "cfs")
        om, _   = agg(domain, variant, "offman")
        cfs_tbl[(domain, variant)] = (cf, cfc)
        print(f"{domain:6s} {variant:11s}  {mo:.3f}±{moc:.3f} {sp:.3f}±{spc:.3f} "
              f"{su:.3f}±{suc:.3f} {cf:.3f} ± {cfc:.3f} {om:8.3f}")
        rows.append([domain, variant, mo, moc, sp, spc, su, suc, cf, cfc, om])

# data-driven verdict (NOT a hard-coded sentence): non-overlapping CIs = real diff
def verdict(domain):
    (cn, cnc), (co, coc) = cfs_tbl[(domain, "naive")], cfs_tbl[(domain, "onmanifold")]
    nlo, nhi = cn - cnc, cn + cnc
    olo, ohi = co - coc, co + coc
    if olo > nhi:   tag = "on-manifold WINS (CIs disjoint)"
    elif nlo > ohi: tag = "naive WINS (CIs disjoint)"
    else:           tag = "TIE (confidence intervals overlap)"
    return f"  {domain:5s}: onmanifold {co:.3f}[{olo:.3f},{ohi:.3f}] vs naive {cn:.3f}[{nlo:.3f},{nhi:.3f}] -> {tag}"

print("\nKEY COMPARISONS (95% CI):")
print(verdict("clean")); print(verdict("ood"))
drop_n = cfs_tbl[("clean", "naive")][0]      - cfs_tbl[("ood", "naive")][0]
drop_o = cfs_tbl[("clean", "onmanifold")][0] - cfs_tbl[("ood", "onmanifold")][0]
robust = "on-manifold degrades LESS (more robust)" if drop_o < drop_n else "naive degrades less"
print(f"  robustness (CFS drop clean->ood):  naive {drop_n:+.3f}   onmanifold {drop_o:+.3f}  -> {robust}")

csv.writer(open("faith_sae_full_results.csv", "w", newline="")).writerows(rows)
print("\nsaved -> faith_sae_full_results.csv")
print("Interpretation is in the table above -- paste it to your guide; no claim is hard-coded.")
