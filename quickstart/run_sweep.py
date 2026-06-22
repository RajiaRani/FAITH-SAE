#!/usr/bin/env python
# =============================================================================
# run_sweep.py  --  the manifold-RANK sweep (FAITH-SAE RQ2)
# -----------------------------------------------------------------------------
# run_full.py showed that on-manifold steering at rank 64 loses to naive,
# because projecting a concept onto a tiny 64-of-1024 slab throws away most of
# the steering signal. But rank is a KNOB we never tuned.
#
# This script asks the real research question: as we ENLARGE the manifold
# (rank 64 -> 128 -> 256 -> 512), does on-manifold steering climb back up to
# (or past) naive?  naive ignores the manifold, so its CFS is a flat reference
# line; on-manifold should rise with rank. A crossover = the headline result.
#
# It is efficient: it loads the ViT, collects activations, and trains the SAEs
# only ONCE, then re-scores every rank (the SAE and the concepts do not depend
# on rank -- only the projection does).
#
# It also reports an honest OOD-DRIFT diagnostic: how much of the corrupted
# data's energy lands OFF the clean manifold. That is the real distribution-
# shift signal, and it is what on-manifold steering is supposed to exploit.
#
# Run `python run_sweep.py --smoke` to test in seconds with no downloads.
# For research and educational purposes only.
# =============================================================================
import argparse, csv, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr

P = argparse.ArgumentParser()
P.add_argument("--model",    default="vit_large_patch16_224")
P.add_argument("--ranks",    default="64,128,256,512", help="comma-separated manifold ranks to sweep")
P.add_argument("--concepts", type=int, default=8)
P.add_argument("--seeds",    type=int, default=3)
P.add_argument("--tokens",   type=int, default=100000)
P.add_argument("--steps",    type=int, default=1000)
P.add_argument("--features", type=int, default=2048)
P.add_argument("--topk",     type=int, default=16)
P.add_argument("--smoke",    action="store_true")
A = P.parse_args()

RANKS = [int(r) for r in A.ranks.split(",")]
STRENGTHS = [0.0, 1.0, 2.0, 4.0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-9
print(f"device = {DEVICE} | model = {A.model} | ranks = {RANKS} | "
      f"concepts = {A.concepts} | seeds = {A.seeds} | smoke = {A.smoke}")


# --------------------------- model + data (same as run_full) -----------------
def load_vit():
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
    import torchvision.transforms as T
    steps = [T.Resize(224), T.CenterCrop(224)]
    if corrupt:
        steps.append(T.GaussianBlur(kernel_size=5, sigma=2.0))
    steps.append(T.ToTensor())
    if corrupt:
        steps.append(T.Lambda(lambda t: (t + 0.12 * torch.randn_like(t)).clamp(0, 1)))
    steps.append(T.Normalize([0.5] * 3, [0.5] * 3))
    return T.Compose(steps)


def collect(model, buf, width, n_tokens, train, corrupt):
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


class TopKSAE(nn.Module):
    def __init__(self, d, nf, k):
        super().__init__(); self.k = k
        self.enc = nn.Linear(d, nf); self.dec = nn.Linear(nf, d, bias=False)
    def forward(self, x):
        z = F.relu(self.enc(x)); val, idx = z.topk(self.k, dim=-1)
        return self.dec(torch.zeros_like(z).scatter_(-1, idx, val)), z


def train_sae(width, X, seed):
    torch.manual_seed(seed)
    sae = TopKSAE(width, A.features, A.topk).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for step in range(A.steps):
        idx = torch.randint(0, X.shape[0], (4096,), device=X.device)
        xhat, _ = sae(X[idx]); loss = F.mse_loss(xhat, X[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return sae, loss.item()


def pick_concepts(sae, X, n):
    with torch.no_grad():
        _, Z = sae(X[: min(8000, X.shape[0])])
    dirs = []
    for j in Z.mean(0).topk(n).indices:
        d = sae.dec.weight[:, int(j)].detach()
        dirs.append(d / (d.norm() + EPS))
    return dirs


def score(kind, d, others, Ur, base):
    proj = lambda v: Ur @ (Ur.t() @ v)
    edit_dir = d if kind == "naive" else proj(d)
    means = [((base + s * edit_dir) @ d).mean().item() for s in STRENGTHS]
    mono = max(0.0, spearmanr(STRENGTHS, means).correlation)
    base_std = (base @ d).std().item() + EPS
    suff = float(np.clip((means[-1] - means[0]) / base_std / 4.0, 0.0, 1.0))
    smax = STRENGTHS[-1]
    on_t = abs((smax * edit_dir) @ d).item() + EPS
    off_t = float(np.mean([abs(((smax * edit_dir) @ o).item()) for o in others])) if others else 0.0
    spec = max(0.0, 1.0 - min(1.0, off_t / on_t))
    e = smax * edit_dir
    offman = 1.0 - (proj(e).norm() / (e.norm() + EPS)).item()
    cfs = 3.0 / (1.0 / (mono + EPS) + 1.0 / (spec + EPS) + 1.0 / (suff + EPS))
    return dict(mono=mono, spec=spec, suff=suff, cfs=cfs, offman=offman)


def manifold_drift(Ur, X):
    """Fraction of X's energy that lies OFF the (clean) rank-r manifold.
    Bigger for OOD than clean = the distribution shift, made quantitative."""
    with torch.no_grad():
        on = (X @ Ur)                      # coordinates inside the manifold
        on_energy = (on ** 2).sum().item()
        tot_energy = (X ** 2).sum().item() + EPS
    return 1.0 - on_energy / tot_energy


# ------------------------------ setup once -----------------------------------
if A.smoke:
    WIDTH, MODEL = 1024, None
    torch.manual_seed(0)
    clean = torch.randn(8000, WIDTH, device=DEVICE) * 3.0
    ood = torch.randn(6000, WIDTH, device=DEVICE) * 4.0 + 1.5
else:
    MODEL = load_vit(); WIDTH = MODEL[2]
    m, buf, _ = MODEL
    clean = collect(m, buf, WIDTH, A.tokens, train=True,  corrupt=False)
    ood = collect(m, buf, WIDTH, 6000,     train=False, corrupt=True)
mu = clean.mean(0, keepdim=True)
clean, ood = clean - mu, ood - mu
base_clean, base_ood = clean[:4000], ood[:4000]
print(f"activations: clean {tuple(clean.shape)}  ood {tuple(ood.shape)}")

# train the SAEs ONCE (rank-independent); keep their concept directions
saes = []
for seed in range(A.seeds):
    sae, mse = train_sae(WIDTH, clean, seed)
    saes.append((sae, pick_concepts(sae, clean, A.concepts)))
    print(f"seed {seed}: SAE mse {mse:.4f}")

# one PCA at the max rank; smaller manifolds are its leading columns (nested)
with torch.no_grad():
    _, _, Vfull = torch.pca_lowrank(clean, q=max(RANKS))


# --------------------------- sweep over ranks --------------------------------
def agg(records, domain, variant, key):
    vals = np.array([r[key] for r in records if r["domain"] == domain and r["variant"] == variant])
    return float(vals.mean()), 1.96 * float(vals.std()) / math.sqrt(len(vals))

print(f"\n========== MANIFOLD-RANK SWEEP  (model={A.model}, samples/cell={A.seeds*A.concepts}) ==========")
print("naive ignores the manifold -> its CFS is the SAME every row (the reference line).\n")
header = (f"{'rank':>5} {'domain':6} | {'naiveCFS':>9} | {'onmanCFS':>13} | "
          f"{'onman_spec':>10} {'onman_suff':>10} {'onman_offman':>12} | {'OODdrift':>8}")
print(header); print("-" * len(header))
rows = [["rank", "domain", "naive_cfs", "naive_ci", "onman_cfs", "onman_ci",
         "onman_spec", "onman_suff", "onman_offman", "drift"]]
summary = {}                                   # (rank,domain) -> (naive_cfs,naive_ci,onman_cfs,onman_ci)
for rank in RANKS:
    Ur = Vfull[:, :rank].contiguous()
    drift = {"clean": manifold_drift(Ur, base_clean), "ood": manifold_drift(Ur, base_ood)}
    records = []
    for sae, dirs in saes:
        for ci, d in enumerate(dirs):
            others = [o for k, o in enumerate(dirs) if k != ci]
            for domain, base in [("clean", base_clean), ("ood", base_ood)]:
                for kind in ["naive", "onmanifold"]:
                    r = score(kind, d, others, Ur, base)
                    r.update(domain=domain, variant=kind); records.append(r)
    for domain in ["clean", "ood"]:
        ncf, nci = agg(records, domain, "naive", "cfs")
        ocf, oci = agg(records, domain, "onmanifold", "cfs")
        osp, _ = agg(records, domain, "onmanifold", "spec")
        osu, _ = agg(records, domain, "onmanifold", "suff")
        oof, _ = agg(records, domain, "onmanifold", "offman")
        summary[(rank, domain)] = (ncf, nci, ocf, oci)
        print(f"{rank:>5} {domain:6} | {ncf:9.3f} | {ocf:7.3f}±{oci:.3f} | "
              f"{osp:10.3f} {osu:10.3f} {oof:12.3f} | {drift[domain]:8.3f}")
        rows.append([rank, domain, ncf, nci, ocf, oci, osp, osu, oof, drift[domain]])
    print("-" * len(header))

# ------------------------- crossover verdict (data-driven) -------------------
print("\nCROSSOVER (does on-manifold reach naive as rank grows?), 95% CI:")
for domain in ["clean", "ood"]:
    hit = None
    for rank in RANKS:
        ncf, nci, ocf, oci = summary[(rank, domain)]
        if ocf + oci >= ncf - nci:          # CIs touch or on-manifold on top
            hit = rank; break
    msg = f"first rank where on-manifold catches naive = {hit}" if hit else \
          "on-manifold never catches naive in the swept range"
    print(f"  {domain:5}: {msg}")
print("\nOOD drift should be LARGER than clean drift at every rank -> that gap IS the distribution shift.")

csv.writer(open("faith_sae_sweep_results.csv", "w", newline="")).writerows(rows)
print("saved -> faith_sae_sweep_results.csv  (paste the table to your guide)")
