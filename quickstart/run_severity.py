#!/usr/bin/env python
# =============================================================================
# run_severity.py  --  the OOD SEVERITY sweep (RQ3, the paper's headline figure)
# -----------------------------------------------------------------------------
# Every test so far used ONE mild corruption. But the paper's headline question
# is "does faithfulness survive distribution shift?" -- which needs the whole
# severity ladder, clean -> light -> ... -> severe (the ImageNet-C idea).
#
# This script corrupts the input at increasing severity, steers each concept,
# finishes the full forward, and measures the downstream off-manifold fraction
# (LOWER = more faithful) for naive vs on-manifold at every severity. The two
# curves it draws ARE figure 1 of the paper -- but with real numbers.
#
# It reuses run_downstream's machinery: steer at the middle block, let the
# network finish, split the output change into realistic vs off-manifold parts.
#
# Run `python run_severity.py --smoke` to test plumbing with no downloads.
# For research and educational purposes only.
# =============================================================================
import argparse, csv, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

P = argparse.ArgumentParser()
P.add_argument("--model",    default="vit_large_patch16_224")
P.add_argument("--rank",     type=int, default=512)
P.add_argument("--rank_out", type=int, default=64)
P.add_argument("--concepts", type=int, default=8)
P.add_argument("--seeds",    type=int, default=3)
P.add_argument("--tokens",   type=int, default=100000)
P.add_argument("--steps",    type=int, default=1000)
P.add_argument("--features", type=int, default=2048)
P.add_argument("--topk",     type=int, default=16)
P.add_argument("--neval",    type=int, default=192)
P.add_argument("--strength", type=float, default=4.0)
P.add_argument("--smoke",    action="store_true")
A = P.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-9
# severity ladder: (name, blur sigma, noise std).  clean -> severe.
SEVERITY = [("clean", 0.0, 0.00), ("s1", 0.5, 0.03), ("s2", 1.0, 0.06),
            ("s3", 2.0, 0.10), ("s4", 3.0, 0.16), ("s5", 4.0, 0.24)]
print(f"device = {DEVICE} | model = {A.model} | severities = {[s[0] for s in SEVERITY]} | "
      f"concepts = {A.concepts} | seeds = {A.seeds} | smoke = {A.smoke}")


def load_vit():
    import timm
    model = timm.create_model(A.model, pretrained=True, num_classes=0).eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    layer = len(model.blocks) // 2
    buf = {"edit": None}

    def hook(m, i, o):
        buf["a"] = o
        e = buf["edit"]
        if e is None:
            return None
        o2 = o.clone(); o2[:, 1:, :] = o2[:, 1:, :] + e
        return o2

    model.blocks[layer].register_forward_hook(hook)
    print(f"loaded {A.model}: width = {model.embed_dim}, steering layer {layer}/{len(model.blocks)}")
    return model, buf, model.embed_dim


def make_transform(sigma, noise):
    import torchvision.transforms as T
    steps = [T.Resize(224), T.CenterCrop(224)]
    if sigma > 0:
        ks = 2 * max(1, int(round(sigma))) + 1          # kernel grows with sigma
        steps.append(T.GaussianBlur(kernel_size=ks, sigma=sigma))
    steps.append(T.ToTensor())
    if noise > 0:
        steps.append(T.Lambda(lambda t, _n=noise: (t + _n * torch.randn_like(t)).clamp(0, 1)))
    steps.append(T.Normalize([0.5] * 3, [0.5] * 3))
    return T.Compose(steps)


def load_images(sigma, noise, n):
    import torchvision
    ds = torchvision.datasets.CIFAR100(root="./data", train=False, download=True,
                                       transform=make_transform(sigma, noise))
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    xs, c = [], 0
    for x, _ in loader:
        xs.append(x); c += x.shape[0]
        if c >= n:
            break
    return torch.cat(xs)[:n].to(DEVICE)


def collect_layer_acts(model, buf, width, n_tokens):
    imgs = load_images(0.0, 0.0, max(600, n_tokens // 150))
    chunks, n = [], 0
    buf["edit"] = None
    with torch.no_grad():
        for i in range(0, imgs.shape[0], 64):
            model(imgs[i:i + 64])
            chunks.append(buf["a"][:, 1:, :].reshape(-1, width).cpu()); n += chunks[-1].shape[0]
            if n >= n_tokens:
                break
    A_ = torch.cat(chunks)[:n_tokens].to(DEVICE)
    return A_ - A_.mean(0, keepdim=True)


def embed(model, buf, imgs, edit):
    buf["edit"] = None if edit is None else edit.view(1, 1, -1)
    outs = []
    with torch.no_grad():
        for i in range(0, imgs.shape[0], 64):
            outs.append(model(imgs[i:i + 64]))
    buf["edit"] = None
    return torch.cat(outs)


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
    return [(lambda d: d / (d.norm() + EPS))(sae.dec.weight[:, int(j)].detach())
            for j in Z.mean(0).topk(n).indices]


def pca_basis(X, q):
    with torch.no_grad():
        _, _, V = torch.pca_lowrank(X, q=min(q, X.shape[0] - 1, X.shape[1]))
    return V


class SmokeNet:
    def __init__(self, width):
        torch.manual_seed(1234); self.R = torch.randn(width, width, device=DEVICE) / math.sqrt(width)
    def __call__(self, base, edit):
        x = base if edit is None else base + edit
        return torch.tanh(x @ self.R)


# ------------------------------- setup ---------------------------------------
if A.smoke:
    WIDTH = 1024
    torch.manual_seed(0)
    A_clean = torch.randn(8000, WIDTH, device=DEVICE) * 3.0
    NET = SmokeNet(WIDTH)
    base0 = torch.randn(A.neval, WIDTH, device=DEVICE) * 3.0
    # simulate severity by adding increasing noise to the synthetic 'images'
    images = {name: base0 + (sig + noi) * torch.randn(A.neval, WIDTH, device=DEVICE)
              for name, sig, noi in SEVERITY}
    embed_fn = lambda imgs, edit: NET(imgs, edit)
else:
    model, buf, WIDTH = load_vit()
    A_clean = collect_layer_acts(model, buf, WIDTH, A.tokens)
    images = {name: load_images(sig, noi, A.neval) for name, sig, noi in SEVERITY}
    embed_fn = lambda imgs, edit: embed(model, buf, imgs, edit)

Ur = pca_basis(A_clean, A.rank)
proj = lambda v: Ur @ (Ur.t() @ v)
E0 = {name: embed_fn(images[name], None) for name, _, _ in SEVERITY}     # per-severity baseline
Uout = pca_basis(E0["clean"] - E0["clean"].mean(0, keepdim=True), A.rank_out)  # CLEAN output manifold
proj_out = lambda M: (M @ Uout) @ Uout.t()
print(f"setup done: output-manifold rank {Uout.shape[1]}, neval {A.neval}")


def downstream(edit, name):
    dE = embed_fn(images[name], edit) - E0[name]
    off = dE - proj_out(dE)
    return dict(eff=dE.norm(dim=1).mean().item(),
                offman=(off.norm(dim=1) / (dE.norm(dim=1) + EPS)).mean().item())


# ------------------------------- run -----------------------------------------
records = []
for seed in range(A.seeds):
    sae, mse = train_sae(WIDTH, A_clean, seed)
    dirs = pick_concepts(sae, A_clean, A.concepts)
    print(f"seed {seed}: SAE mse {mse:.4f}, {len(dirs)} concepts")
    for d in dirs:
        for name, _, _ in SEVERITY:
            for kind in ["naive", "onmanifold"]:
                edit = A.strength * (d if kind == "naive" else proj(d))
                r = downstream(edit, name); r.update(sev=name, variant=kind); records.append(r)


def agg(name, variant, key):
    vals = np.array([r[key] for r in records if r["sev"] == name and r["variant"] == variant])
    return float(vals.mean()), 1.96 * float(vals.std()) / math.sqrt(len(vals))

print(f"\n====== OOD SEVERITY SWEEP ({A.model}, samples/cell={A.seeds*A.concepts}) ======")
print("offman = off-manifold fraction of the output change (LOWER = more faithful)\n")
hdr = f"{'severity':9} | {'naive_offman':>14} {'onman_offman':>14} | {'naive_eff':>9} {'onman_eff':>9}"
print(hdr); print("-" * len(hdr))
rows = [["severity", "naive_offman", "naive_ci", "onman_offman", "onman_ci", "naive_eff", "onman_eff"]]
curve = {"sev": [], "naive": [], "naive_ci": [], "onman": [], "onman_ci": []}
for name, _, _ in SEVERITY:
    no, noc = agg(name, "naive", "offman"); oo, ooc = agg(name, "onmanifold", "offman")
    ne, _ = agg(name, "naive", "eff");     oe, _ = agg(name, "onmanifold", "eff")
    print(f"{name:9} | {no:.3f}±{noc:.3f}  {oo:.3f}±{ooc:.3f} | {ne:9.3f} {oe:9.3f}")
    rows.append([name, no, noc, oo, ooc, ne, oe])
    curve["sev"].append(name); curve["naive"].append(no); curve["naive_ci"].append(noc)
    curve["onman"].append(oo); curve["onman_ci"].append(ooc)

# slope clean->severe = the "collapse" rate; gap = method separation
d_naive = curve["naive"][-1] - curve["naive"][0]
d_onman = curve["onman"][-1] - curve["onman"][0]
sep = float(np.mean([abs(a - b) for a, b in zip(curve["naive"], curve["onman"])]))
print("\nHEADLINE (RQ3):")
print(f"  faithfulness collapse clean->s5 (offman rise):  naive {d_naive:+.3f}   onman {d_onman:+.3f}")
print(f"  mean naive-vs-onman separation across severities: {sep:.3f}  "
      f"-> {'methods DIFFER' if sep > 0.02 else 'methods essentially IDENTICAL at every severity'}")

csv.writer(open("faith_sae_severity_results.csv", "w", newline="")).writerows(rows)
print("saved -> faith_sae_severity_results.csv")

# optional: render the real Figure 1
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = list(range(len(SEVERITY)))
    plt.figure(figsize=(6, 4))
    plt.errorbar(x, curve["naive"], yerr=curve["naive_ci"], marker="o", capsize=3, label="naive steering")
    plt.errorbar(x, curve["onman"], yerr=curve["onman_ci"], marker="s", capsize=3, label="on-manifold steering")
    plt.xticks(x, curve["sev"]); plt.xlabel("distribution-shift severity")
    plt.ylabel("downstream off-manifold fraction\n(lower = more faithful)")
    plt.title(f"FAITH-SAE: faithfulness vs OOD severity ({A.model})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("fig1_cfs_ood_sweep_REAL.png", dpi=200)
    print("saved -> fig1_cfs_ood_sweep_REAL.png  (your real Figure 1)")
except Exception as e:
    print(f"(figure skipped: {e}) -- the CSV has everything to plot later")
print("\npaste the table + HEADLINE lines to your guide")
