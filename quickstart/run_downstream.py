#!/usr/bin/env python
# =============================================================================
# run_downstream.py  --  the DOWNSTREAM causal-faithfulness test (the real one)
# -----------------------------------------------------------------------------
# run_sweep.py showed naive beats on-manifold on CFS at every rank. But that CFS
# is measured AT the steering layer, ALONG the direction we push -- which naive
# maximises by construction. It never asks the question that actually matters:
#
#     "When I steer at a middle layer and let the rest of the network run,
#      does the change in the model's OUTPUT look like something the model
#      could really produce -- or is it off-manifold garbage?"
#
# That is the faithfulness the paper is about. This script:
#   1. steers concept `d` at the middle block,
#   2. lets the FULL forward pass finish,
#   3. measures the output change dE = E(steered) - E(clean), and splits it into
#        on_eff  = part of dE that lies ON the natural-output manifold (realistic)
#        off_eff = part that lies OFF it                       (unrealistic/garbage)
#   4. reports this for naive vs on-manifold, on clean AND corrupted (OOD) inputs.
#
# Hypothesis: on-manifold keeps off_eff ~ 0 (its output changes stay realistic),
# naive does not -- and the gap WIDENS under OOD. That is the paper's real claim.
#
# Run `python run_downstream.py --smoke` to test plumbing with no downloads.
# For research and educational purposes only.
# =============================================================================
import argparse, csv, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

P = argparse.ArgumentParser()
P.add_argument("--model",    default="vit_large_patch16_224")
P.add_argument("--rank",     type=int, default=512, help="manifold rank for the on-manifold edit")
P.add_argument("--rank_out", type=int, default=64,  help="rank of the natural-OUTPUT manifold")
P.add_argument("--concepts", type=int, default=8)
P.add_argument("--seeds",    type=int, default=3)
P.add_argument("--tokens",   type=int, default=100000)
P.add_argument("--steps",    type=int, default=1000)
P.add_argument("--features", type=int, default=2048)
P.add_argument("--topk",     type=int, default=16)
P.add_argument("--neval",    type=int, default=256, help="images used to measure the output change")
P.add_argument("--strength", type=float, default=4.0)
P.add_argument("--smoke",    action="store_true")
A = P.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-9
print(f"device = {DEVICE} | model = {A.model} | rank = {A.rank} | concepts = {A.concepts} | "
      f"seeds = {A.seeds} | neval = {A.neval} | smoke = {A.smoke}")


# ----------------------- a STEERABLE ViT (hook can edit) ---------------------
def load_vit():
    """Hook the middle block. The hook (a) records activations, and (b) if an
    edit vector is staged in buf['edit'], ADDS it to the patch tokens and returns
    the modified activations -- so the edit propagates through layers 13..24."""
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
            return None                      # leave the forward untouched
        o2 = o.clone()
        o2[:, 1:, :] = o2[:, 1:, :] + e      # add edit to patch tokens, keep CLS
        return o2

    model.blocks[layer].register_forward_hook(hook)
    print(f"loaded {A.model}: width = {model.embed_dim}, steering layer {layer}/{len(model.blocks)}")
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


def load_images(corrupt, n):
    import torchvision
    ds = torchvision.datasets.CIFAR100(root="./data", train=False, download=True,
                                       transform=make_transform(corrupt))
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    xs, c = [], 0
    for x, _ in loader:
        xs.append(x); c += x.shape[0]
        if c >= n:
            break
    return torch.cat(xs)[:n].to(DEVICE)


def collect_layer_acts(model, buf, width, n_tokens):
    """Clean middle-layer patch activations -> train the SAE and the layer-manifold."""
    imgs = load_images(corrupt=False, n=max(600, n_tokens // 150))
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
    """Run the FULL forward (optionally with a staged edit) and return the
    model's output embedding for every image."""
    buf["edit"] = None if edit is None else edit.view(1, 1, -1)
    outs = []
    with torch.no_grad():
        for i in range(0, imgs.shape[0], 64):
            outs.append(model(imgs[i:i + 64]))
    buf["edit"] = None
    return torch.cat(outs)


# ------------------------------- SAE -----------------------------------------
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


def pca_basis(X, q):
    with torch.no_grad():
        _, _, V = torch.pca_lowrank(X, q=min(q, X.shape[0] - 1, X.shape[1]))
    return V


# ---------------------- smoke surrogate (no model) ---------------------------
class SmokeNet:
    """Stand-in 'rest of the network': a fixed random nonlinearity from the
    (pooled) steered activation to an output embedding, so plumbing can be
    tested with no downloads. Real runs use the actual ViT."""
    def __init__(self, width):
        torch.manual_seed(1234)
        self.R = torch.randn(width, width, device=DEVICE) / math.sqrt(width)
    def __call__(self, base, edit):
        x = base if edit is None else base + edit          # (n, width)
        return torch.tanh(x @ self.R)


# ------------------------------- setup ---------------------------------------
if A.smoke:
    WIDTH = 1024
    torch.manual_seed(0)
    A_clean = torch.randn(8000, WIDTH, device=DEVICE) * 3.0
    base_clean = torch.randn(A.neval, WIDTH, device=DEVICE) * 3.0          # 'pooled' per image
    base_ood = torch.randn(A.neval, WIDTH, device=DEVICE) * 4.0 + 1.5
    NET = SmokeNet(WIDTH)
    embed_fn = lambda imgs, edit: NET(imgs, edit)
    img_clean, img_ood = base_clean, base_ood
else:
    model, buf, WIDTH = load_vit()
    A_clean = collect_layer_acts(model, buf, WIDTH, A.tokens)
    img_clean = load_images(corrupt=False, n=A.neval)
    img_ood = load_images(corrupt=True,  n=A.neval)
    embed_fn = lambda imgs, edit: embed(model, buf, imgs, edit)

Ur = pca_basis(A_clean, A.rank)                  # middle-layer manifold (for the edit)
proj = lambda v: Ur @ (Ur.t() @ v)

# baseline outputs (no edit) and the natural-OUTPUT manifold
E0_clean = embed_fn(img_clean, None)
E0_ood = embed_fn(img_ood, None)
Uout = pca_basis(E0_clean - E0_clean.mean(0, keepdim=True), A.rank_out)
proj_out = lambda M: (M @ Uout) @ Uout.t()       # project rows onto output manifold
print(f"setup done: clean outputs {tuple(E0_clean.shape)}, output-manifold rank {Uout.shape[1]}")


def downstream(edit, imgs, E0):
    """Steer with `edit`, finish the forward, split the output change into the
    part ON vs OFF the natural-output manifold."""
    dE = embed_fn(imgs, edit) - E0                       # (n, width)
    on = proj_out(dE)                                    # realistic component
    off = dE - on                                        # unrealistic component
    eff = dE.norm(dim=1).mean().item()
    on_eff = on.norm(dim=1).mean().item()
    off_eff = off.norm(dim=1).mean().item()
    offman = (off.norm(dim=1) / (dE.norm(dim=1) + EPS)).mean().item()
    return dict(eff=eff, on_eff=on_eff, off_eff=off_eff, offman=offman)


# ------------------------------- run -----------------------------------------
records = []
for seed in range(A.seeds):
    sae, mse = train_sae(WIDTH, A_clean, seed)
    dirs = pick_concepts(sae, A_clean, A.concepts)
    print(f"seed {seed}: SAE mse {mse:.4f}, {len(dirs)} concepts")
    for d in dirs:
        for domain, imgs, E0 in [("clean", img_clean, E0_clean), ("ood", img_ood, E0_ood)]:
            for kind in ["naive", "onmanifold"]:
                edit = A.strength * (d if kind == "naive" else proj(d))
                r = downstream(edit, imgs, E0)
                r.update(domain=domain, variant=kind); records.append(r)


def agg(domain, variant, key):
    vals = np.array([r[key] for r in records if r["domain"] == domain and r["variant"] == variant])
    return float(vals.mean()), 1.96 * float(vals.std()) / math.sqrt(len(vals))

print(f"\n====== DOWNSTREAM FAITHFULNESS ({A.model}, samples/cell={A.seeds*A.concepts}) ======")
print("eff = total output change | on_eff = realistic part | off_eff = unrealistic part")
print("offman = fraction of the output change that is OFF the natural-output manifold (LOWER = more faithful)\n")
hdr = f"{'domain':6} {'variant':11} | {'eff':>8} {'on_eff':>8} {'off_eff':>8} | {'offman':>13}"
print(hdr); print("-" * len(hdr))
rows = [["domain", "variant", "eff", "on_eff", "off_eff", "offman", "offman_ci"]]
tbl = {}
for domain in ["clean", "ood"]:
    for variant in ["naive", "onmanifold"]:
        ef, _ = agg(domain, variant, "eff"); on, _ = agg(domain, variant, "on_eff")
        of, _ = agg(domain, variant, "off_eff"); om, omc = agg(domain, variant, "offman")
        tbl[(domain, variant)] = (om, omc)
        print(f"{domain:6} {variant:11} | {ef:8.3f} {on:8.3f} {of:8.3f} | {om:.3f}±{omc:.3f}")
        rows.append([domain, variant, ef, on, of, om, omc])
    print("-" * len(hdr))

# --------------------------- data-driven verdict -----------------------------
print("\nVERDICT (off-manifold fraction of the output change; LOWER = more faithful), 95% CI:")
for domain in ["clean", "ood"]:
    (nm, nci), (om, oci) = tbl[(domain, "naive")], tbl[(domain, "onmanifold")]
    if om + oci < nm - nci:   tag = "on-manifold MORE faithful downstream (CIs disjoint)  <-- supports the thesis"
    elif nm + nci < om - oci: tag = "naive more faithful downstream (CIs disjoint)"
    else:                     tag = "TIE (CIs overlap)"
    print(f"  {domain:5}: onmanifold {om:.3f}[{om-oci:.3f},{om+oci:.3f}] vs naive {nm:.3f}[{nm-nci:.3f},{nm+nci:.3f}] -> {tag}")
gap_c = tbl[("clean", "naive")][0] - tbl[("clean", "onmanifold")][0]
gap_o = tbl[("ood",   "naive")][0] - tbl[("ood",   "onmanifold")][0]
print(f"  naive's off-manifold penalty (naive-onman offman):  clean {gap_c:+.3f}   ood {gap_o:+.3f}"
      f"   -> {'gap WIDENS under OOD (thesis)' if gap_o > gap_c else 'gap does not widen under OOD'}")

csv.writer(open("faith_sae_downstream_results.csv", "w", newline="")).writerows(rows)
print("\nsaved -> faith_sae_downstream_results.csv  (paste the table to your guide)")
