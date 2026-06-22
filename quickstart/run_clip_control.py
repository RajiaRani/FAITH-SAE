#!/usr/bin/env python
# =============================================================================
# run_clip_control.py  --  positive control + the field-standard steering method
# -----------------------------------------------------------------------------
# run_clip_faith.py found faithfulness ~ 0 for SAE-concept steering. But a metric
# that reads ~0 might be measuring "no faithful effect exists" OR might just be a
# blunt ruler (the offman metric fooled us this way once). This script settles it
# with a POSITIVE CONTROL that is also a real experiment:
#
#   We add a third steering direction -- DIFFERENCE-OF-MEANS (the standard
#   "TCAV"-style concept vector): the layer-6 direction along which real
#   high-concept photos actually differ from low-concept photos.
#
#   * If diff-of-means scores HIGH faith while the SAE direction scores ~0,
#     the metric WORKS, and the finding is "SAE decoder directions are not the
#     faithful concept axis; difference-of-means directions are."
#   * If diff-of-means ALSO scores ~0, the metric is fine and the finding is
#     "uniform additive middle-layer steering is unfaithful for ANY direction."
#
# Three variants compared at matched strength on clean + OOD photos:
#   naive       = SAE decoder column d
#   onmanifold  = projection of d onto the data manifold
#   diffmeans   = mean(high-concept patch acts) - mean(low-concept patch acts)
#
# Run `python run_clip_control.py --smoke` to test plumbing with no downloads.
# For research and educational purposes only.
# =============================================================================
import argparse, csv, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

P = argparse.ArgumentParser()
P.add_argument("--model",    default="vit_base_patch16_clip_224.openai")
P.add_argument("--rank",     type=int, default=256)
P.add_argument("--concepts", type=int, default=8)
P.add_argument("--seeds",    type=int, default=3)
P.add_argument("--tokens",   type=int, default=80000)
P.add_argument("--steps",    type=int, default=1000)
P.add_argument("--features", type=int, default=2048)
P.add_argument("--topk",     type=int, default=16)
P.add_argument("--neval",    type=int, default=512)
P.add_argument("--strength", type=float, default=4.0)
P.add_argument("--smoke",    action="store_true")
A = P.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-9
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
VARIANTS = ["naive", "onmanifold", "diffmeans"]
print(f"device = {DEVICE} | model = {A.model} | rank = {A.rank} | concepts = {A.concepts} | "
      f"seeds = {A.seeds} | neval = {A.neval} | smoke = {A.smoke}")


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


def make_transform(corrupt):
    import torchvision.transforms as T
    steps = [T.Resize(224), T.CenterCrop(224)]
    if corrupt:
        steps.append(T.GaussianBlur(kernel_size=9, sigma=3.0))
    steps.append(T.ToTensor())
    if corrupt:
        steps.append(T.Lambda(lambda t: (t + 0.15 * torch.randn_like(t)).clamp(0, 1)))
    steps.append(T.Normalize(CLIP_MEAN, CLIP_STD))
    return T.Compose(steps)


def load_photos(corrupt, n):
    import torchvision
    ds = torchvision.datasets.Flowers102(root="./data_flowers", split="test",
                                         download=True, transform=make_transform(corrupt))
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    xs, c = [], 0
    for x, _ in loader:
        xs.append(x); c += x.shape[0]
        if c >= n:
            break
    return torch.cat(xs)[:n]


def forward_collect(model, buf, imgs, want_patches, edit):
    buf["edit"] = None if edit is None else edit.view(1, 1, -1)
    Es, Ps = [], []
    with torch.no_grad():
        for i in range(0, imgs.shape[0], 64):
            x = imgs[i:i + 64].to(DEVICE)
            Es.append(model(x))
            if want_patches:
                Ps.append(buf["a"][:, 1:, :].mean(1))
    buf["edit"] = None
    return torch.cat(Es), (torch.cat(Ps) if want_patches else None)


def collect_tokens(model, buf, imgs, width, n_tokens):
    chunks, n = [], 0
    buf["edit"] = None
    with torch.no_grad():
        for i in range(0, imgs.shape[0], 64):
            model(imgs[i:i + 64].to(DEVICE))
            chunks.append(buf["a"][:, 1:, :].reshape(-1, width).cpu()); n += chunks[-1].shape[0]
            if n >= n_tokens:
                break
    A_ = torch.cat(chunks)[:n_tokens].to(DEVICE)
    return A_ - A_.mean(0, keepdim=True)


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


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + EPS))


# ------------------------------- setup ---------------------------------------
if A.smoke:
    WIDTH = 768
    torch.manual_seed(0)
    A_clean = torch.randn(8000, WIDTH, device=DEVICE) * 3.0
    R = torch.randn(WIDTH, WIDTH, device=DEVICE) / math.sqrt(WIDTH)
    P_clean = torch.randn(A.neval, WIDTH, device=DEVICE) * 3.0
    embed_s = lambda Pp, edit: torch.tanh((Pp if edit is None else Pp + edit) @ R)
    E_clean = embed_s(P_clean, None)
    P_ood = torch.randn(A.neval, WIDTH, device=DEVICE) * 4.0 + 1.0
    E_ood0 = embed_s(P_ood, None)
    steer_embed = lambda domain, edit: embed_s(P_clean if domain == "clean" else P_ood, edit)
    Pmean = torch.zeros(WIDTH, device=DEVICE)
else:
    model, buf, WIDTH = load_vit()
    sae_imgs = load_photos(corrupt=False, n=max(700, A.tokens // 100))
    A_clean = collect_tokens(model, buf, sae_imgs, WIDTH, A.tokens)
    photos_clean = load_photos(corrupt=False, n=A.neval)
    photos_ood = load_photos(corrupt=True,  n=A.neval)
    E_clean, P_clean = forward_collect(model, buf, photos_clean, True, None)
    E_ood0, _ = forward_collect(model, buf, photos_ood, False, None)
    steer_embed = lambda domain, edit: forward_collect(
        model, buf, photos_clean if domain == "clean" else photos_ood, False, edit)[0]
    Pmean = torch.zeros(WIDTH, device=DEVICE)

Ur = pca_basis(A_clean, A.rank)
proj = lambda v: Ur @ (Ur.t() @ v)
amean = A_clean.mean(0, keepdim=True)
P_clean_c = P_clean - amean if not A.smoke else P_clean
E0 = {"clean": E_clean, "ood": E_ood0}
print(f"setup done: outputs clean {tuple(E_clean.shape)}, neval {A.neval}")


# ---------------------------- run --------------------------------------------
records = []
for seed in range(A.seeds):
    sae, mse = train_sae(WIDTH, A_clean, seed)
    dirs = pick_concepts(sae, A_clean, A.concepts)
    print(f"seed {seed}: SAE mse {mse:.4f}, {len(dirs)} concepts")
    for d in dirs:
        a = P_clean_c @ d                                  # per-photo concept activation
        k = max(8, a.shape[0] // 4)
        hi = torch.topk(a, k).indices
        lo = torch.topk(-a, k).indices
        ref = E_clean[hi].mean(0) - E_clean[lo].mean(0)    # real OUTPUT effect of the concept
        dmean = P_clean_c[hi].mean(0) - P_clean_c[lo].mean(0)   # diff-of-means (TCAV) direction
        dmean = dmean / (dmean.norm() + EPS)
        edit_dirs = {"naive": d, "onmanifold": proj(d), "diffmeans": dmean}
        for domain in ["clean", "ood"]:
            for kind in VARIANTS:
                dE = (steer_embed(domain, A.strength * edit_dirs[kind]) - E0[domain]).mean(0)
                records.append(dict(domain=domain, variant=kind, faith=cos(dE, ref)))


def agg(domain, variant):
    vals = np.array([r["faith"] for r in records if r["domain"] == domain and r["variant"] == variant])
    return float(vals.mean()), 1.96 * float(vals.std()) / math.sqrt(len(vals))

print(f"\n====== POSITIVE CONTROL: semantic faithfulness by steering direction ({A.model}) ======")
print("faith = cosine(steer effect, real concept-photo effect)  [HIGHER = more faithful, max 1.0]\n")
hdr = f"{'domain':6} | {'naive(SAE)':>14} {'onmanifold':>14} {'diffmeans(TCAV)':>16}"
print(hdr); print("-" * len(hdr))
rows = [["domain", "naive", "naive_ci", "onmanifold", "onman_ci", "diffmeans", "diffmeans_ci"]]
tbl = {}
for domain in ["clean", "ood"]:
    cells = {v: agg(domain, v) for v in VARIANTS}
    tbl[domain] = cells
    print(f"{domain:6} | " + " ".join(f"{cells[v][0]:+.3f}±{cells[v][1]:.3f}".rjust(14 if v!='diffmeans' else 16)
                                       for v in VARIANTS))
    rows.append([domain, cells["naive"][0], cells["naive"][1], cells["onmanifold"][0],
                 cells["onmanifold"][1], cells["diffmeans"][0], cells["diffmeans"][1]])

print("\nVERDICT:")
for domain in ["clean", "ood"]:
    dm, dmc = tbl[domain]["diffmeans"]; na, nac = tbl[domain]["naive"]
    if dm - dmc > 0.1:
        ctrl = "metric WORKS (diff-of-means is faithful) -> SAE directions are genuinely UNfaithful"
    elif dm - dmc > na + nac:
        ctrl = "diff-of-means beats SAE (CIs disjoint) -> SAE directions less faithful than TCAV"
    else:
        ctrl = "diff-of-means also ~0 -> NO additive middle-layer steer is faithful here"
    print(f"  {domain:5}: diffmeans {dm:+.3f}  vs  naive(SAE) {na:+.3f}  ->  {ctrl}")

csv.writer(open("faith_sae_control_results.csv", "w", newline="")).writerows(rows)
print("\nsaved -> faith_sae_control_results.csv  (paste the table + verdict to your guide)")
