#!/usr/bin/env python
# =============================================================================
# run_clip_faith.py  --  the REAL-setup faithfulness test (CLIP + real photos +
#                        a semantic, real-image-referenced metric)
# -----------------------------------------------------------------------------
# The quickstart found a null, but on a weak proxy: upscaled 32px CIFAR, the
# vit_large timm model, and a blunt geometric "off-manifold fraction" metric
# that saturated at ~0.89 for everything. This script removes all three
# weaknesses and gives the hypothesis a fair test:
#
#   * MODEL  : CLIP ViT-B/16 (the design brief's backbone).
#   * IMAGES : Flowers-102 -- real, native-resolution photographs.
#   * METRIC : SEMANTIC FAITHFULNESS with a real-image reference. Instead of
#              "does the edit look geometrically realistic", we ask the real
#              question:
#                 does steering a concept move the model's OUTPUT the same way
#                 that REAL photos containing that concept move it?
#              i.e. cosine( output-change from steering ,
#                          output-change between real high-concept and
#                          low-concept photos ).
#              Naive steering cannot win this by construction.
#
# We report semantic faithfulness, specificity, sufficiency (and their harmonic
# mean) for naive vs on-manifold, on clean photos and corrupted (OOD) photos.
#
# Run `python run_clip_faith.py --smoke` to test plumbing with no downloads.
# For research and educational purposes only.
# =============================================================================
import argparse, csv, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

P = argparse.ArgumentParser()
P.add_argument("--model",    default="vit_base_patch16_clip_224.openai", help="timm CLIP ViT-B/16")
P.add_argument("--rank",     type=int, default=256, help="manifold rank for the on-manifold edit")
P.add_argument("--concepts", type=int, default=8)
P.add_argument("--seeds",    type=int, default=3)
P.add_argument("--tokens",   type=int, default=80000)
P.add_argument("--steps",    type=int, default=1000)
P.add_argument("--features", type=int, default=2048)
P.add_argument("--topk",     type=int, default=16)
P.add_argument("--neval",    type=int, default=512, help="photos used to measure effects")
P.add_argument("--strength", type=float, default=4.0)
P.add_argument("--smoke",    action="store_true")
A = P.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-9
# CLIP's own normalisation constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
print(f"device = {DEVICE} | model = {A.model} | rank = {A.rank} | concepts = {A.concepts} | "
      f"seeds = {A.seeds} | neval = {A.neval} | smoke = {A.smoke}")


# ----------------------- steerable CLIP ViT -----------------------
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
    """Real native-resolution flower photographs (no labels needed)."""
    import torchvision
    ds = torchvision.datasets.Flowers102(root="./data_flowers", split="test",
                                         download=True, transform=make_transform(corrupt))
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    xs, c = [], 0
    for x, _ in loader:
        xs.append(x); c += x.shape[0]
        if c >= n:
            break
    return torch.cat(xs)[:n]                 # kept on CPU; moved to GPU per batch


# --------------- forward passes: patch activations & output embeddings -------
def forward_collect(model, buf, imgs, want_patches, edit):
    """Run the full forward (optionally steered). Returns output embeddings E,
    and (if want_patches) each image's mean middle-layer patch activation."""
    buf["edit"] = None if edit is None else edit.view(1, 1, -1)
    Es, Ps = [], []
    with torch.no_grad():
        for i in range(0, imgs.shape[0], 64):
            x = imgs[i:i + 64].to(DEVICE)
            Es.append(model(x))
            if want_patches:
                Ps.append(buf["a"][:, 1:, :].mean(1))      # per-image mean patch act
    buf["edit"] = None
    E = torch.cat(Es)
    Pp = torch.cat(Ps) if want_patches else None
    return E, Pp


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
    return [(lambda d: d / (d.norm() + EPS))(sae.dec.weight[:, int(j)].detach())
            for j in Z.mean(0).topk(n).indices]


def pca_basis(X, q):
    with torch.no_grad():
        _, _, V = torch.pca_lowrank(X, q=min(q, X.shape[0] - 1, X.shape[1]))
    return V


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + EPS))


# ---------------------------- setup ------------------------------------------
if A.smoke:
    WIDTH = 768
    torch.manual_seed(0)
    A_clean = torch.randn(8000, WIDTH, device=DEVICE) * 3.0
    R = torch.randn(WIDTH, WIDTH, device=DEVICE) / math.sqrt(WIDTH)   # fake 'rest of net'
    P_clean = torch.randn(A.neval, WIDTH, device=DEVICE) * 3.0        # per-image patch acts
    def embed_smoke(imgs_unused, P, edit):
        x = P if edit is None else P + edit
        return torch.tanh(x @ R)
    E_clean = embed_smoke(None, P_clean, None)
    P_ood = torch.randn(A.neval, WIDTH, device=DEVICE) * 4.0 + 1.0
    E_ood0 = embed_smoke(None, P_ood, None)
    def steer_embed(domain, edit):
        P = P_clean if domain == "clean" else P_ood
        return embed_smoke(None, P, edit)
else:
    model, buf, WIDTH = load_vit()
    sae_imgs = load_photos(corrupt=False, n=max(700, A.tokens // 100))
    A_clean = collect_tokens(model, buf, sae_imgs, WIDTH, A.tokens)
    photos_clean = load_photos(corrupt=False, n=A.neval)
    photos_ood = load_photos(corrupt=True,  n=A.neval)
    E_clean, P_clean = forward_collect(model, buf, photos_clean, want_patches=True, edit=None)
    E_ood0, _ = forward_collect(model, buf, photos_ood, want_patches=False, edit=None)
    def steer_embed(domain, edit):
        imgs = photos_clean if domain == "clean" else photos_ood
        E, _ = forward_collect(model, buf, imgs, want_patches=False, edit=edit)
        return E

Ur = pca_basis(A_clean, A.rank)
proj = lambda v: Ur @ (Ur.t() @ v)
P_clean_c = P_clean - A_clean.mean(0, keepdim=True) if not A.smoke else P_clean   # center like A_clean
E0 = {"clean": E_clean, "ood": E_ood0}
print(f"setup done: outputs clean {tuple(E_clean.shape)}, neval {A.neval}")


def real_reference(d):
    """ΔE_real: how real HIGH-concept photos differ from LOW-concept photos in
    the output embedding -- the ground-truth semantic effect of the concept."""
    a = P_clean_c @ d                                 # per-photo concept activation
    k = max(8, a.shape[0] // 4)
    hi = torch.topk(a, k).indices
    lo = torch.topk(-a, k).indices
    return E_clean[hi].mean(0) - E_clean[lo].mean(0)


# ---------------------------- run --------------------------------------------
records = []
for seed in range(A.seeds):
    sae, mse = train_sae(WIDTH, A_clean, seed)
    dirs = pick_concepts(sae, A_clean, A.concepts)
    refs = [real_reference(d) for d in dirs]          # real semantic effect per concept
    print(f"seed {seed}: SAE mse {mse:.4f}, {len(dirs)} concepts")
    for ci, d in enumerate(dirs):
        ref = refs[ci]
        others = [refs[k] for k in range(len(dirs)) if k != ci]
        for domain in ["clean", "ood"]:
            for kind in ["naive", "onmanifold"]:
                edit = A.strength * (d if kind == "naive" else proj(d))
                dE = (steer_embed(domain, edit) - E0[domain]).mean(0)   # steer's output effect
                faith = cos(dE, ref)                                    # vs real concept photos
                spec = max(0.0, 1.0 - float(np.mean([abs(cos(dE, o)) for o in others])))
                suff = float(np.clip(dE.norm().item() / (E_clean.std(0).norm().item() + EPS), 0, 1))
                fc = max(0.0, faith)
                cfs = 3.0 / (1/(fc+EPS) + 1/(spec+EPS) + 1/(suff+EPS))
                records.append(dict(domain=domain, variant=kind,
                                    faith=faith, spec=spec, suff=suff, cfs=cfs))


def agg(domain, variant, key):
    vals = np.array([r[key] for r in records if r["domain"] == domain and r["variant"] == variant])
    return float(vals.mean()), 1.96 * float(vals.std()) / math.sqrt(len(vals))

print(f"\n====== CLIP SEMANTIC FAITHFULNESS ({A.model}, samples/cell={A.seeds*A.concepts}) ======")
print("faith = cosine(steer effect, real concept-photo effect)  [HIGHER = more faithful, max 1.0]")
print("spec  = avoids other concepts | suff = effect size | semCFS = harmonic mean\n")
hdr = f"{'domain':6} {'variant':11} | {'faith':>13} {'spec':>12} {'suff':>12} {'semCFS':>12}"
print(hdr); print("-" * len(hdr))
rows = [["domain", "variant", "faith", "faith_ci", "spec", "spec_ci", "suff", "suff_ci", "cfs", "cfs_ci"]]
tbl = {}
for domain in ["clean", "ood"]:
    for variant in ["naive", "onmanifold"]:
        fa, fac = agg(domain, variant, "faith"); sp, spc = agg(domain, variant, "spec")
        su, suc = agg(domain, variant, "suff");  cf, cfc = agg(domain, variant, "cfs")
        tbl[(domain, variant)] = (fa, fac)
        print(f"{domain:6} {variant:11} | {fa:.3f}±{fac:.3f} {sp:.3f}±{spc:.3f} {su:.3f}±{suc:.3f} {cf:.3f}±{cfc:.3f}")
        rows.append([domain, variant, fa, fac, sp, spc, su, suc, cf, cfc])
    print("-" * len(hdr))

print("\nVERDICT (semantic faithfulness; HIGHER = better), 95% CI:")
for domain in ["clean", "ood"]:
    (om, oc), (nm, nc) = tbl[(domain, "onmanifold")], tbl[(domain, "naive")]
    if om - oc > nm + nc:   tag = "on-manifold MORE faithful (CIs disjoint)  <-- SUPPORTS the thesis"
    elif nm - nc > om + oc: tag = "naive more faithful (CIs disjoint)"
    else:                   tag = "TIE (CIs overlap)"
    print(f"  {domain:5}: onmanifold {om:.3f}[{om-oc:.3f},{om+oc:.3f}] vs naive {nm:.3f}[{nm-nc:.3f},{nm+nc:.3f}] -> {tag}")
gc = tbl[("clean", "onmanifold")][0] - tbl[("clean", "naive")][0]
go = tbl[("ood",   "onmanifold")][0] - tbl[("ood",   "naive")][0]
print(f"  on-manifold's faithfulness advantage (onman-naive):  clean {gc:+.3f}   ood {go:+.3f}"
      f"   -> {'advantage GROWS under OOD (thesis)' if go > gc else 'advantage does not grow under OOD'}")

csv.writer(open("faith_sae_clip_results.csv", "w", newline="")).writerows(rows)
print("\nsaved -> faith_sae_clip_results.csv  (paste the table + verdict to your guide)")
