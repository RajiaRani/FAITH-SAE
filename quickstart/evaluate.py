# evaluate.py - on-manifold subspace, the two steering methods, the CFS
import torch, numpy as np
from scipy.stats import spearmanr
from utils import STRENGTHS, PROJ_RANK
def manifold_basis(A):
    with torch.no_grad(): _, _, V = torch.pca_lowrank(A, q=PROJ_RANK)
    return V
def pick_concept(sae, A):
    with torch.no_grad(): _, Z = sae(A[:8000])
    d = sae.dec.weight[:, int(Z.mean(0).argmax())].detach()
    return d / d.norm()
def score(kind, d, Ur, base):
    proj = lambda v: Ur @ (Ur.t() @ v)
    base_std = (base @ d).std().item() + 1e-6
    means = [((base + s * (d if kind == "naive" else proj(d))) @ d).mean().item() for s in STRENGTHS]
    mono = max(0.0, spearmanr(STRENGTHS, means).correlation)
    suff = float(np.clip((means[-1] - means[0]) / base_std / 4, 0, 1))
    e = STRENGTHS[-1] * (d if kind == "naive" else proj(d))
    resid = 1 - (proj(e).norm() / (e.norm() + 1e-9)).item(); spec = max(0.0, 1 - resid)
    return mono, spec, suff, 3 / (1/(mono+1e-6) + 1/(spec+1e-6) + 1/(suff+1e-6)), resid
