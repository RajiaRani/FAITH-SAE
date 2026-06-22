# run.py - the main pipeline: ties the other files together
import csv
from utils import N_FEATURES, TOPK, DEVICE
from model import load_vit, TopKSAE
from data import collect_activations
from train import train_sae
from evaluate import manifold_basis, pick_concept, score
print("device =", DEVICE)
model, buf, width = load_vit(); print("loaded ViT, width =", width)
A = collect_activations(model, buf, width); print("activations:", tuple(A.shape))
sae = train_sae(TopKSAE(width, N_FEATURES, TOPK).to(DEVICE), A)
Ur = manifold_basis(A); d = pick_concept(sae, A); base = A[:4000]
print("\nvariant       mono   spec   suff    CFS   off-manifold")
rows = [["variant", "mono", "spec", "suff", "cfs", "offmanifold"]]
for kind in ["naive", "onmanifold"]:
    mo, sp, su, cf, re = score(kind, d, Ur, base)
    print(f"{kind:11s}  {mo:.3f}  {sp:.3f}  {su:.3f}  {cf:.3f}   {re:.3f}")
    rows.append([kind, mo, sp, su, cf, re])
csv.writer(open("faith_sae_results.csv", "w", newline="")).writerows(rows)
print("\nsaved -> faith_sae_results.csv")
print("RESULT: on-manifold CFS > naive CFS with off-manifold ~ 0 means it holds on your real ViT.")
