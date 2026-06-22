# train.py - train the TopK SAE on the collected activations
import torch, torch.nn.functional as F
from utils import SAE_STEPS
def train_sae(sae, A):
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for step in range(SAE_STEPS):
        idx = torch.randint(0, A.shape[0], (4096,), device=A.device)
        xhat, _ = sae(A[idx]); loss = F.mse_loss(xhat, A[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0: print(f"  SAE step {step:4d}  mse {loss.item():.4f}")
    return sae
