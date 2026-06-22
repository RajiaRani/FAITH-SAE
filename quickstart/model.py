# model.py - frozen ViT backbone + TopK Sparse Autoencoder
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from utils import MODEL_NAME, LAYER, DEVICE
def load_vit():
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).eval().to(DEVICE)
    for p in model.parameters(): p.requires_grad_(False)
    buf = {}
    model.blocks[LAYER].register_forward_hook(lambda m, i, o: buf.__setitem__("a", o))
    return model, buf, model.embed_dim
class TopKSAE(nn.Module):
    def __init__(self, d, nf, k):
        super().__init__(); self.k = k
        self.enc = nn.Linear(d, nf); self.dec = nn.Linear(nf, d, bias=False)
    def forward(self, x):
        z = F.relu(self.enc(x)); val, idx = z.topk(self.k, dim=-1)
        return self.dec(torch.zeros_like(z).scatter_(-1, idx, val)), z
