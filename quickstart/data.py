# data.py - load real images and collect the ViT's patch activations
import torch, torchvision, torchvision.transforms as T
from utils import N_TOKENS, DEVICE
def collect_activations(model, buf, width):
    tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    ds = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=2)
    chunks, n = [], 0
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(DEVICE))
            chunks.append(buf["a"][:, 1:, :].reshape(-1, width).cpu()); n += chunks[-1].shape[0]
            if n >= N_TOKENS: break
    A = torch.cat(chunks)[:N_TOKENS].to(DEVICE)
    return A - A.mean(0, keepdim=True)
