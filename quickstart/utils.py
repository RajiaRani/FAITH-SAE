# utils.py - all your settings in one place (edit these)
import torch
MODEL_NAME = "vit_base_patch16_224"   # your ViT: vit_small_/vit_base_/vit_large_patch16_224
LAYER, N_TOKENS = 6, 100000
N_FEATURES, TOPK, SAE_STEPS = 2048, 16, 400
STRENGTHS, PROJ_RANK = [0.0, 1.0, 2.0, 4.0], 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
