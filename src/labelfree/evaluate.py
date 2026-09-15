"""Evaluation: PSNR / SSIM over a loader (used during training and final eval)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_np(t):
    return t.detach().cpu().numpy()


@torch.no_grad()
def eval_metrics(model, loader, device) -> tuple[float, float]:
    """Pooled PSNR / SSIM across channels and samples."""
    model.eval()
    psnrs, ssims = [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        p = _to_np(pred)
        t = _to_np(y)
        for c in range(p.shape[1]):
            p_c = np.clip(p[0, c], 0, 1)
            t_c = np.clip(t[0, c], 0, 1)
            psnrs.append(peak_signal_noise_ratio(t_c, p_c, data_range=1.0))
            ssims.append(structural_similarity(t_c, p_c, data_range=1.0))
    model.train()
    return float(np.mean(psnrs)), float(np.mean(ssims))


@torch.no_grad()
def eval_per_channel(model, loader, device, ch_names=("green", "red")):
    """Per-channel PSNR/SSIM lists."""
    model.eval()
    acc = {c: {"psnr": [], "ssim": []} for c in ch_names}
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        p = _to_np(pred)
        t = _to_np(y)
        for ci, c in enumerate(ch_names):
            p_c = np.clip(p[0, ci], 0, 1)
            t_c = np.clip(t[0, ci], 0, 1)
            acc[c]["psnr"].append(peak_signal_noise_ratio(t_c, p_c, data_range=1.0))
            acc[c]["ssim"].append(structural_similarity(t_c, p_c, data_range=1.0))
    model.train()
    return {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in acc.items()}
