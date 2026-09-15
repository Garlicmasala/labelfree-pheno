"""Plotting helpers: montage grid and metrics bar chart."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _crop_square(arr, size=256):
    h, w = arr.shape[:2]
    top, left = (h - size) // 2, (w - size) // 2
    return arr[top:top + size, left:left + size]


def montage(inputs, targets, preds, stds=None, n: int = 4, path: str = "montage.png",
            crop: int = 256):
    """Grid: rows = fields, columns = BF | target green | predicted green | std (if given)."""
    n = min(n, len(inputs), len(targets), len(preds))
    ncols = 4 if stds else 3
    fig, axes = plt.subplots(n, ncols, figsize=(ncols * 3.2, n * 3.2))
    if n == 1:
        axes = axes[None, :]
    labels = ["Bright-field", "Target (green)", "Virtual stain (green)"] + (["Uncertainty"] if stds else [])
    for i in range(n):
        inp = _crop_square(inputs[i], crop)
        tgt = _crop_square(targets[i], crop)
        prd = _crop_square(preds[i], crop)
        imgs = [inp, tgt, prd]
        if stds:
            imgs.append(_crop_square(stds[i], crop))
        for j, img in enumerate(imgs):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(labels[j], fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def metrics_bar(summary: dict, path: str = "metrics.png"):
    ch = summary.get("channels", {})
    names = list(ch.keys())
    psnr = [ch[c]["psnr"] for c in names]
    ssim = [ch[c]["ssim"] for c in names]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].bar(names, psnr, color="#4C72B0")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_ylim(0, max(psnr) * 1.25)
    for x, v in zip(names, psnr):
        axes[0].text(x, v + 0.3, f"{v:.2f}", ha="center", fontsize=10)
    axes[1].bar(names, ssim, color="#DD8452")
    axes[1].set_ylabel("SSIM")
    axes[1].set_ylim(0, 1.1)
    for x, v in zip(names, ssim):
        axes[1].text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
