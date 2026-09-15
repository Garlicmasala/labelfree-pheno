"""Losses: L1, Fourier-spectrum, perceptual (VGG), and adversarial helpers."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_loss(pred, target):
    return F.l1_loss(pred, target)


def fourier_loss(pred, target):
    """Spectral loss: L1 on log-magnitude of 2D FFT, keeps high-frequency detail."""
    def mag(x):
        f = torch.fft.rfft2(x, norm="ortho")
        return torch.log1p(torch.abs(f))
    return F.l1_loss(mag(pred), mag(target))


def perceptual_loss(pred, target, vgg):
    """VGG16 relu1_2 + relu2_2 perceptual loss (inputs in [0,1])."""
    def feats(x):
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        x = vgg(x)
        return x
    return F.l1_loss(feats(pred), feats(target))


def make_vgg():
    """VGG16 features up to relu2_2 (used only if config enables perceptual loss)."""
    from torchvision import models
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:9]
    for p in vgg.parameters():
        p.requires_grad_(False)
    return vgg.eval()
