"""PatchGAN-style discriminator for adversarial training (optional)."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(8, base * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(8, base * 4), nn.LeakyReLU(0.2),
            nn.Conv2d(base * 4, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def hinge_d_loss(d_real, d_fake) -> float:
    """Hinge adversarial loss; returns Python float (sum of mean terms)."""
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def hinge_g_loss(d_fake):
    return -d_fake.mean()
