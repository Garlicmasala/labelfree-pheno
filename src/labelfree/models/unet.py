"""AttResUNet: attention-gated residual U-Net for image-to-image translation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two conv + group-norm + ReLU with a residual shortcut and MC-dropout."""

    def __init__(self, cin: int, cout: int, norm_groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(norm_groups, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(norm_groups, cout)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.relu(self.n1(self.conv1(x)))
        h = F.relu(self.n2(self.conv2(h)))
        h = self.drop(h)
        return h + self.shortcut(x)


class SelfAttn(nn.Module):
    """Spatial self-attention (query/key/value 1x1 convs) at deepest level."""

    def __init__(self, c: int):
        super().__init__()
        self.q = nn.Conv2d(c, c // 4, 1)
        self.k = nn.Conv2d(c, c // 4, 1)
        self.v = nn.Conv2d(c, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)     # [b, hw, c/4]
        k = self.k(x).flatten(2)                     # [b, c/4, hw]
        v = self.v(x).flatten(2).transpose(1, 2)     # [b, hw, c]
        attn = torch.softmax(q @ k / (c ** 0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return self.gamma * out + x


class AttResUNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 2, base: int = 32,
                 depth: int = 3, attn: bool = True, norm_groups: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.depth = depth
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        cin = in_ch
        for d in range(depth):
            cout = base * (2 ** d)
            self.enc.append(ConvBlock(cin, cout, norm_groups, dropout))
            if d < depth - 1:
                self.downs.append(nn.MaxPool2d(2))
            cin = cout
        self.attn = SelfAttn(cin) if attn else nn.Identity()
        self.dec = nn.ModuleList()
        self.ups = nn.ModuleList()
        for d in range(depth - 2, -1, -1):
            cout = base * (2 ** d)
            self.ups.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.dec.append(ConvBlock(cin + cout, cout, norm_groups, dropout))
            cin = cout
        self.head = nn.Conv2d(cin, out_ch, 1)

    def forward(self, x):
        skips = []
        h = x
        for d, blk in enumerate(self.enc):
            h = blk(h)
            if d < self.depth - 1:
                skips.append(h)
                h = self.downs[d](h)
        h = self.attn(h)
        for up, blk, sk in zip(self.ups, self.dec, reversed(skips)):
            h = up(h)
            h = torch.cat([h, sk], dim=1)
            h = blk(h)
        return self.head(h)
