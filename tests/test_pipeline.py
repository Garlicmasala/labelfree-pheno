"""Unit tests: dataset integrity, model shapes, losses, hill fit.
Usage:  python -m pytest tests/test_pipeline.py  (or python tests/test_pipeline.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labelfree.datasets import SyntheticPairsDataset      # noqa: E402
from labelfree.losses import fourier_loss, l1_loss        # noqa: E402
from labelfree.models import AttResUNet                   # noqa: E402
from labelfree.phenotype import fit_hill                  # noqa: E402


def test_model_forward_shape():
    m = AttResUNet(in_ch=1, out_ch=2, base=8, depth=3, attn=True)
    x = torch.randn(2, 1, 64, 64)
    y = m(x)
    assert y.shape == (2, 2, 64, 64), y.shape


def test_synthetic_dataset():
    ds = SyntheticPairsDataset(n=16, size=64, seed=0)
    x, y, sid = ds[0]
    assert x.shape == (1, 64, 64) and y.shape == (2, 64, 64)
    assert 0 <= x.min() and x.max() <= 1
    assert sid == "synth_0"


def test_losses_finite():
    p = torch.rand(2, 2, 32, 32) * 0.5
    t = torch.rand(2, 2, 32, 32) * 0.5
    assert torch.isfinite(l1_loss(p, t))
    assert torch.isfinite(fourier_loss(p, t))


def test_fit_hill():
    doses = np.array([0.1, 1.0, 10.0, 100.0])
    emax, ec50, n = 0.8, 5.0, 1.5
    resp = emax * doses ** n / (ec50 ** n + doses ** n)
    fit = fit_hill(doses, resp)
    assert fit is not None
    assert abs(fit["EC50"] - ec50) / ec50 < 0.2
    assert fit["R2"] > 0.9


if __name__ == "__main__":
    for fn in (test_model_forward_shape, test_synthetic_dataset,
               test_losses_finite, test_fit_hill):
        fn()
        print("PASS", fn.__name__)
    print("ALL TESTS PASS")
