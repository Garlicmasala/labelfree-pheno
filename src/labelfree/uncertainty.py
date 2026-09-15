"""MC-dropout uncertainty estimation for the translator."""
from __future__ import annotations

import torch


@torch.no_grad()
def mc_dropout_infer(model, x, n: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean_pred, std_pred) over n stochastic forward passes.

    The model must contain dropout layers; dropout is active in eval via
    model.train() so `n` samples approximate the predictive posterior.
    """
    model.train()
    preds = []
    for _ in range(n):
        preds.append(model(x).unsqueeze(0))
    model.eval()
    stack = torch.cat(preds, 0)          # [n, C, H, W]
    mean = stack.mean(0)
    std = stack.std(0, unbiased=False)
    return mean, std
