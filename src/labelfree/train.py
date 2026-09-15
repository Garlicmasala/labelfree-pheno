"""Training CLI: `python -m labelfree.train --config configs/train_*.yaml`."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets import Caco2PairsDataset, SyntheticPairsDataset
from .losses import fourier_loss, l1_loss, make_vgg, perceptual_loss
from .models import AttResUNet
from .utils import get_device, load_config, set_seed
from .evaluate import eval_metrics


def build_datasets(cfg):
    if cfg["data"].get("synthetic", False):
        n = cfg["data"].get("n", 64)
        train = SyntheticPairsDataset(n=n, size=cfg["data"]["crop"], seed=0)
        val = SyntheticPairsDataset(n=max(8, n // 8), size=cfg["data"]["crop"], seed=1)
        return train, val
    d = cfg["data"]
    train = Caco2PairsDataset(d["pairs_csv"], d["root"], split="train", crop=d["crop"])
    val = Caco2PairsDataset(d["pairs_csv"], d["root"], split="val", crop=d["crop"])
    if d.get("limit"):
        train = torch.utils.data.Subset(train, list(range(min(d["limit"], len(train)))))
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device()
    out = Path(args.out or cfg["out"])
    out.mkdir(parents=True, exist_ok=True)

    m = cfg["model"]
    model = AttResUNet(in_ch=cfg["data"]["in_ch"], out_ch=cfg["data"]["out_ch"],
                       base=m["base"], depth=m["depth"], attn=m.get("attn", True),
                       dropout=float(m.get("dropout", 0.1))).to(device)
    print(f"model params={sum(p.numel() for p in model.parameters())/1e6:.2f}M device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]),
                            weight_decay=float(cfg["train"].get("wd", 0.0)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["train"]["epochs"])

    train_ds, val_ds = build_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch"],
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    vgg = make_vgg().to(device) if cfg["train"].get("perceptual") else None
    w_f = float(cfg["train"].get("w_fourier", 0.0))
    w_p = float(cfg["train"].get("w_perceptual", 0.0))

    log_path = out / "train_log.csv"
    best_psnr = -1.0
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "time_s", "loss_total", "loss_l1", "val_psnr", "val_ssim"])

        for epoch in range(1, cfg["train"]["epochs"] + 1):
            t0 = time.time()
            model.train()
            total_loss, total_l1, n_batch = 0.0, 0.0, 0
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = l1_loss(pred, y)
                if w_f > 0:
                    loss = loss + w_f * fourier_loss(pred, y)
                if w_p > 0:
                    loss = loss + w_p * perceptual_loss(pred, y, vgg)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                total_l1 += l1_loss(pred, y).item()
                n_batch += 1
            sched.step()

            val_psnr, val_ssim = eval_metrics(model, val_loader, device)
            elapsed = time.time() - t0
            print(f"epoch {epoch:02d} loss={total_loss/n_batch:.4f} l1={total_l1/n_batch:.4f} "
                  f"val_psnr={val_psnr:.3f} val_ssim={val_ssim:.4f} ({elapsed:.0f}s)")
            writer.writerow([epoch, round(elapsed, 1), round(total_loss / n_batch, 5),
                             round(total_l1 / n_batch, 5), round(val_psnr, 4), round(val_ssim, 4)])
            f.flush()
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                            "val_psnr": val_psnr, "val_ssim": val_ssim},
                           out / "best.ckpt")
    print(f"done -> {out} (best val PSNR {best_psnr:.3f})")


if __name__ == "__main__":
    main()
