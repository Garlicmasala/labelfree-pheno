"""Inference CLI: `python -m labelfree.infer --config configs/infer.yaml ...`."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .datasets import Caco2PairsDataset
from .models import AttResUNet
from .uncertainty import mc_dropout_infer
from .utils import get_device, load_config, save_tensor_image, set_seed

log = logging.getLogger("infer")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--mc-samples", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base", type=int, default=None, help="override model base width")
    ap.add_argument("--depth", type=int, default=None, help="override model depth")
    args = ap.parse_args()

    set_seed(42)
    device = get_device()
    cfg = load_config(args.config)
    m = cfg["model"]
    model = AttResUNet(in_ch=cfg["data"]["in_ch"], out_ch=cfg["data"]["out_ch"],
                       base=args.base if args.base else m["base"],
                       depth=args.depth if args.depth else m["depth"],
                       attn=m.get("attn", True),
                       dropout=float(m.get("dropout", 0.1))).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log.info("loaded %s (epoch=%s)", args.ckpt, ckpt.get("epoch"))

    ds = Caco2PairsDataset(args.csv, args.root, split=args.split, crop=args.crop)
    if args.limit:
        ds.df = ds.df.iloc[: args.limit]
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with torch.no_grad():
        for x, y, sid in loader:
            if isinstance(sid, (tuple, list)):
                sid = sid[0]
            x = x.to(device)
            pred, std = mc_dropout_infer(model, x, n=args.mc_samples)
            save_tensor_image(x[0].cpu(), out / f"{sid}_input.png")
            save_tensor_image(pred[0, 0:1].cpu(), out / f"{sid}_pred_green.png")
            save_tensor_image(pred[0, 1:2].cpu(), out / f"{sid}_pred_red.png")
            save_tensor_image(std[0, 0:1].cpu(), out / f"{sid}_std_green.png")
            save_tensor_image(std[0, 1:2].cpu(), out / f"{sid}_std_red.png")
            n += 1
            log.info("inferred %d/%d  %s", n, len(loader), sid)
    print(f"done: {n} images -> {out}")


if __name__ == "__main__":
    main()
