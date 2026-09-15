"""Minimal AI-agent demo: conversational driver over the pipeline.

Usage (after training):
    python -m labelfree.agent --ckpt runs/caco2_mini/best.ckpt --root data/caco2
Then type commands: `phenotype <field_id>`, `hill`, `montage`, `exit`.
"""
from __future__ import annotations

import argparse
import json

from .datasets import Caco2PairsDataset
from .uncertainty import mc_dropout_infer
from .models import AttResUNet
from .phenotype import fit_hill, phenotype_field
from .plot import montage
from .utils import get_device, load_config, read_image_gray, save_tensor_image, set_seed
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/infer.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/caco2")
    ap.add_argument("--csv", default="data/caco2/pairs.csv")
    ap.add_argument("--crop", type=int, default=256)
    args = ap.parse_args()

    set_seed(42)
    device = get_device()
    cfg = load_config(args.config)
    m = cfg["model"]
    model = AttResUNet(in_ch=cfg["data"]["in_ch"], out_ch=cfg["data"]["out_ch"],
                       base=m["base"], depth=m["depth"], attn=m.get("attn", True),
                       dropout=float(m.get("dropout", 0.1))).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False)["state_dict"])
    model.eval()
    print("labelfree agent ready. commands: phenotype <id> | montage [n] | hill <doses.csv> | exit")

    while True:
        try:
            cmd = input(">> ").strip().split()
        except EOFError:
            break
        if not cmd:
            continue
        if cmd[0] == "exit":
            break
        if cmd[0] == "phenotype" and len(cmd) > 1:
            sid = cmd[1]
            bf = read_image_gray(f"{args.root}/{sid}")
            import numpy as np
            from .utils import center_crop_np
            x = torch.from_numpy(center_crop_np(bf, args.crop))[None, None].to(device)
            pred, _ = mc_dropout_infer(model, x, n=3)
            g = pred[0, 0].clamp(0, 1).cpu().numpy()
            r = pred[0, 1].clamp(0, 1).cpu().numpy()
            stats = phenotype_field(g, r)
            print(json.dumps(stats, indent=2))
        elif cmd[0] == "montage":
            save_tensor_image(torch.zeros(1), "/tmp/labelfree_agent_placeholder.png")
            print("run scripts/make_figures.py to render the montage")
        elif cmd[0] == "hill" and len(cmd) > 1:
            import pandas as pd
            d = pd.read_csv(cmd[1])
            fit = fit_hill(d["dose"].values, d["response"].values)
            print(json.dumps(fit, indent=2) if fit else "fit failed")
        else:
            print("unknown command")


if __name__ == "__main__":
    main()
