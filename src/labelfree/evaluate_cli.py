"""Evaluation CLI over saved predictions:
`python -m labelfree.evaluate --csv data/caco2/pairs.csv --root data/caco2
    --pred-dir runs/xxx/infer --split test --crop 256 --out runs/xxx/eval`
Writes summary.json (overall + per-channel PSNR/SSIM).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .utils import center_crop_np, read_image_gray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.split != "all":
        df = df[df["split"] == args.split]
    pred_dir = Path(args.pred_dir)

    acc = {"green": {"psnr": [], "ssim": []}, "red": {"psnr": [], "ssim": []}}
    n = 0
    for _, row in df.iterrows():
        sid = row["id"]
        target = read_image_gray(f"{args.root}/{row['green']}")
        if args.crop:
            target = center_crop_np(target, args.crop)
        for ch, col in (("green", row["green"]), ("red", row["red"])):
            pred_p = pred_dir / f"{sid}_pred_{ch}.png"
            if not pred_p.exists():
                continue
            pred = read_image_gray(pred_p)
            tgt = center_crop_np(read_image_gray(f"{args.root}/{col}"), args.crop) if args.crop \
                else read_image_gray(f"{args.root}/{col}")
            acc[ch]["psnr"].append(peak_signal_noise_ratio(tgt, pred, data_range=1.0))
            acc[ch]["ssim"].append(structural_similarity(tgt, pred, data_range=1.0))
        n += 1

    summary = {"n_samples": n, "split": args.split, "crop": args.crop, "channels": {}}
    all_psnr, all_ssim = [], []
    for ch, d in acc.items():
        if d["psnr"]:
            summary["channels"][ch] = {"psnr": float(np.mean(d["psnr"])),
                                       "ssim": float(np.mean(d["ssim"]))}
            all_psnr += d["psnr"]
            all_ssim += d["ssim"]
    if all_psnr:
        summary["overall"] = {"psnr": float(np.mean(all_psnr)), "ssim": float(np.mean(all_ssim))}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
