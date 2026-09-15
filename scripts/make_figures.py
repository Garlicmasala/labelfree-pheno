"""Generate report figures from a finished run.

Usage:  python scripts/make_figures.py --run runs/caco2_mini --out report/assets
Produces <out>/caco2_montage.png and <out>/metrics.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labelfree.plot import metrics_bar, montage           # noqa: E402
from labelfree.utils import read_image_gray               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/caco2_mini")
    ap.add_argument("--out", default="report/assets")
    ap.add_argument("--root", default="data/caco2/raw")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred_dir = run / "infer"
    eval_sum = json.load(open(run / "eval" / "summary.json"))

    pairs = pd.read_csv(ROOT / "data" / "caco2" / "pairs.csv")
    path_of = dict(zip(pairs["id"], pairs["green"]))

    preds = sorted(pred_dir.glob("*_pred_green.png"))[: args.n]
    inputs, targets, pred_imgs, stds = [], [], [], []
    for p in preds:
        sid = p.name.replace("_pred_green.png", "")
        inputs.append(read_image_gray(pred_dir / f"{sid}_input.png"))
        targets.append(read_image_gray(ROOT / "data" / "caco2" / "raw" / path_of[sid]))
        pred_imgs.append(read_image_gray(p))
        s = pred_dir / f"{sid}_std_green.png"
        stds.append(read_image_gray(s) if s.exists() else None)
    has_std = all(s is not None for s in stds)
    montage(inputs, targets, pred_imgs, stds if has_std else None,
            n=len(inputs), path=str(out / "caco2_montage.png"))
    print("montage ->", out / "caco2_montage.png")
    metrics_bar(eval_sum, path=str(out / "metrics.png"))
    print("metrics  ->", out / "metrics.png")


if __name__ == "__main__":
    main()
