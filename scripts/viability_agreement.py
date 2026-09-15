"""Viability agreement: predicted fluorescence -> viability vs real stained viability.

Usage:  python scripts/viability_agreement.py --run runs/caco2_mini
Writes runs/<name>/viability_agreement.json + figure to report/assets/viability_agreement.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labelfree.phenotype import phenotype_field          # noqa: E402
from labelfree.utils import read_image_gray              # noqa: E402


def viability_of(g, r):
    return phenotype_field(g, r)["viability"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/caco2_mini")
    ap.add_argument("--csv", default="data/caco2/pairs.csv")
    ap.add_argument("--root", default="data/caco2/raw")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(ROOT / "report" / "assets"),
                    help="output dir for the agreement figure")
    args = ap.parse_args()

    run = Path(args.run)
    df = pd.read_csv(args.csv)
    df = df[df["split"] == args.split]
    pred_dir = run / "infer"

    true_viab, pred_viab, ids = [], [], []
    for _, row in df.iterrows():
        sid = row["id"]
        p = pred_dir / f"{sid}_pred_green.png"
        if not p.exists():
            continue
        pred_viab.append(viability_of(
            read_image_gray(p), read_image_gray(pred_dir / f"{sid}_pred_red.png")))
        true_viab.append(viability_of(
            read_image_gray(f"{args.root}/{row['green']}"),
            read_image_gray(f"{args.root}/{row['red']}")))
        ids.append(sid)

    t = np.array(true_viab); v = np.array(pred_viab)
    pearson = float(np.corrcoef(t, v)[0, 1])
    from scipy.stats import spearmanr
    spearman = float(spearmanr(t, v).statistic)
    mae = float(np.mean(np.abs(t - v)))
    result = {"n": int(len(t)), "pearson": pearson, "spearman": spearman, "mae": mae}
    print(json.dumps(result, indent=2))
    (run / "viability_agreement.json").write_text(json.dumps(result, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(t, v, s=40, alpha=0.7, edgecolor="k", linewidth=0.5)
    lims = [min(t.min(), v.min()) - 0.05, max(t.max(), v.max()) + 0.05]
    ax.plot(lims, lims, "--", color="gray", label="y=x")
    ax.set_xlabel("viability from real stain")
    ax.set_ylabel("viability from virtual stain")
    ax.set_title(f"Pearson r={pearson:.3f}  Spearman ρ={spearman:.3f}  MAE={mae:.3f}")
    ax.legend()
    ax.set_xlim(lims); ax.set_ylim(lims)
    plt.tight_layout()
    out_png = Path(args.out) / "viability_agreement.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("saved", out_png)


if __name__ == "__main__":
    main()
