"""Build paired field table (pairs.csv) from the raw Caco-2 dataset.

Each field has 3 grayscale JPEGs (BF / green / red). Channel roles are
auto-detected from mean image intensity: green (calcein) is mid-bright,
red (PI nuclei) is darkest on average in these data; the brightest channel
is bright-field. If detection disagrees with the folder/camera name, set
`--channels bf,green,red` explicitly (suffixes are parsed from filenames).

Usage:  python data/make_caco2_pairs.py --root data/caco2/raw --out data/caco2/pairs.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image


def _mean_gray(p: Path) -> float:
    with Image.open(p) as im:
        a = np.asarray(im.convert("L"), dtype=np.float32)
    return float(a.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/caco2/raw")
    ap.add_argument("--out", default="data/caco2/pairs.csv")
    ap.add_argument("--channels", default=None,
                    help="explicit mapping e.g. 'bf,green,red' from suffix group order")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for series in sorted(root.iterdir()):
        if not series.is_dir():
            continue
        # group files by field index; suffixes look like "<k>.0.jpg|<k>.1.jpg|..."
        files = sorted(series.glob("*.jpg"))
        groups: dict[int, list[tuple[int, Path]]] = {}
        for p in files:
            m = re.match(r"(\d+)\.(\d+)\.jpg$", p.name)
            if not m:
                continue
            k, s = int(m.group(1)), int(m.group(2))
            groups.setdefault(k, []).append((s, p))
        for k in sorted(groups):
            suff = sorted(groups[k])                      # by suffix
            if len(suff) < 3:
                continue
            sid = f"{series.name}__{k}"
            if args.channels:
                order = [s for s, _ in suff]
                cmap = dict(zip([x.strip() for x in args.channels.split(",")], order))
                bf_s, g_s, r_s = cmap["bf"], cmap["green"], cmap["red"]
            else:
                means = {s: _mean_gray(p) for s, p in suff}
                order = sorted(means, key=means.get, reverse=True)   # brightest -> darkest
                bf_s, g_s, r_s = order[0], order[1], order[2]
            rows.append({
                "id": sid, "series": series.name,
                "bf": f"{series.name}/{suff[bf_s][1].name}",
                "green": f"{series.name}/{suff[g_s][1].name}",
                "red": f"{series.name}/{suff[r_s][1].name}",
            })

    df = pd.DataFrame(rows)
    # Slide-stratified split: 70/15/15 train/val/test (whole series stay together)
    rng = np.random.default_rng(42)
    series = sorted(df["series"].unique())
    rng.shuffle(series)
    n_tr = max(1, int(round(len(series) * 0.70)))
    n_va = max(1, int(round(len(series) * 0.15)))
    split_of = {s: "train" for s in series[:n_tr]}
    split_of.update({s: "val" for s in series[n_tr:n_tr + n_va]})
    split_of.update({s: "test" for s in series[n_tr + n_va:]})
    df["split"] = df["series"].map(split_of)
    print(f"paired {len(df)} fields; channel auto-map used "
          f"(set --channels if wrong). Split: {df['split'].value_counts().to_dict()}")
    print(df.head(3).to_string())
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
