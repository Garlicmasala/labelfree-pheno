"""Generate a synthetic paired dataset (smoke tests / CI).

Usage:  python data/synthetic.py --out data/synthetic --n 64 --size 128
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def make_pair(rng, size: int, dead_frac: float = 0.3):
    from scipy.ndimage import gaussian_filter
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    bf = rng.uniform(0.25, 0.45, (size, size)).astype(np.float32)
    green = np.zeros((size, size), np.float32)
    red = np.zeros((size, size), np.float32)
    for _c in range(int(rng.integers(10, 20))):
        cx, cy = rng.uniform(15, size - 15, 2)
        r = rng.uniform(4, 9)
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
        bf[mask] -= rng.uniform(0.08, 0.18)
        if rng.random() < dead_frac:
            ring = ((xx - cx) ** 2 + (yy - cy) ** 2)
            ring = (ring < (r + 2) ** 2) & ~mask
            red[ring] = rng.uniform(0.5, 0.95)
        else:
            green[mask] = rng.uniform(0.25, 0.6)
    bf = gaussian_filter(bf, 1.2)
    green = gaussian_filter(green, 1.0)
    red = gaussian_filter(red, 0.8)
    bf = (bf + rng.normal(0, 0.01, (size, size))).clip(0, 1)
    green = (green + rng.normal(0, 0.01, (size, size))).clip(0, 1)
    red = (red + rng.normal(0, 0.01, (size, size))).clip(0, 1)
    return bf, green, red


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        bf, green, red = make_pair(rng, args.size)
        Image.fromarray((bf * 255).clip(0, 255).astype(np.uint8)).save(out / f"{i}.0.jpg")
        Image.fromarray((green * 255).clip(0, 255).astype(np.uint8)).save(out / f"{i}.1.jpg")
        Image.fromarray((red * 255).clip(0, 255).astype(np.uint8)).save(out / f"{i}.2.jpg")
    print(f"wrote {args.n} synthetic triples to {out}")


if __name__ == "__main__":
    main()
