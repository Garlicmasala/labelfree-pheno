"""Datasets for paired bright-field / fluorescence virtual staining."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import center_crop_np, read_image_gray


class Caco2PairsDataset(Dataset):
    """Paired (BF, green, red) fields from the Caco-2 Virtual-Staining dataset.

    pairs.csv columns: id, bf, green, red, split
    All image paths are relative to `root` and are grayscale JPEGs.
    """

    def __init__(self, pairs_csv: str, root: str, split: str = "train",
                 crop: int = 192, input_ch: int = 1, target_ch: int = 2):
        self.df = pd.read_csv(pairs_csv)
        if split != "all":
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.root = root
        self.crop = crop
        self.input_ch = input_ch
        self.target_ch = target_ch

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, rel: str) -> np.ndarray:
        img = read_image_gray(f"{self.root}/{rel}")
        if self.crop and self.crop > 0:
            img = center_crop_np(img, self.crop)
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        bf = self._load(row["bf"])
        green = self._load(row["green"])
        red = self._load(row["red"])
        x = torch.from_numpy(bf)[None].float()                       # [1,H,W]
        y = torch.from_numpy(np.stack([green, red], 0)).float()      # [2,H,W]
        return x, y, row["id"]


class SyntheticPairsDataset(Dataset):
    """Synthetic paired data for smoke tests / pipeline CI.

    BF: blurred blobs + Gaussian noise (nucleus-like dark spots).
    Green: viable-cell cytoplasmic signal (soft blobs where nuclei intact).
    Red: dead-cell signal (bright ring/blob on a subset of nuclei).
    """

    def __init__(self, n: int = 64, size: int = 128, seed: int = 0,
                 dead_frac: float = 0.3):
        self.n = n
        self.size = size
        rng = np.random.default_rng(seed)
        self.dead_frac = dead_frac
        self._precompute(rng)

    def _precompute(self, rng):
        self.bfs, self.greens, self.reds = [], [], []
        for _ in range(self.n):
            size = self.size
            yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
            bf = rng.uniform(0.25, 0.45, (size, size)).astype(np.float32)
            green = np.zeros((size, size), np.float32)
            red = np.zeros((size, size), np.float32)
            n_cells = int(rng.integers(10, 20))
            for _c in range(n_cells):
                cx, cy = rng.uniform(15, size - 15, 2)
                r = rng.uniform(4, 9)
                mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
                bf[mask] -= rng.uniform(0.08, 0.18)          # dark nucleus
                if rng.random() < self.dead_frac:           # dead: red PI ring
                    ring = ((xx - cx) ** 2 + (yy - cy) ** 2)
                    ring = (ring < (r + 2) ** 2) & ~mask
                    red[ring] = rng.uniform(0.5, 0.95)
                else:                                       # viable: green cytosol
                    green[mask] = rng.uniform(0.25, 0.6)
            # focal blur & sensor noise
            from scipy.ndimage import gaussian_filter
            bf = gaussian_filter(bf, 1.2)
            green = gaussian_filter(green, 1.0)
            red = gaussian_filter(red, 0.8)
            bf = (bf + rng.normal(0, 0.01, (size, size))).clip(0, 1)
            green = (green + rng.normal(0, 0.01, (size, size))).clip(0, 1)
            red = (red + rng.normal(0, 0.01, (size, size))).clip(0, 1)
            self.bfs.append(bf.astype(np.float32))
            self.greens.append(green.astype(np.float32))
            self.reds.append(red.astype(np.float32))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.bfs[idx])[None].float()
        y = torch.from_numpy(np.stack([self.greens[idx], self.reds[idx]], 0)).float()
        return x, y, f"synth_{idx}"
