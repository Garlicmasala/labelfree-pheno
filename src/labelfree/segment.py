"""Segmentation: label-free cell/dead masks from predicted fluorescence.

Adaptive percentile thresholds make the readout robust to the compressed
dynamic range of virtual stains (and to any microscope).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import measure, morphology


def _threshold(img: np.ndarray, p: float, k: float, floor: float) -> float:
    return max(floor, float(np.percentile(img, p)) * k)


def dead_mask(red: np.ndarray, p: float = 99.0, k: float = 0.75,
              min_size: int = 30, floor: float = 0.05) -> np.ndarray:
    """Dead-cell mask: PI-positive nuclei after smoothing + adaptive threshold."""
    r = ndimage.gaussian_filter(red, 1.0)
    m = r > _threshold(r, p, k, floor)
    m = morphology.remove_small_objects(m, max_size=max(1, min_size - 1))
    m = ndimage.binary_closing(m, iterations=1)
    return m


def viable_mask(green: np.ndarray, p: float = 99.0, k: float = 0.75,
                min_size: int = 60, floor: float = 0.05) -> np.ndarray:
    g = ndimage.gaussian_filter(green, 1.2)
    m = g > _threshold(g, p, k, floor)
    m = morphology.remove_small_objects(m, max_size=max(1, min_size - 1))
    return m


def cell_stats(viable: np.ndarray, dead: np.ndarray) -> dict:
    n_dead = len(np.unique(measure.label(dead))) - 1
    n_viable = len(np.unique(measure.label(viable))) - 1
    dead_area = float(dead.mean())
    viable_area = float(viable.mean())
    return {
        "n_dead": int(n_dead), "n_viable": int(n_viable),
        "dead_area_frac": dead_area, "viable_area_frac": viable_area,
        "viability": float(1.0 - min(1.0, n_dead / max(1, n_dead + n_viable))),
    }
