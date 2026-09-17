"""Shared utilities: IO, seeds, config, transforms."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """TPU (torch_xla) first, then CUDA, then CPU."""
    try:
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        if dev is not None:
            return dev
    except Exception:
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mark_step(device: torch.device) -> None:
    """Flush a lazy PyTorch/XLA step when running on TPU; no-op otherwise."""
    if str(device).startswith("xla"):
        try:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        except Exception:
            pass


def read_image_gray(path: str | Path) -> np.ndarray:
    """Read a grayscale image as float32 in [0, 1]."""
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def read_image_rgb(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def save_tensor_image(t: torch.Tensor, path: str | Path) -> None:
    """Save a [1,H,W] or [3,H,W] float tensor in [0,1] as uint8 PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    Image.fromarray(arr).save(path)


def center_crop_np(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[top : top + size, left : left + size]


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT])
