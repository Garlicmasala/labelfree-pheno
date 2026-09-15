"""Phenotype scoring and Hill dose-response fitting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .segment import cell_stats, dead_mask, viable_mask
from .utils import read_image_gray


def phenotype_field(green: np.ndarray, red: np.ndarray) -> dict:
    return cell_stats(viable_mask(green), dead_mask(red))


def phenotype_directory(pred_dir: str | Path, out_json: str | Path | None = None,
                        ch_green: str = "green", ch_red: str = "red") -> pd.DataFrame:
    pred_dir = Path(pred_dir)
    rows = []
    for p in sorted(pred_dir.glob(f"*_pred_{ch_green}.png")):
        sid = p.name.replace(f"_pred_{ch_green}.png", "")
        g = read_image_gray(p)
        r = read_image_gray(pred_dir / f"{sid}_pred_{ch_red}.png")
        s = phenotype_field(g, r)
        s["id"] = sid
        rows.append(s)
    df = pd.DataFrame(rows)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(df.to_dict("records"), indent=2))
    return df


def _hill(x, Emax, EC50, n):
    return Emax * x ** n / (EC50 ** n + x ** n)


def fit_hill(doses: np.ndarray, responses: np.ndarray):
    """Fit Emax*D^n/(EC50^n + D^n); returns dict or None."""
    doses = np.asarray(doses, float)
    responses = np.asarray(responses, float)
    if len(doses) < 4 or np.any(doses <= 0):
        return None
    try:
        p0 = [float(np.max(responses) - np.min(responses)), float(np.median(doses)), 1.0]
        popt, _ = curve_fit(_hill, doses, responses, p0=p0, maxfev=20000)
        Emax, EC50, n = popt
        rss = float(np.sum((_hill(doses, *popt) - responses) ** 2))
        tss = float(np.sum((responses - responses.mean()) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else 0.0
        return {"EC50": float(EC50), "Emax": float(Emax), "Hill_n": float(n), "R2": float(r2)}
    except Exception:
        return None
