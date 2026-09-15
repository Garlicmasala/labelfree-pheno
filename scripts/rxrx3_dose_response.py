"""Module C — RxRx3-core dose-response phenomics (CPU, ~5 min with data cached).

Downloads metadata + OpenPhenom embeddings, computes per-well within-plate
perturbation scores, aggregates per (compound, dose), fits Hill curves, and
writes runs/rxrx3/{dose_response_curves.csv, dose_response_fits.json, control_sanity.png}.

Usage:  python scripts/rxrx3_dose_response.py --out runs/rxrx3
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

HF_REPO = "recursionpharma/rxrx3-core"
META_FILE = "metadata_rxrx3_core.csv"
EMB_FILE = "OpenPhenom_rxrx3_core_embeddings.parquet"


def load_hf(filename: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(HF_REPO, filename, repo_type="dataset"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/rxrx3")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading metadata + embeddings ...")
    meta_path = load_hf(META_FILE)
    emb_path = load_hf(EMB_FILE)
    meta = pd.read_csv(meta_path)
    emb = pd.read_parquet(emb_path)
    print(f"      metadata: {len(meta)} rows | embeddings: {len(emb)} rows")

    id_col = "well_id" if "well_id" in meta.columns else meta.columns[0]
    emb_id_col = "well_id" if "well_id" in emb.columns else emb.columns[0]
    df = meta.merge(emb, left_on=id_col, right_on=emb_id_col, how="inner", suffixes=("", "_e"))
    # keep only compound wells (drop CRISPR)
    if "perturbation_type" in df.columns:
        df = df[df["perturbation_type"] == "COMPOUND"].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c.startswith("feature_") or c.startswith("feat_")
                 or c.startswith("emb_")]
    if not feat_cols:
        feat_cols = [c for c in df.columns
                     if df[c].dtype in (np.float32, np.float64) and c != id_col]
    print(f"      wells after COMPOUND filter: {len(df)}; features used: {len(feat_cols)}")

    compound_col = next((c for c in ("treatment", "compound", "compound_id", "inchi_key") if c in df.columns), None)
    dose_col = next((c for c in ("concentration", "dose", "dose_uM") if c in df.columns), None)
    plate_col = next((c for c in ("plate", "plate_id", "experiment_plate") if c in df.columns), None)
    ctrl_col = "well_type_label" if "well_type_label" in df.columns else None
    print(f"      columns: compound={compound_col} dose={dose_col} plate={plate_col} ctrl={ctrl_col}")
    if not (compound_col and dose_col and plate_col):
        raise SystemExit("column layout not recognized — inspect df.columns and adapt")

    F = df[feat_cols].to_numpy(np.float32)

    print("[2/4] within-plate control-normalized perturbation scores ...")
    ctrl_mask = df[compound_col].astype(str).str.lower().str.contains(
        "control|dmso|empty", na=False)
    rows = []
    for plate, g in df.groupby(plate_col):
        cm = ctrl_mask.loc[g.index]
        if cm.sum() < 5:
            cm = pd.Series([False] * len(g))
        centroid = F[g.index[cm.values]].mean(0) if cm.any() else F[g.index].mean(0)
        scores = np.linalg.norm(F[g.index] - centroid, axis=1)
        for i, idx in enumerate(g.index):
            rows.append({"plate": int(plate), "well": df.loc[idx, id_col],
                         "compound": df.loc[idx, compound_col],
                         "dose": float(df.loc[idx, dose_col]), "score": float(scores[i])})
    scores = pd.DataFrame(rows)

    print("[3/4] sanity check: negative controls near zero ...")
    ctrl_scores = scores[scores["compound"].astype(str).str.lower().str.contains(
        "control|dmso|empty", na=False)]
    treat_scores = scores[~scores["compound"].astype(str).str.lower().str.contains(
        "control|dmso|empty", na=False)]
    print(f"      control score mean={ctrl_scores['score'].mean():.4f}  "
          f"treated score mean={treat_scores['score'].mean():.4f}")
    assert ctrl_scores["score"].mean() < treat_scores["score"].mean(), "sanity check failed"

    print("[4/4] dose-response aggregation + Hill fit ...")
    from labelfree.phenotype import fit_hill
    agg = scores.groupby(["compound", "dose"])["score"].median().reset_index()
    fits = {}
    for compound, g in agg.groupby("compound"):
        if len(g) < 4:
            continue
        fit = fit_hill(g["dose"].values, g["score"].values)
        if fit:
            fits[compound] = {**fit, "n_doses": int(len(g)), "max_score": float(g["score"].max())}
    fits_sorted = sorted(fits.items(), key=lambda kv: kv[1]["max_score"], reverse=True)
    print(f"      fitted compounds: {len(fits_sorted)}")
    for name, f in fits_sorted[:5]:
        print(f"        {name}: EC50={f['EC50']:.3f} Emax={f['Emax']:.3f} R2={f['R2']:.3f}")

    agg.to_csv(out / "dose_response_curves.csv", index=False)
    with open(out / "dose_response_fits.json", "w") as f:
        json.dump({k: v for k, v in fits_sorted}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ctrl_scores["score"], bins=40, alpha=0.6,
            label=f"controls (n={len(ctrl_scores)}, μ={ctrl_scores['score'].mean():.3f})")
    ax.hist(treat_scores["score"], bins=60, alpha=0.6,
            label=f"treatments (n={len(treat_scores)}, μ={treat_scores['score'].mean():.3f})")
    ax.set_xlabel("within-plate perturbation score")
    ax.set_ylabel("wells")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "control_sanity.png", dpi=150)
    plt.close(fig)

    # example Hill curve for the strongest hit
    if fits_sorted:
        top = fits_sorted[0][0]
        g = agg[agg["compound"] == top].sort_values("dose")
        f = fits_sorted[0][1]
        xx = np.logspace(np.log10(g["dose"].min()), np.log10(g["dose"].max()), 100)
        yy = f["Emax"] * xx ** f["Hill_n"] / (f["EC50"] ** f["Hill_n"] + xx ** f["Hill_n"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(g["dose"], g["score"], "o", label=top)
        ax.plot(xx, yy, "-", label=f"Hill fit: EC50={f['EC50']:.3f} Emax={f['Emax']:.3f} R2={f['R2']:.3f}")
        ax.set_xscale("log"); ax.set_xlabel("dose (μM)")
        ax.set_ylabel("perturbation score"); ax.legend()
        plt.tight_layout()
        plt.savefig(out / "hill_example.png", dpi=150)
        plt.close(fig)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
