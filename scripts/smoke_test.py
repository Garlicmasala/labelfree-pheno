"""End-to-end smoke test on synthetic data (CPU, ~3-5 min).

Runs: data -> train -> infer -> evaluate, asserts thresholds.
Usage:  python scripts/smoke_test.py
Exit code 0 = PASS.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    n = 64
    run(["python", "data/synthetic.py", "--out", "data/synthetic", "--n", str(n), "--size", "64"])

    rows = []
    for i in range(n):
        split = "train" if i < int(n * 0.8) else ("val" if i < int(n * 0.9) else "test")
        rows.append({"id": f"synth_{i}", "series": "synth",
                     "bf": f"../synthetic/{i}.0.jpg", "green": f"../synthetic/{i}.1.jpg",
                     "red": f"../synthetic/{i}.2.jpg", "split": split})
    pd.DataFrame(rows).to_csv("data/synthetic/pairs.csv", index=False)

    run(["python", "-m", "labelfree.train", "--config", "configs/train_smoke.yaml",
         "--out", "runs/smoke"])
    run(["python", "-m", "labelfree.infer", "--config", "configs/infer.yaml",
         "--ckpt", "runs/smoke/best.ckpt", "--base", "16", "--depth", "3",
         "--csv", "data/synthetic/pairs.csv", "--root", "data/synthetic",
         "--split", "test", "--crop", "64", "--mc-samples", "2",
         "--out", "runs/smoke/infer"])
    run(["python", "-m", "labelfree.evaluate_cli",
         "--csv", "data/synthetic/pairs.csv", "--root", "data/synthetic",
         "--pred-dir", "runs/smoke/infer", "--split", "test", "--crop", "64",
         "--out", "runs/smoke/eval"])

    summary = json.load(open(ROOT / "runs/smoke/eval/summary.json"))
    ov = summary["overall"]
    print(f"overall PSNR={ov['psnr']:.2f}  SSIM={ov['ssim']:.3f}")
    assert ov["psnr"] >= 15.0, "PSNR too low"
    assert ov["ssim"] >= 0.30, "SSIM too low"
    print("SMOKE TEST PASS")


if __name__ == "__main__":
    sys.exit(main())
