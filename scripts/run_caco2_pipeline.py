"""One-command reproduction of the full Caco-2 pipeline.

    python scripts/run_caco2_pipeline.py --epochs 8 --limit 120

Steps: download -> pairs -> train (mini) -> infer (test) -> evaluate -> phenotype.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=120, help="train fields cap (0 = all)")
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--mc", type=int, default=3)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    if not args.skip_download and not (ROOT / "data/caco2/raw").exists():
        run(["python", "data/download_caco2.py", "--out", "data/caco2/raw"])
    if not (ROOT / "data/caco2/pairs.csv").exists():
        run(["python", "data/make_caco2_pairs.py", "--root", "data/caco2/raw",
             "--out", "data/caco2/pairs.csv"])

    cfg = ROOT / "runs" / "caco2_mini" / "config.yaml"
    run(["python", "-m", "labelfree.train", "--config", "configs/train_caco2_mini.yaml",
         "--out", "runs/caco2_mini"])
    run(["python", "-m", "labelfree.infer", "--config", "configs/infer.yaml",
         "--ckpt", "runs/caco2_mini/best.ckpt",
         "--csv", "data/caco2/pairs.csv", "--root", "data/caco2/raw",
         "--split", "test", "--crop", str(args.crop), "--mc-samples", str(args.mc),
         "--out", "runs/caco2_mini/infer"])
    run(["python", "-m", "labelfree.evaluate_cli",
         "--csv", "data/caco2/pairs.csv", "--root", "data/caco2/raw",
         "--pred-dir", "runs/caco2_mini/infer", "--split", "test", "--crop", str(args.crop),
         "--out", "runs/caco2_mini/eval"])
    run(["python", "-c",
         "import sys; sys.path.insert(0,'src'); "
         "from labelfree.phenotype import phenotype_directory; "
         "phenotype_directory('runs/caco2_mini/infer', 'runs/caco2_mini/phenotype.json')"])
    print(json.dumps(json.load(open(ROOT / "runs/caco2_mini/eval/summary.json")), indent=2))
    print("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
