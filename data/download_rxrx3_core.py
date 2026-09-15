"""Download RxRx3-core metadata + OpenPhenom embeddings (Module C extension).

Repo: recursionpharma/rxrx3-core on HuggingFace (CC BY 4.0).
Usage:  python data/download_rxrx3_core.py --out data/rxrx3
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/rxrx3")
    ap.add_argument("--metadata-only", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = ["metadata_rxrx3_core.csv"]
    if not args.metadata_only:
        files.append("OpenPhenom_rxrx3_core_embeddings.parquet")
    for f in files:
        p = hf_hub_download("recursionpharma/rxrx3-core", f, repo_type="dataset")
        print("downloaded", p)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
