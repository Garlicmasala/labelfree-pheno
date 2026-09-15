"""Download the Caco-2 Virtual-Staining dataset from figshare.

Article: Scientific Data 10:160 (2023), DOI 10.1038/s41597-023-02065-7
Usage:  python data/download_caco2.py --out data/caco2/raw
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

API = "https://api.figshare.com/v2/articles/21971558"
FILE_URL = "https://ndownloader.figshare.com/files/38982755"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/caco2/raw")
    ap.add_argument("--keep-zip", action="store_true", help="keep the zip after extraction")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out.parent / "2022-09-06_Dataset_Update.zip"

    if not zip_path.exists():
        print("downloading 506MB zip from figshare ...")
        with urlopen(FILE_URL) as resp, open(zip_path, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{100*done/total:.1f}%", end="", flush=True)
        print()

    print("extracting ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    if not args.keep_zip:
        zip_path.unlink()
        print("removed zip")
    dirs = sorted([p.name for p in out.iterdir() if p.is_dir()])
    print(f"extracted {len(dirs)} series folders under {out}:")
    for d in dirs[:12]:
        n = len(list((out / d).glob("*.jpg")))
        print(f"  {d}: {n} jpg")
    print("total jpg:", sum(len(list((out / d).glob('*.jpg'))) for d in out.iterdir() if d.is_dir()))


if __name__ == "__main__":
    main()
