"""Robust parallel downloader with per-chunk retry + zip integrity loop.

Usage:  python data/download_caco2_fast.py --out data/caco2/raw --threads 6 --max-attempts 3
"""
from __future__ import annotations

import argparse
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

FILE_URL = "https://ndownloader.figshare.com/files/38982755"
CHUNK = 4 * 1024 * 1024  # 4MB per chunk


def _probe_size(url: str) -> int:
    for attempt in range(5):
        try:
            req = Request(url, method="HEAD")
            with urlopen(req, timeout=60) as r:
                return int(r.headers["Content-Length"])
        except Exception:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("cannot probe file size")


def _fetch(url: str, start: int, end: int, fp, lock: threading.Lock, progress: list,
           nretry: int):
    data = None
    for attempt in range(nretry):
        try:
            req = Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != (end - start + 1):
                raise IOError(f"short read {len(data)} != {end - start + 1}")
            break
        except Exception as e:
            if attempt == nretry - 1:
                raise
            time.sleep(2 + attempt * 3)
    with lock:
        fp.seek(start)
        fp.write(data)
        progress[0] += len(data)
        print(f"\r{progress[0]/1048576:.1f}/{total_mb:.1f} MB", end="", flush=True)


total_mb = 0.0  # set in main


def _download(zip_path: Path, threads: int, nretry: int) -> None:
    global total_mb
    total = _probe_size(FILE_URL)
    total_mb = total / 1048576
    print(f"file size: {total_mb:.1f} MB, threads={threads}")
    with open(zip_path, "wb") as fp:
        fp.truncate(total)
        lock = threading.Lock()
        progress = [0]
        ranges = [(i, min(i + CHUNK - 1, total - 1)) for i in range(0, total, CHUNK)]
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = [ex.submit(_fetch, FILE_URL, s, e, fp, lock, progress, nretry)
                    for s, e in ranges]
            for f in as_completed(futs):
                f.result()   # raises if a chunk failed after retries
    print("\ndownload complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/caco2/raw")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max-attempts", type=int, default=4,
                    help="whole-file attempts (integrity loop)")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out.parent / "2022-09-06_Dataset_Update.zip"

    for attempt in range(args.max_attempts):
        print(f"=== attempt {attempt + 1}/{args.max_attempts} ===")
        try:
            _download(zip_path, args.threads, 5)
            with zipfile.ZipFile(zip_path) as z:
                bad = z.testzip()
                if bad:
                    raise IOError(f"bad member {bad}")
                z.extractall(out)
            print("extraction OK")
            if not args.keep_zip:
                zip_path.unlink()
            dirs = sorted([p.name for p in out.iterdir() if p.is_dir()])
            n_jpg = sum(len(list((out / d).glob("*.jpg"))) for d in out.iterdir() if d.is_dir())
            print(f"{len(dirs)} series folders, jpg total: {n_jpg}")
            return
        except Exception as e:
            print(f"attempt failed: {e}")
            time.sleep(3)
    raise SystemExit("download failed after all attempts")


if __name__ == "__main__":
    main()
